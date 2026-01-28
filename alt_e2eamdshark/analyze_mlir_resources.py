#!/usr/bin/env python3
"""
Utility script for analyzing MLIR resources in the testsuite workflow.

This script demonstrates how to integrate the MLIR resource extractors
into the existing testsuite for debugging and analysis.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import json

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlir_resource_extractor_simple import MLIRResourceExtractor


def analyze_test_run_resources(test_run_dir: str):
    """Analyze all MLIR files in a test run directory."""
    test_path = Path(test_run_dir)
    
    if not test_path.exists():
        print(f"Test run directory not found: {test_run_dir}")
        return
    
    # Find all MLIR files
    mlir_files = list(test_path.glob("**/*.mlir"))
    
    if not mlir_files:
        print(f"No MLIR files found in {test_run_dir}")
        return
    
    print(f"Found {len(mlir_files)} MLIR files in {test_run_dir}")
    print("=" * 60)
    
    all_resources = {}
    
    for mlir_file in mlir_files:
        print(f"\nAnalyzing: {mlir_file.name}")
        print("-" * 40)
        
        try:
            extractor = MLIRResourceExtractor(str(mlir_file))
            if extractor.extract_all():
                resources = extractor.list_resources()
                all_resources[mlir_file.name] = resources
                
                if resources:
                    print(f"Resources found: {len(resources)}")
                    for resource in resources[:5]:  # Show first 5
                        print(f"  - {resource}")
                    if len(resources) > 5:
                        print(f"  ... and {len(resources) - 5} more")
                else:
                    print("No resources found")
            else:
                print("Failed to extract resources")
                
        except Exception as e:
            print(f"Error processing {mlir_file.name}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_resources = sum(len(resources) for resources in all_resources.values())
    files_with_resources = len([r for r in all_resources.values() if r])
    
    print(f"Files processed: {len(mlir_files)}")
    print(f"Files with resources: {files_with_resources}")
    print(f"Total resources: {total_resources}")
    
    return all_resources


def extract_weights_for_debugging(mlir_file: str, output_dir: str = None):
    """Extract all weights from an MLIR file for debugging."""
    if output_dir is None:
        output_dir = "extracted_weights"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Extracting weights from: {mlir_file}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    extractor = MLIRResourceExtractor(mlir_file)
    if not extractor.extract_all():
        print("Failed to extract resources")
        return
    
    resources = extractor.list_resources()
    if not resources:
        print("No resources found")
        return
    
    print(f"Found {len(resources)} resources:")
    
    extracted_count = 0
    metadata = {}
    
    for resource_name in resources:
        try:
            tensor = extractor.get_resource(resource_name)
            if tensor is not None:
                # Save tensor
                output_file = output_path / f"{resource_name}.npy"
                np.save(output_file, tensor)
                
                # Save metadata
                metadata[resource_name] = {
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'size': int(tensor.size),
                    'min': float(tensor.min()),
                    'max': float(tensor.max()),
                    'mean': float(tensor.mean()),
                    'std': float(tensor.std()),
                    'file': str(output_file.name)
                }
                
                print(f"  ✓ {resource_name}: shape={tensor.shape}, range=[{tensor.min():.4f}, {tensor.max():.4f}]")
                extracted_count += 1
            else:
                print(f"  ✗ {resource_name}: extraction failed")
                
        except Exception as e:
            print(f"  ✗ {resource_name}: error - {e}")
    
    # Save metadata
    metadata_file = output_path / "weights_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nExtracted {extracted_count}/{len(resources)} resources")
    print(f"Metadata saved to: {metadata_file}")


def compare_model_weights(mlir_file1: str, mlir_file2: str):
    """Compare weights between two MLIR files (e.g., before/after optimization)."""
    print(f"Comparing weights between:")
    print(f"  File 1: {mlir_file1}")
    print(f"  File 2: {mlir_file2}")
    print()
    
    # Extract from both files
    extractors = [
        MLIRResourceExtractor(mlir_file1),
        MLIRResourceExtractor(mlir_file2)
    ]
    
    resources_lists = []
    for i, extractor in enumerate(extractors):
        if extractor.extract_all():
            resources_lists.append(set(extractor.list_resources()))
        else:
            print(f"Failed to extract from file {i+1}")
            return
    
    resources1, resources2 = resources_lists
    
    # Find common and different resources
    common = resources1 & resources2
    only_in_1 = resources1 - resources2
    only_in_2 = resources2 - resources1
    
    print(f"Resources in file 1: {len(resources1)}")
    print(f"Resources in file 2: {len(resources2)}")
    print(f"Common resources: {len(common)}")
    print(f"Only in file 1: {len(only_in_1)}")
    print(f"Only in file 2: {len(only_in_2)}")
    print()
    
    if only_in_1:
        print("Resources only in file 1:")
        for resource in sorted(only_in_1):
            print(f"  - {resource}")
        print()
    
    if only_in_2:
        print("Resources only in file 2:")
        for resource in sorted(only_in_2):
            print(f"  - {resource}")
        print()
    
    if common:
        print("Comparing common resources:")
        print("-" * 40)
        
        for resource_name in sorted(common):
            try:
                tensor1 = extractors[0].get_resource(resource_name)
                tensor2 = extractors[1].get_resource(resource_name)
                
                if tensor1 is not None and tensor2 is not None:
                    if tensor1.shape == tensor2.shape:
                        diff = np.abs(tensor1 - tensor2)
                        max_diff = diff.max()
                        mean_diff = diff.mean()
                        
                        if max_diff < 1e-6:
                            status = "IDENTICAL"
                        elif max_diff < 1e-3:
                            status = "VERY_SIMILAR"
                        elif max_diff < 1e-1:
                            status = "SIMILAR"
                        else:
                            status = "DIFFERENT"
                        
                        print(f"  {resource_name}: {status} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                    else:
                        print(f"  {resource_name}: SHAPE_MISMATCH ({tensor1.shape} vs {tensor2.shape})")
                else:
                    print(f"  {resource_name}: EXTRACTION_FAILED")
                    
            except Exception as e:
                print(f"  {resource_name}: ERROR - {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MLIR resources in testsuite workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all MLIR files in a test run
  python analyze_mlir_resources.py --analyze-test-run test-run/my_model_test
  
  # Extract all weights for debugging
  python analyze_mlir_resources.py --extract-weights model.torch_onnx.mlir
  
  # Compare weights between two files
  python analyze_mlir_resources.py --compare model1.mlir model2.mlir
        """
    )
    
    parser.add_argument(
        "--analyze-test-run",
        help="Analyze all MLIR files in a test run directory"
    )
    
    parser.add_argument(
        "--extract-weights",
        help="Extract all weights from an MLIR file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="extracted_weights",
        help="Output directory for extracted weights"
    )
    
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Compare weights between two MLIR files"
    )
    
    args = parser.parse_args()
    
    if args.analyze_test_run:
        analyze_test_run_resources(args.analyze_test_run)
    
    elif args.extract_weights:
        extract_weights_for_debugging(args.extract_weights, args.output_dir)
    
    elif args.compare:
        compare_model_weights(args.compare[0], args.compare[1])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()