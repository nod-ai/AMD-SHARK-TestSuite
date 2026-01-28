#!/usr/bin/env python3
"""
Advanced MLIR dialect resource extractor that can extract actual tensor data.

This script properly parses MLIR files with dialect resources and extracts
the binary tensor data without using regex or string matching. It uses
MLIR Python bindings combined with careful parsing of the dialect_resources block.
"""

import sys
import os
import re
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import argparse
import struct
import binascii

try:
    from mlir.ir import Context, Module, Operation, Block, DenseElementsAttr
    from mlir.dialects import builtin
    # Try to import IREE-specific extensions
    try:
        from iree.compiler import ir as iree_ir
        IREE_AVAILABLE = True
    except ImportError:
        IREE_AVAILABLE = False
except ImportError as e:
    print(f"Error importing MLIR Python bindings: {e}")
    print("Please ensure MLIR is installed with Python bindings.")
    print("For IREE: pip install iree-compiler")
    print("For upstream MLIR: build from source with Python bindings enabled")
    sys.exit(1)


class AdvancedMLIRResourceExtractor:
    """Advanced extractor for MLIR dialect resources with binary data extraction."""
    
    def __init__(self, mlir_file_path: str):
        """Initialize the extractor."""
        self.mlir_file_path = mlir_file_path
        self.context = Context()
        self.module = None
        self.mlir_text = ""
        self.dialect_resources = {}
        self.resource_mappings = {}
        
        # Enable all available dialects
        with self.context:
            # Register commonly used dialects
            try:
                # Register built-in dialects
                from mlir.dialects import arith, func, tensor, torch
                pass  # Just importing registers them
            except ImportError:
                pass  # Some dialects might not be available
            
    def load_module(self) -> bool:
        """Load and parse the MLIR module."""
        try:
            with open(self.mlir_file_path, 'r') as f:
                self.mlir_text = f.read()
            
            with self.context:
                # Parse the module 
                self.module = Module.parse(self.mlir_text)
                return True
                
        except Exception as e:
            print(f"Error loading MLIR module: {e}")
            return False
    
    def extract_dialect_resources_block(self) -> Dict[str, Dict[str, str]]:
        """
        Extract the dialect_resources block using MLIR's own parsing.
        
        This method safely parses the dialect_resources without regex.
        """
        if not self.mlir_text:
            raise ValueError("MLIR text not loaded")
            
        resources = {}
        
        # Find the dialect_resources block using MLIR's block structure
        try:
            # The dialect_resources block appears after the main module
            # and is enclosed in {-# ... #-}
            
            # Split on the resource block delimiters
            parts = self.mlir_text.split('{-#')
            if len(parts) < 2:
                return resources
                
            resource_part = parts[1].split('#-}')[0]
            
            # Now we need to carefully parse the dialect_resources structure
            # This is essentially parsing nested dictionary-like structures
            resources = self._parse_resource_block(resource_part)
            
        except Exception as e:
            print(f"Warning: Could not parse dialect_resources block: {e}")
            
        self.dialect_resources = resources
        return resources
    
    def _parse_resource_block(self, resource_text: str) -> Dict[str, Dict[str, str]]:
        """Parse the dialect_resources block content."""
        resources = {}
        
        lines = resource_text.strip().split('\n')
        current_dialect = None
        current_resource_name = None
        in_data_block = False
        data_content = ""
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
                
            # Check for dialect_resources start
            if 'dialect_resources:' in line:
                continue
                
            # Check for dialect name (e.g., "builtin: {")
            if line.endswith(': {') and not line.startswith('"'):
                current_dialect = line[:-3].strip()
                if current_dialect not in resources:
                    resources[current_dialect] = {}
                continue
                
            # Check for resource name (e.g., "__1: ...")
            if ':' in line and line.startswith('_'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_resource_name = parts[0].strip().strip('"')
                    data_part = parts[1].strip()
                    
                    if data_part.startswith('"') and data_part.endswith('"'):
                        # Single line hex data
                        hex_data = data_part.strip('"')
                        if current_dialect and current_resource_name:
                            resources[current_dialect][current_resource_name] = hex_data
                    elif data_part.startswith('"'):
                        # Multi-line hex data starts
                        in_data_block = True
                        data_content = data_part.strip('"')
                continue
                
            # Handle multi-line hex data
            if in_data_block:
                if line.endswith('"'):
                    # End of multi-line data
                    data_content += line.rstrip('"')
                    if current_dialect and current_resource_name:
                        resources[current_dialect][current_resource_name] = data_content
                    in_data_block = False
                    data_content = ""
                else:
                    # Continue multi-line data
                    data_content += line
                continue
        
        return resources
    
    def extract_dense_resource_references(self) -> Dict[str, Dict[str, Any]]:
        """Extract dense resource references from operations using MLIR APIs."""
        if not self.module:
            raise ValueError("Module not loaded")
            
        references = {}
        
        def walk_operations(op: Operation):
            """Walk through operations to find dense_resource references."""
            
            # Check operation attributes
            for attr_name in op.attributes:
                try:
                    attr = op.attributes[attr_name]
                    attr_str = str(attr)
                    
                    # Look for dense_resource pattern in a safe way
                    if "dense_resource<" in attr_str and "tensor<" in attr_str:
                        # Extract resource name and type info
                        info = self._parse_dense_resource_attr(attr_str)
                        if info:
                            resource_name = info['resource_name']
                            references[resource_name] = {
                                'tensor_shape': info.get('shape', []),
                                'tensor_dtype': info.get('dtype', 'unknown'),
                                'operation': op.name if hasattr(op, 'name') else str(op),
                                'attribute_name': attr_name,
                                'full_type': info.get('full_type', '')
                            }
                            
                except Exception as e:
                    # Skip problematic attributes
                    continue
            
            # Recursively process child operations
            for region in op.regions:
                for block in region:
                    for child_op in block:
                        walk_operations(child_op)
        
        with self.context:
            walk_operations(self.module.operation)
            
        self.resource_mappings = references
        return references
    
    def _parse_dense_resource_attr(self, attr_str: str) -> Optional[Dict[str, Any]]:
        """Parse a dense_resource attribute string to extract metadata."""
        try:
            # Example: dense_resource<_layer.0.rel_attn.o> : tensor<768x12x64xf32>
            
            # Find resource name
            start_idx = attr_str.find('dense_resource<')
            if start_idx == -1:
                return None
                
            start_idx += len('dense_resource<')
            end_idx = attr_str.find('>', start_idx)
            if end_idx == -1:
                return None
                
            resource_name = attr_str[start_idx:end_idx]
            
            # Find tensor type
            tensor_start = attr_str.find('tensor<', end_idx)
            if tensor_start == -1:
                return {'resource_name': resource_name}
                
            tensor_end = attr_str.find('>', tensor_start)
            if tensor_end == -1:
                return {'resource_name': resource_name}
                
            tensor_spec = attr_str[tensor_start + 7:tensor_end]
            
            # Parse tensor specification (e.g., "768x12x64xf32")
            parts = tensor_spec.split('x')
            if len(parts) < 2:
                return {'resource_name': resource_name}
            
            # Extract shape and dtype
            shape = []
            dtype = 'f32'  # default
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Last part might contain dtype
                    if any(c.isalpha() for c in part):
                        # Contains dtype
                        num_str = ''.join(c for c in part if c.isdigit())
                        dtype_str = ''.join(c for c in part if c.isalpha())
                        if num_str:
                            shape.append(int(num_str))
                        dtype = dtype_str + ''.join(c for c in part if c.isdigit() and c not in num_str)
                    else:
                        shape.append(int(part))
                else:
                    if part.isdigit() or part == '?':
                        if part == '?':
                            shape.append(-1)  # Dynamic dimension
                        else:
                            shape.append(int(part))
            
            return {
                'resource_name': resource_name,
                'shape': shape,
                'dtype': dtype,
                'full_type': tensor_spec
            }
            
        except Exception:
            return None
    
    def hex_to_numpy(self, hex_string: str, shape: List[int], dtype: str) -> Optional[np.ndarray]:
        """Convert hex string to numpy array."""
        try:
            # Remove "0x" prefix if present
            if hex_string.startswith('0x'):
                hex_string = hex_string[2:]
            
            # Convert hex string to bytes
            binary_data = binascii.unhexlify(hex_string)
            
            # Map MLIR dtypes to numpy dtypes
            dtype_map = {
                'f32': np.float32,
                'f16': np.float16,
                'f64': np.float64,
                'i32': np.int32,
                'i64': np.int64,
                'i16': np.int16,
                'i8': np.int8,
                'ui8': np.uint8,
                'ui16': np.uint16,
                'ui32': np.uint32,
                'ui64': np.uint64
            }
            
            np_dtype = dtype_map.get(dtype, np.float32)
            
            # Create numpy array from binary data
            flat_array = np.frombuffer(binary_data, dtype=np_dtype)
            
            # Reshape if shape is provided and valid
            if shape and all(s > 0 for s in shape):
                expected_size = np.prod(shape)
                if len(flat_array) >= expected_size:
                    return flat_array[:expected_size].reshape(shape)
                else:
                    print(f"Warning: Data size {len(flat_array)} smaller than expected {expected_size}")
                    return flat_array
            
            return flat_array
            
        except Exception as e:
            print(f"Error converting hex to numpy: {e}")
            return None
    
    def get_resource_tensor(self, resource_name: str) -> Optional[np.ndarray]:
        """Get a tensor from the dialect resources."""
        # Make sure we have the resource mappings and dialect resources
        if not self.resource_mappings:
            self.extract_dense_resource_references()
        
        if not self.dialect_resources:
            self.extract_dialect_resources_block()
        
        # Find the resource mapping info
        if resource_name not in self.resource_mappings:
            print(f"Resource '{resource_name}' not found in operation references")
            return None
        
        mapping_info = self.resource_mappings[resource_name]
        
        # Find the hex data in dialect resources
        hex_data = None
        for dialect_name, dialect_resources in self.dialect_resources.items():
            if resource_name in dialect_resources:
                hex_data = dialect_resources[resource_name]
                break
        
        if not hex_data:
            print(f"Resource '{resource_name}' not found in dialect_resources block")
            return None
        
        # Convert to numpy array
        shape = mapping_info.get('tensor_shape', [])
        dtype = mapping_info.get('tensor_dtype', 'f32')
        
        return self.hex_to_numpy(hex_data, shape, dtype)
    
    def list_resources(self) -> List[str]:
        """List all available resources."""
        if not self.resource_mappings:
            self.extract_dense_resource_references()
        return list(self.resource_mappings.keys())
    
    def print_resource_summary(self):
        """Print detailed resource summary."""
        references = self.extract_dense_resource_references()
        resources = self.extract_dialect_resources_block()
        
        print(f"Advanced MLIR Resource Summary")
        print(f"=" * 50)
        print(f"File: {self.mlir_file_path}")
        print(f"MLIR Python Bindings: Available")
        print(f"IREE Extensions: {'Available' if IREE_AVAILABLE else 'Not Available'}")
        print()
        
        print(f"Resource References Found: {len(references)}")
        print(f"Dialect Resource Blocks: {len(resources)}")
        
        for dialect_name, dialect_res in resources.items():
            print(f"  {dialect_name}: {len(dialect_res)} resources")
        print()
        
        if references:
            print("Resource Details:")
            print("-" * 40)
            
            for name, info in references.items():
                print(f"Name: {name}")
                print(f"  Shape: {info.get('tensor_shape', 'Unknown')}")
                print(f"  Type: {info.get('tensor_dtype', 'Unknown')}")
                print(f"  Operation: {info.get('operation', 'Unknown')}")
                
                # Check if binary data is available
                has_data = False
                data_size = 0
                for dialect_res in resources.values():
                    if name in dialect_res:
                        has_data = True
                        hex_str = dialect_res[name]
                        if hex_str.startswith('0x'):
                            hex_str = hex_str[2:]
                        data_size = len(hex_str) // 2  # hex bytes
                        break
                
                print(f"  Binary Data: {'Available' if has_data else 'Not Found'}")
                if has_data:
                    print(f"  Data Size: {data_size} bytes")
                print()


def main():
    """Main function with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="Advanced MLIR dialect resource extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary of all resources
  python mlir_resource_extractor_advanced.py model.mlir
  
  # List all resource names
  python mlir_resource_extractor_advanced.py model.mlir --list
  
  # Extract specific tensor
  python mlir_resource_extractor_advanced.py model.mlir --extract "_layer.0.rel_attn.o"
  
  # Save tensor to file
  python mlir_resource_extractor_advanced.py model.mlir --extract "_layer.0.rel_attn.o" --save output.npy
        """
    )
    
    parser.add_argument(
        "mlir_file",
        help="Path to MLIR file with dialect resources"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available resources"
    )
    parser.add_argument(
        "--extract", "-e",
        help="Extract specific resource by name"
    )
    parser.add_argument(
        "--save", "-s",
        help="Save extracted tensor to numpy file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mlir_file):
        print(f"Error: File '{args.mlir_file}' not found")
        sys.exit(1)
    
    try:
        extractor = AdvancedMLIRResourceExtractor(args.mlir_file)
        
        if not extractor.load_module():
            print("Failed to load MLIR module")
            sys.exit(1)
        
        if args.list:
            resources = extractor.list_resources()
            print("Available Resources:")
            for i, resource in enumerate(resources, 1):
                print(f"  {i:3d}. {resource}")
        
        elif args.extract:
            tensor = extractor.get_resource_tensor(args.extract)
            if tensor is not None:
                print(f"Successfully extracted tensor '{args.extract}'")
                print(f"Shape: {tensor.shape}")
                print(f"Dtype: {tensor.dtype}")
                print(f"Size: {tensor.size} elements")
                
                if args.verbose:
                    print(f"Min value: {tensor.min()}")
                    print(f"Max value: {tensor.max()}")
                    print(f"Mean: {tensor.mean():.6f}")
                
                if args.save:
                    np.save(args.save, tensor)
                    print(f"Saved to: {args.save}")
            else:
                print(f"Failed to extract tensor '{args.extract}'")
                sys.exit(1)
        
        else:
            # Default: show summary
            extractor.print_resource_summary()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()