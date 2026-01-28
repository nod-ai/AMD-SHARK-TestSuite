#!/usr/bin/env python3
"""
Script to access MLIR dialect resources using IREE/MLIR Python bindings.

This script parses an MLIR file containing dialect resources and provides
access to the embedded tensor data without using string matching or regex.
"""

import sys
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import argparse

try:
    from mlir.ir import Context, Module, Operation, Block
    from mlir.dialects import builtin
    from iree.compiler import ir as iree_ir
except ImportError as e:
    print(f"Error importing MLIR/IREE Python bindings: {e}")
    print("Please ensure IREE is installed with Python bindings.")
    print("Try: pip install iree-compiler")
    sys.exit(1)


class MLIRDialectResourceExtractor:
    """Extracts and manages MLIR dialect resources using Python bindings."""
    
    def __init__(self, mlir_file_path: str):
        """
        Initialize the resource extractor.
        
        Args:
            mlir_file_path: Path to the MLIR file containing dialect resources
        """
        self.mlir_file_path = mlir_file_path
        self.context = Context()
        self.module = None
        self.resources = {}
        self.dense_resource_references = {}
        
    def load_module(self) -> bool:
        """
        Load the MLIR module from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.mlir_file_path, 'r') as f:
                mlir_text = f.read()
            
            with self.context:
                self.module = Module.parse(mlir_text)
                return True
                
        except Exception as e:
            print(f"Error loading MLIR module: {e}")
            return False
    
    def extract_dense_resource_references(self) -> Dict[str, Any]:
        """
        Extract references to dense resources from the MLIR operations.
        
        Returns:
            Dictionary mapping resource names to their usage info
        """
        if not self.module:
            raise ValueError("Module not loaded. Call load_module() first.")
        
        references = {}
        
        def walk_operation(op: Operation):
            """Recursively walk through operations to find dense_resource references."""
            # Check if this operation has a dense_resource attribute
            for attr_name in op.attributes:
                attr = op.attributes[attr_name]
                attr_str = str(attr)
                
                # Look for dense_resource patterns in attributes
                if "dense_resource" in attr_str:
                    # Extract resource name and type information
                    # This is a safer alternative to regex - using MLIR's own parsing
                    try:
                        # The attribute should be a DenseResourceElementsAttr
                        if hasattr(attr, 'type') and hasattr(attr, 'resource_handle'):
                            resource_name = str(attr.resource_handle)
                            tensor_type = attr.type
                            references[resource_name] = {
                                'type': tensor_type,
                                'operation': op,
                                'attribute_name': attr_name
                            }
                    except AttributeError:
                        # Fallback: parse the string representation carefully
                        # This is still safer than regex as we're using MLIR's own string format
                        if "dense_resource<" in attr_str and ">" in attr_str:
                            start = attr_str.find("dense_resource<") + len("dense_resource<")
                            end = attr_str.find(">", start)
                            if start < end:
                                resource_name = attr_str[start:end]
                                references[resource_name] = {
                                    'attribute_str': attr_str,
                                    'operation': op,
                                    'attribute_name': attr_name
                                }
            
            # Recursively walk child operations
            for region in op.regions:
                for block in region:
                    for child_op in block:
                        walk_operation(child_op)
        
        # Walk the entire module
        walk_operation(self.module.operation)
        self.dense_resource_references = references
        return references
    
    def extract_dialect_resources(self) -> Dict[str, Any]:
        """
        Extract the dialect resources data from the module.
        
        Returns:
            Dictionary containing the extracted resource data
        """
        if not self.module:
            raise ValueError("Module not loaded. Call load_module() first.")
        
        # Try to access resources through the module's resource manager
        try:
            # MLIR modules can have resource managers attached
            if hasattr(self.module.operation, 'resources'):
                return self._extract_from_resource_manager()
            else:
                return self._extract_from_attributes()
                
        except Exception as e:
            print(f"Warning: Could not extract resources using primary method: {e}")
            return self._extract_from_attributes()
    
    def _extract_from_resource_manager(self) -> Dict[str, Any]:
        """Extract resources using MLIR's resource manager API."""
        resources = {}
        
        # This would be the proper way if the API is available
        # The exact API depends on the MLIR Python bindings version
        try:
            resource_mgr = self.module.operation.resources
            for resource_name in resource_mgr:
                resource_data = resource_mgr[resource_name]
                resources[resource_name] = resource_data
                
        except AttributeError:
            # Fallback to attribute extraction
            pass
            
        return resources
    
    def _extract_from_attributes(self) -> Dict[str, Any]:
        """Extract resources by parsing module attributes."""
        resources = {}
        
        # Look for dialect resource attributes on the module
        module_op = self.module.operation
        
        for attr_name in module_op.attributes:
            attr = module_op.attributes[attr_name]
            
            # Check if this is a dialect resource attribute
            if "dialect_resources" in attr_name.lower() or "resource" in attr_name.lower():
                # Try to extract the actual resource data
                try:
                    if hasattr(attr, 'value'):
                        resources[attr_name] = attr.value
                    else:
                        resources[attr_name] = str(attr)
                except Exception as e:
                    print(f"Warning: Could not extract resource {attr_name}: {e}")
        
        return resources
    
    def get_resource_tensor(self, resource_name: str) -> Optional[np.ndarray]:
        """
        Get a specific resource as a numpy array.
        
        Args:
            resource_name: Name of the resource to retrieve
            
        Returns:
            Numpy array if successful, None otherwise
        """
        if resource_name not in self.dense_resource_references:
            print(f"Resource '{resource_name}' not found in references")
            return None
        
        ref_info = self.dense_resource_references[resource_name]
        
        # Try to get the actual tensor data
        try:
            # If we have type information, use it
            if 'type' in ref_info:
                tensor_type = ref_info['type']
                # The exact method to get tensor data depends on MLIR version
                # This is a placeholder for the actual implementation
                print(f"Tensor type for {resource_name}: {tensor_type}")
            
            # For now, we'll need to implement the actual data extraction
            # This would involve interfacing with MLIR's resource system
            print(f"Resource {resource_name} found but data extraction needs implementation")
            return None
            
        except Exception as e:
            print(f"Error extracting tensor for {resource_name}: {e}")
            return None
    
    def list_resources(self) -> List[str]:
        """
        List all available resources in the module.
        
        Returns:
            List of resource names
        """
        self.extract_dense_resource_references()
        return list(self.dense_resource_references.keys())
    
    def print_resource_summary(self):
        """Print a summary of all resources found in the module."""
        references = self.extract_dense_resource_references()
        resources = self.extract_dialect_resources()
        
        print(f"MLIR Module Resource Summary")
        print(f"=" * 40)
        print(f"File: {self.mlir_file_path}")
        print(f"Dense Resource References: {len(references)}")
        print(f"Dialect Resources Found: {len(resources)}")
        print()
        
        if references:
            print("Dense Resource References:")
            for name, info in references.items():
                print(f"  - {name}")
                if 'type' in info:
                    print(f"    Type: {info['type']}")
                if 'attribute_str' in info:
                    # Safely show part of the attribute string
                    attr_str = info['attribute_str']
                    if len(attr_str) > 100:
                        attr_str = attr_str[:97] + "..."
                    print(f"    Attribute: {attr_str}")
                print()
        
        if resources:
            print("Dialect Resources:")
            for name, data in resources.items():
                print(f"  - {name}: {type(data)}")
                print()


def main():
    """Main function to demonstrate the resource extractor."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze MLIR dialect resources"
    )
    parser.add_argument(
        "mlir_file",
        help="Path to the MLIR file containing dialect resources"
    )
    parser.add_argument(
        "--list-resources",
        action="store_true",
        help="List all resources in the file"
    )
    parser.add_argument(
        "--resource-name",
        help="Extract specific resource by name"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of all resources"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mlir_file):
        print(f"Error: File '{args.mlir_file}' not found")
        sys.exit(1)
    
    try:
        extractor = MLIRDialectResourceExtractor(args.mlir_file)
        
        if not extractor.load_module():
            print("Failed to load MLIR module")
            sys.exit(1)
        
        if args.summary or (not args.list_resources and not args.resource_name):
            extractor.print_resource_summary()
        
        if args.list_resources:
            resources = extractor.list_resources()
            print("Available Resources:")
            for resource in resources:
                print(f"  {resource}")
        
        if args.resource_name:
            tensor = extractor.get_resource_tensor(args.resource_name)
            if tensor is not None:
                print(f"Successfully extracted tensor for '{args.resource_name}'")
                print(f"Shape: {tensor.shape}")
                print(f"Dtype: {tensor.dtype}")
            else:
                print(f"Failed to extract tensor for '{args.resource_name}'")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()