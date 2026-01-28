#!/usr/bin/env python3
"""
Simplified MLIR dialect resource extractor that works with available packages.

This script extracts MLIR dialect resources using IREE/torch-mlir packages
available in the testsuite environment, with fallback to direct parsing.
"""

import sys
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import argparse
import binascii
import json

# Try different MLIR/IREE import options
MLIR_AVAILABLE = False
IREE_AVAILABLE = False

try:
    # Try IREE compiler bindings
    import iree.compiler.ir as ir
    from iree.compiler.dialects import builtin
    IREE_AVAILABLE = True
    print("Using IREE compiler bindings")
except ImportError:
    try:
        # Try torch-mlir bindings
        import torch_mlir._mlir_libs._mlir as mlir
        print("Using torch-mlir bindings")
        MLIR_AVAILABLE = True
    except ImportError:
        try:
            # Try standard MLIR bindings
            from mlir import ir
            from mlir.dialects import builtin
            MLIR_AVAILABLE = True
            print("Using standard MLIR bindings")
        except ImportError:
            print("Warning: No MLIR Python bindings found. Using fallback parser.")


class MLIRResourceExtractor:
    """MLIR resource extractor with multiple backend support."""
    
    def __init__(self, mlir_file_path: str):
        """Initialize the extractor."""
        self.mlir_file_path = mlir_file_path
        self.mlir_text = ""
        self.dialect_resources = {}
        self.resource_references = {}
        
    def load_file(self) -> bool:
        """Load the MLIR file."""
        try:
            with open(self.mlir_file_path, 'r') as f:
                self.mlir_text = f.read()
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def parse_with_iree(self) -> bool:
        """Parse using IREE bindings if available."""
        if not IREE_AVAILABLE:
            return False
        
        try:
            with ir.Context() as ctx:
                module = ir.Module.parse(self.mlir_text)
                self._extract_from_module_iree(module)
                return True
        except Exception as e:
            print(f"IREE parsing failed: {e}")
            return False
    
    def parse_with_mlir(self) -> bool:
        """Parse using standard MLIR bindings if available."""
        if not MLIR_AVAILABLE:
            return False
        
        try:
            # This will depend on the exact MLIR binding available
            # Implementation would go here
            print("MLIR binding parsing not fully implemented yet")
            return False
        except Exception as e:
            print(f"MLIR parsing failed: {e}")
            return False
    
    def parse_fallback(self) -> bool:
        """Fallback parser using text processing (but not regex)."""
        try:
            # Extract dense resource references
            self._extract_references_fallback()
            
            # Extract dialect resources block
            self._extract_resources_fallback()
            
            return True
        except Exception as e:
            print(f"Fallback parsing failed: {e}")
            return False
    
    def _extract_references_fallback(self):
        """Extract resource references using careful text parsing."""
        lines = self.mlir_text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Look for dense_resource patterns
            if 'dense_resource<' in line and 'tensor<' in line:
                # Parse the line carefully
                info = self._parse_dense_resource_line(line)
                if info:
                    resource_name = info['resource_name']
                    self.resource_references[resource_name] = {
                        'line_number': line_num,
                        'shape': info.get('shape', []),
                        'dtype': info.get('dtype', 'unknown'),
                        'full_line': line
                    }
    
    def _parse_dense_resource_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a line containing dense_resource."""
        try:
            # Find dense_resource<name>
            start = line.find('dense_resource<')
            if start == -1:
                return None
            
            start += len('dense_resource<')
            end = line.find('>', start)
            if end == -1:
                return None
            
            resource_name = line[start:end]
            
            # Find tensor specification
            tensor_start = line.find('tensor<', end)
            if tensor_start == -1:
                return {'resource_name': resource_name}
            
            tensor_end = line.find('>', tensor_start)
            if tensor_end == -1:
                return {'resource_name': resource_name}
            
            tensor_spec = line[tensor_start + 7:tensor_end]
            
            # Parse shape and dtype
            shape, dtype = self._parse_tensor_spec(tensor_spec)
            
            return {
                'resource_name': resource_name,
                'shape': shape,
                'dtype': dtype,
                'tensor_spec': tensor_spec
            }
        except Exception:
            return None
    
    def _parse_tensor_spec(self, spec: str) -> Tuple[List[int], str]:
        """Parse tensor specification like '768x12x64xf32'."""
        shape = []
        dtype = 'f32'
        
        try:
            parts = spec.split('x')
            
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Last part might contain dtype
                    if any(c.isalpha() for c in part):
                        # Extract numeric part
                        num_part = ""
                        dtype_part = ""
                        in_dtype = False
                        
                        for c in part:
                            if c.isdigit():
                                if not in_dtype:
                                    num_part += c
                                else:
                                    dtype_part += c
                            elif c.isalpha():
                                in_dtype = True
                                dtype_part += c
                        
                        if num_part:
                            shape.append(int(num_part))
                        if dtype_part:
                            dtype = dtype_part
                    else:
                        if part != '?':
                            shape.append(int(part))
                        else:
                            shape.append(-1)  # Dynamic dimension
                else:
                    if part != '?':
                        shape.append(int(part))
                    else:
                        shape.append(-1)
        except Exception:
            pass
        
        return shape, dtype
    
    def _extract_resources_fallback(self):
        """Extract dialect resources block using text parsing."""
        # Find the start of dialect resources block
        start_marker = '{-#'
        end_marker = '#-}'
        
        start_idx = self.mlir_text.find(start_marker)
        if start_idx == -1:
            return
        
        end_idx = self.mlir_text.find(end_marker, start_idx)
        if end_idx == -1:
            return
        
        resource_block = self.mlir_text[start_idx + len(start_marker):end_idx]
        
        # Parse the resource block structure
        self.dialect_resources = self._parse_resource_block(resource_block)
    
    def _parse_resource_block(self, block_text: str) -> Dict[str, Dict[str, str]]:
        """Parse the dialect_resources block content."""
        resources = {}
        lines = block_text.strip().split('\n')
        
        current_dialect = None
        current_resource = None
        in_hex_data = False
        hex_data_buffer = ""
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('//'):
                continue
            
            # Check for dialect_resources start
            if 'dialect_resources:' in line:
                continue
            
            # Check for dialect section (e.g., "builtin: {")
            if line.endswith(': {') and not line.startswith('_'):
                current_dialect = line.replace(': {', '').strip()
                resources[current_dialect] = {}
                continue
            
            # Check for resource entry
            if line.startswith('_') and ': ' in line:
                parts = line.split(': ', 1)
                current_resource = parts[0].strip()
                data_part = parts[1].strip()
                
                if data_part.startswith('"') and data_part.endswith('"'):
                    # Single line hex data
                    hex_data = data_part.strip('"')
                    if current_dialect and current_resource:
                        resources[current_dialect][current_resource] = hex_data
                elif data_part.startswith('"'):
                    # Multi-line hex data starts
                    in_hex_data = True
                    hex_data_buffer = data_part.strip('"')
                continue
            
            # Handle multi-line hex data
            if in_hex_data:
                if line.endswith('"'):
                    # End of hex data
                    hex_data_buffer += line.rstrip('"')
                    if current_dialect and current_resource:
                        resources[current_dialect][current_resource] = hex_data_buffer
                    in_hex_data = False
                    hex_data_buffer = ""
                else:
                    hex_data_buffer += line
                continue
            
            # Handle closing braces
            if line in ['}', '},']:
                continue
        
        return resources
    
    def _extract_from_module_iree(self, module):
        """Extract resources from IREE module object."""
        # Implementation would depend on IREE API
        pass
    
    def hex_to_numpy(self, hex_string: str, shape: List[int], dtype: str) -> Optional[np.ndarray]:
        """Convert hex string to numpy array."""
        try:
            # Remove 0x prefix if present
            if hex_string.startswith('0x'):
                hex_string = hex_string[2:]
            
            # Convert hex to bytes
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
            
            # Create array from binary data
            array = np.frombuffer(binary_data, dtype=np_dtype)
            
            # Reshape if valid shape provided
            if shape and all(s > 0 for s in shape):
                expected_size = np.prod(shape)
                if len(array) >= expected_size:
                    return array[:expected_size].reshape(shape)
                else:
                    print(f"Warning: Array size {len(array)} < expected {expected_size}")
            
            return array
            
        except Exception as e:
            print(f"Error converting hex to numpy: {e}")
            return None
    
    def extract_all(self) -> bool:
        """Extract resources using best available method."""
        if not self.load_file():
            return False
        
        # Try methods in order of preference
        if self.parse_with_iree():
            return True
        elif self.parse_with_mlir():
            return True
        else:
            return self.parse_fallback()
    
    def get_resource(self, resource_name: str) -> Optional[np.ndarray]:
        """Get a specific resource as numpy array."""
        if resource_name not in self.resource_references:
            print(f"Resource '{resource_name}' not found")
            return None
        
        ref_info = self.resource_references[resource_name]
        
        # Find hex data in dialect resources
        hex_data = None
        for dialect_resources in self.dialect_resources.values():
            if resource_name in dialect_resources:
                hex_data = dialect_resources[resource_name]
                break
        
        if not hex_data:
            print(f"No binary data found for resource '{resource_name}'")
            return None
        
        # Convert to numpy
        shape = ref_info.get('shape', [])
        dtype = ref_info.get('dtype', 'f32')
        
        return self.hex_to_numpy(hex_data, shape, dtype)
    
    def list_resources(self) -> List[str]:
        """List all available resources."""
        return list(self.resource_references.keys())
    
    def print_summary(self):
        """Print summary of extracted resources."""
        print(f"MLIR Resource Summary")
        print(f"=" * 40)
        print(f"File: {self.mlir_file_path}")
        print(f"Resource References: {len(self.resource_references)}")
        
        total_resources = sum(len(dr) for dr in self.dialect_resources.values())
        print(f"Dialect Resources: {total_resources}")
        
        for dialect, resources in self.dialect_resources.items():
            print(f"  {dialect}: {len(resources)}")
        
        print(f"\nBackend: {'IREE' if IREE_AVAILABLE else 'MLIR' if MLIR_AVAILABLE else 'Fallback'}")
        print()
        
        if self.resource_references:
            print("Available Resources:")
            for name, info in self.resource_references.items():
                shape = info.get('shape', [])
                dtype = info.get('dtype', 'unknown')
                
                # Check if binary data exists
                has_data = any(name in dr for dr in self.dialect_resources.values())
                
                print(f"  {name}")
                print(f"    Shape: {shape}")
                print(f"    Type: {dtype}")
                print(f"    Binary Data: {'✓' if has_data else '✗'}")
                print()


def main():
    parser = argparse.ArgumentParser(description="MLIR Dialect Resource Extractor")
    parser.add_argument("mlir_file", help="MLIR file with dialect resources")
    parser.add_argument("--list", "-l", action="store_true", help="List resources")
    parser.add_argument("--extract", "-e", help="Extract specific resource")
    parser.add_argument("--save", "-s", help="Save extracted tensor to file")
    parser.add_argument("--format", choices=['npy', 'txt', 'json'], default='npy', 
                       help="Output format for saved tensor")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mlir_file):
        print(f"Error: File not found: {args.mlir_file}")
        sys.exit(1)
    
    extractor = MLIRResourceExtractor(args.mlir_file)
    
    if not extractor.extract_all():
        print("Failed to extract resources")
        sys.exit(1)
    
    if args.list:
        resources = extractor.list_resources()
        print("Available Resources:")
        for i, name in enumerate(resources, 1):
            print(f"  {i:3d}. {name}")
    
    elif args.extract:
        tensor = extractor.get_resource(args.extract)
        if tensor is not None:
            print(f"Extracted tensor: {args.extract}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Size: {tensor.size}")
            print(f"  Min/Max: {tensor.min():.6f} / {tensor.max():.6f}")
            
            if args.save:
                if args.format == 'npy':
                    np.save(args.save, tensor)
                elif args.format == 'txt':
                    np.savetxt(args.save, tensor.flatten() if len(tensor.shape) > 1 else tensor)
                elif args.format == 'json':
                    with open(args.save, 'w') as f:
                        json.dump({
                            'shape': tensor.shape.tolist(),
                            'dtype': str(tensor.dtype),
                            'data': tensor.tolist()
                        }, f, indent=2)
                print(f"Saved to: {args.save}")
        else:
            print(f"Failed to extract: {args.extract}")
            sys.exit(1)
    
    else:
        extractor.print_summary()


if __name__ == "__main__":
    main()