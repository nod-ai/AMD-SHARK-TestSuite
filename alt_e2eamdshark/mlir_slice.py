#!/usr/bin/env python3
"""
MLIR Slicing Tool

Takes an MLIR file and extracts a slice of operations that lead to a specific SSA value.
Creates a new valid MLIR module with the minimal set of operations needed.
Preserves original SSA value names and handles dialect_resources for constants.
"""

import argparse
import sys
import re
from collections import deque
from typing import Set, Dict, List, Tuple, Optional

try:
    from mlir.ir import Context, Module, Operation, Value, BlockArgument, OpResult, Type, Location
    from mlir.dialects import func
    USING_TORCH_MLIR = False
except ImportError:
    try:
        # Try torch-mlir bindings
        import torch_mlir
        from torch_mlir.ir import Context, Module, Operation, Value, BlockArgument, OpResult, Type, Location
        from torch_mlir.dialects import func
        USING_TORCH_MLIR = True
    except ImportError:
        print("Error: MLIR Python bindings not found.")
        print("Install with: pip install torch-mlir or pip install mlir")
        sys.exit(1)


class MLIRSlicer:
    def __init__(self, module_str: str, target_value: str, depth: Optional[int] = None):
        """
        Initialize the MLIR slicer.
        
        Args:
            module_str: MLIR module as a string
            target_value: The SSA value name to return (e.g., "%123")
            depth: Maximum depth of ancestors to include (None = unlimited)
        """
        self.module_str = module_str
        self.ctx = Context()
        self.ctx.allow_unregistered_dialects = True
        
        # Register torch-mlir dialects if using torch-mlir
        if USING_TORCH_MLIR:
            try:
                import torch_mlir.dialects.torch as torch_dialect
                torch_dialect.register_dialect(self.ctx)
            except:
                pass
        
        self.module = Module.parse(module_str, self.ctx)
        self.target_value = target_value
        self.depth = depth
        
        # Track operations and values we need
        self.required_ops: Set[Operation] = set()
        self.required_values: Set[Value] = set()
        self.external_values: List[Tuple[Value, Type]] = []  # Values that need to become inputs
        self.required_constants: Set[str] = set()  # Resource names for constants
        self.resource_to_op: Dict[str, Operation] = {}  # Map resource names to defining operations
        
    def find_value_by_name(self, name: str) -> Optional[Value]:
        """Find an SSA value by its name in the module."""
        # Clean up the name
        name = name.strip()
        if not name.startswith('%'):
            name = '%' + name
        
        # Try to extract numeric portion
        numeric_match = re.match(r'%(\d+)', name)
        target_num = int(numeric_match.group(1)) if numeric_match else None
        
        for func_op in self.module.body.operations:
            # Just search in all top-level operations
            found = self._search_value_in_op(func_op, name, target_num)
            if found:
                return found
        
        return None
    
    def _search_value_in_op(self, op: Operation, name: str, target_num: Optional[int]) -> Optional[Value]:
        """Recursively search for a value in an operation and its nested regions."""
        # Check this operation's results
        for result_idx, result in enumerate(op.results):
            # Get the value's string representation
            result_str = str(result).strip()
            
            # Try multiple matching strategies
            # The result string looks like: Value(%1495 = torch.operator ...)
            if target_num is not None:
                # Match patterns like "Value(%1495 = " or just "%1495"
                result_match = re.search(r'%(\d+)\s*[=)]', result_str)
                if result_match and int(result_match.group(1)) == target_num:
                    return result
            
            if name in result_str or result_str.startswith(name):
                return result
        
        # Search nested regions
        for region in op.regions:
            for block in region.blocks:
                # Check block arguments
                for arg_idx, arg in enumerate(block.arguments):
                    arg_str = str(arg).strip()
                    if name in arg_str or f"%arg{arg_idx}" == name:
                        return arg
                
                # Check operations in the block
                for inner_op in block.operations:
                    found = self._search_value_in_op(inner_op, name, target_num)
                    if found:
                        return found
        
        return None
    
    def get_defining_op(self, value: Value) -> Optional[Operation]:
        """Get the operation that defines a value, if any."""
        try:
            # Try the standard way for OpResult
            if isinstance(value, OpResult):
                return value.owner
            # Try torch-mlir's way - check if value has .owner attribute
            if hasattr(value, 'owner'):
                owner = value.owner
                # Make sure it's an Operation, not a Block
                if isinstance(owner, Operation):
                    return owner
            # Try through the value's methods
            if hasattr(value, 'get_defining_op'):
                return value.get_defining_op()
        except:
            pass
        return None
    
    def build_resource_mapping(self) -> None:
        """Build a mapping from resource names to constant operations."""
        for func_op in self.module.body.operations:
            self._scan_for_constants(func_op)
    
    def _scan_for_constants(self, op: Operation) -> None:
        """Recursively scan for constant operations and build resource mapping."""
        op_str = str(op)
        if 'onnx.Constant' in op_str or 'torch.constant' in op_str:
            resource_match = re.search(r'dense_resource<([^>]+)>', op_str)
            if resource_match:
                resource_name = resource_match.group(1)
                self.resource_to_op[resource_name] = op
        
        # Scan nested regions
        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.operations:
                    self._scan_for_constants(inner_op)
    
    def collect_ancestor_ops(self, target_value: Value) -> None:
        """
        Collect all ancestor operations needed to compute the target value.
        Uses BFS with depth limiting if specified.
        """
        # Queue items: (value, current_depth)
        queue = deque([(target_value, 0)])
        visited_values: Set[Value] = {target_value}
        
        while queue:
            current_value, current_depth = queue.popleft()
            
            # If it's a block argument, it's already an input
            if isinstance(current_value, BlockArgument):
                if current_value not in [v for v, _ in self.external_values]:
                    self.external_values.append((current_value, current_value.type))
                continue
            
            # Get the defining operation
            defining_op = self.get_defining_op(current_value)
            if defining_op is None:
                # External value (not defined in this context)
                if current_value not in [v for v, _ in self.external_values]:
                    self.external_values.append((current_value, current_value.type))
                continue
            
            # Add this operation to required set
            self.required_ops.add(defining_op)
            
            # Check if this is a constant and extract its resource name
            op_str = str(defining_op)
            if 'onnx.Constant' in op_str or 'torch.constant' in op_str:
                resource_match = re.search(r'dense_resource<([^>]+)>', op_str)
                if resource_match:
                    self.required_constants.add(resource_match.group(1))
            
            # Check depth limit for going deeper
            if self.depth is not None and current_depth + 1 >= self.depth:
                # We're at the depth limit, mark operands as external
                for operand in defining_op.operands:
                    if operand not in visited_values and operand not in [v for v, _ in self.external_values]:
                        self.external_values.append((operand, operand.type))
                        visited_values.add(operand)
                continue
            
            # Traverse operands (dependencies)
            for operand in defining_op.operands:
                if operand not in visited_values:
                    visited_values.add(operand)
                    queue.append((operand, current_depth + 1))
        
        # After BFS, we need to scan all operations for any constant resource references
        # and ensure we track those resources
        for op in list(self.required_ops):
            op_str = str(op)
            # Look for any dense_resource references in attributes
            for resource_match in re.finditer(r'dense_resource<([^>]+)>', op_str):
                resource_name = resource_match.group(1)
                self.required_constants.add(resource_name)
        
        # Now add constant operations that define required resources
        for resource_name in self.required_constants:
            if resource_name in self.resource_to_op:
                const_op = self.resource_to_op[resource_name]
                if const_op not in self.required_ops:
                    self.required_ops.add(const_op)
    
    def extract_dialect_resources(self) -> str:
        """Extract the dialect_resources block with only required constants."""
        if not self.required_constants:
            return ""
        
        # Find the dialect_resources comment block: {-# ... #-}
        start_marker = "{-#"
        end_marker = "#-}"
        
        start_idx = self.module_str.find(start_marker)
        if start_idx == -1:
            print("Warning: Could not find dialect_resources block start marker {-#")
            return ""
        
        end_idx = self.module_str.find(end_marker, start_idx)
        if end_idx == -1:
            print("Warning: Could not find dialect_resources block end marker #-}")
            return ""
        
        # Extract the full resources block
        resources_block = self.module_str[start_idx:end_idx + len(end_marker)]
        
        # Parse and filter resources - each resource is a single line: name: "0x..."
        lines = resources_block.split('\n')
        filtered_lines = []
        resource_data_lines = []  # Collect resource lines separately
        
        print(f"DEBUG: Processing {len(lines)} lines from resources block")
        resource_lines_skipped = 0
        resource_lines_included = 0
        
        for line in lines:
            # Keep structural lines (headers, markers, braces)
            if ('{-#' in line or 'dialect_resources:' in line or 
                '#-}' in line or 'builtin:' in line or 
                line.strip() == '{' or line.strip() == '}'):
                filtered_lines.append(line)
                continue
            
            # Check if this is a resource entry line (name: "0x...")
            # Each resource is a complete line, not multiple lines
            if ':' in line and '"0x' in line:
                colon_idx = line.find(':')
                resource_name = line[:colon_idx].strip()
                # Include this line only if the resource is required
                if resource_name in self.required_constants:
                    print(f"Including resource: {resource_name}")
                    # Strip trailing comma if present - we'll add them back correctly
                    cleaned_line = line.rstrip().rstrip(',')
                    resource_data_lines.append(cleaned_line)
                    resource_lines_included += 1
                else:
                    resource_lines_skipped += 1
        
        # Insert resource lines back with proper comma handling
        # Find the position after "builtin: {" line
        builtin_line_idx = -1
        for i, line in enumerate(filtered_lines):
            if 'builtin:' in line and '{' in line:
                # This is the "builtin: {" line
                builtin_line_idx = i
                break
        
        # Insert resources with commas (all but last should have comma)
        if builtin_line_idx >= 0 and resource_data_lines:
            # Insert after the "builtin: {" line
            insert_pos = builtin_line_idx + 1
            for i, res_line in enumerate(resource_data_lines):
                if i < len(resource_data_lines) - 1:
                    # Not the last resource - add comma
                    filtered_lines.insert(insert_pos + i, res_line + ',')
                else:
                    # Last resource - no comma
                    filtered_lines.insert(insert_pos + i, res_line)
        elif resource_data_lines:
            print("Warning: Could not find builtin brace position, appending resources at end")
                # Otherwise skip this resource line completely
        
        print(f"DEBUG: Included {resource_lines_included} resource lines, skipped {resource_lines_skipped}")
        print(f"DEBUG: Total filtered lines: {len(filtered_lines)}")
        
        # Check if we have any resources
        has_resources = any(':' in line and '\"' in line for line in filtered_lines)
        if not has_resources:
            print("Warning: No matching resources found in dialect_resources block")
            return ""
        
        result = '\n'.join(filtered_lines)
        print(f"DEBUG: Returning filtered resources, size = {len(result)} bytes")
        return result
    
    def build_sliced_module(self, target_value: Value) -> str:
        """
        Build a new MLIR module containing only the required operations.
        Preserves original SSA value names.
        """
        # Sort operations in topological order (dependencies first)
        sorted_ops = self.topological_sort()
        
        print(f"DEBUG: sorted_ops length = {len(sorted_ops)}", file=sys.stderr)
        
        # Build the new function
        lines = []
        lines.append("module {")
        
        # Build function signature with named arguments
        # Also build a mapping from SSA numbers to argument names
        arg_list = []
        ssa_to_arg = {}  # Map SSA number string to argument name
        
        for idx, (ext_value, ext_type) in enumerate(self.external_values):
            arg_name = f"%arg{idx}"
            arg_list.append(f"{arg_name}: {ext_type}")
            
            # Extract SSA number from external value
            ext_str = str(ext_value)
            ext_match = re.search(r'%(\d+)\s*=', ext_str)
            if ext_match:
                ssa_num = ext_match.group(1)
                ssa_to_arg[ssa_num] = arg_name
        
        input_args_str = ", ".join(arg_list)
        lines.append(f"  func.func @sliced_func({input_args_str}) -> {target_value.type} {{")
        
        # Add operations, preserving original SSA names but replacing external references
        for op in sorted_ops:
            op_str = str(op).strip()
            
            # Replace external SSA values with argument names in operand positions
            if '=' in op_str:
                parts = op_str.split('=', 1)
                result_part = parts[0]
                body_part = parts[1]
                
                # Replace each SSA number that maps to an argument
                for ssa_num, arg_name in ssa_to_arg.items():
                    # Use negative lookahead/lookbehind to match %NNN but not %NNNN
                    body_part = re.sub(r'%' + ssa_num + r'(?!\d)', arg_name, body_part)
                
                op_str = result_part + '=' + body_part
            else:
                # No result (shouldn't happen for our ops, but handle it)
                for ssa_num, arg_name in ssa_to_arg.items():
                    op_str = re.sub(r'%' + ssa_num + r'(?!\d)', arg_name, op_str)
            
            lines.append(f"    {op_str}")
        
        # Add return statement
        target_str = str(target_value)
        target_match = re.search(r'%(\d+)\s*=', target_str)
        if target_match:
            return_ssa = target_match.group(1)
            # Check if this is an external value (use arg name instead)
            if return_ssa in ssa_to_arg:
                return_name = ssa_to_arg[return_ssa]
            else:
                return_name = f"%{return_ssa}"
        else:
            return_name = str(target_value)
        
        lines.append(f"    return {return_name} : {target_value.type}")
        
        lines.append("  }")
        lines.append("}")
        
        # Build module first
        result = "\n".join(lines)
        
        print(f"DEBUG: Module (first 500 chars): {result[:500]}", file=sys.stderr)
        
        # Add dialect_resources AFTER the module block (outside module)
        print(f"DEBUG: About to extract resources", file=sys.stderr)
        resources = self.extract_dialect_resources()
        print(f"DEBUG: Resources extracted, length = {len(resources)}", file=sys.stderr)
        print(f"DEBUG: Resources (first 100 chars): {resources[:100]}", file=sys.stderr)
        if resources:
            result = result + "\n" + resources
            print(f"DEBUG: Added resources after module", file=sys.stderr)
            print(f"DEBUG: Result after adding resources (first 500 chars): {result[:500]}", file=sys.stderr)
        
        print(f"DEBUG: Final result length = {len(result)}", file=sys.stderr)
        return result
    
    def topological_sort(self) -> List[Operation]:
        """Sort operations in topological order (dependencies before users)."""
        # Simple topological sort based on dependencies
        sorted_ops = []
        added = set()
        
        def add_op_recursive(op: Operation):
            if op in added or op not in self.required_ops:
                return
            
            # First add all dependencies
            for operand in op.operands:
                defining_op = self.get_defining_op(operand)
                if defining_op and defining_op in self.required_ops:
                    add_op_recursive(defining_op)
            
            # Then add this operation
            sorted_ops.append(op)
            added.add(op)
        
        # First add operations with dependencies (via recursive traversal)
        for op in self.required_ops:
            add_op_recursive(op)
        
        # Then explicitly add any remaining operations (e.g., constants with no operands)
        # These have no dependencies, so they can go at the beginning
        remaining_ops = []
        for op in self.required_ops:
            if op not in added:
                remaining_ops.append(op)
                added.add(op)
                print(f"DEBUG: Added orphan operation: {str(op)[:100]}", file=sys.stderr)
        
        print(f"DEBUG: {len(remaining_ops)} remaining operations added", file=sys.stderr)
        
        # Put constants at the beginning since they have no dependencies
        sorted_ops = remaining_ops + sorted_ops
        
        return sorted_ops
    
    def slice(self) -> str:
        """Perform the slicing and return the new MLIR module."""
        # Build resource mapping first
        print("Building resource-to-operation mapping...", file=sys.stderr)
        self.build_resource_mapping()
        print(f"Found {len(self.resource_to_op)} constant resources", file=sys.stderr)
        
        # Find the target value
        target_value = self.find_value_by_name(self.target_value)
        if target_value is None:
            raise ValueError(f"Could not find SSA value: {self.target_value}")
        
        print(f"Found target value: {target_value}", file=sys.stderr)
        
        # Collect ancestor operations
        self.collect_ancestor_ops(target_value)
        
        print(f"Collected {len(self.required_ops)} operations", file=sys.stderr)
        print(f"Found {len(self.external_values)} external inputs", file=sys.stderr)
        print(f"Found {len(self.required_constants)} required constants: {self.required_constants}", file=sys.stderr)
        
        # Build the sliced module
        sliced_module = self.build_sliced_module(target_value)
        
        return sliced_module


def main():
    parser = argparse.ArgumentParser(
        description="Slice an MLIR module to only include operations needed for a specific value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Slice to get operations producing %123 with unlimited depth
  python mlir_slice.py input.mlir %123 -o output.mlir
  
  # Slice with depth limit of 10 ancestors
  python mlir_slice.py input.mlir %123 --depth 10 -o output.mlir
  
  # Print to stdout
  python mlir_slice.py input.mlir %456
        """
    )
    
    parser.add_argument("input", help="Input MLIR file")
    parser.add_argument("value", help="Target SSA value (e.g., %%123)")
    parser.add_argument("-o", "--output", help="Output MLIR file (default: stdout)")
    parser.add_argument("-d", "--depth", type=int, help="Maximum depth of ancestors to include (default: unlimited)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input, 'r') as f:
            module_str = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Perform slicing
    try:
        slicer = MLIRSlicer(module_str, args.value, args.depth)
        sliced_module = slicer.slice()
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(sliced_module)
            print(f"Sliced module written to {args.output}", file=sys.stderr)
        else:
            print(sliced_module)
    
    except Exception as e:
        print(f"Error during slicing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
