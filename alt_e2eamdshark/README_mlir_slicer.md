# MLIR Slicer

A Python tool for extracting subgraphs from MLIR files based on SSA values, with intelligent handling of dialect resources and constants.

## Overview

`mlir_slice.py` analyzes an MLIR file and creates a minimal, valid MLIR file containing only the operations needed to compute a specific SSA value. It automatically:

- Traces dependencies backward from the target SSA value
- Includes only required operations up to a configurable depth
- Filters dialect resources to include only referenced constants
- Maintains proper MLIR structure with resources outside the module
- Generates valid, verifiable MLIR output

## Requirements

- Python 3.x
- `torch-mlir` Python bindings

```bash
pip install --no-deps -r torch_mlir_requirements.txt
```

## Usage

### Basic Syntax

```bash
python mlir_slice.py <input.mlir> <ssa_value> -d <depth> -o <output.mlir>
```

### Arguments

- **`input.mlir`** (required): Path to the input MLIR file
- **`ssa_value`** (required): Target SSA value (e.g., `%288`, `%196`)
- **`-d, --depth`** (optional): Maximum depth of ancestor operations to include (default: unlimited)
- **`-o, --output`** (required): Path to the output sliced MLIR file

### Examples

#### Example 1: Extract operations for %196 with depth 4

```bash
python mlir_slice.py test-onnx-7/xlnet_Opset16/model.torch_onnx.mlir %196 \
  -d 4 \
  -o test-onnx-7/xlnet_Opset16/sliced_196.mlir
```

**Output:**
```
Found 545 constant resources
Found target value: %196 = torch.operator "onnx.Neg"...
Collected 4 operations
Found 1 required constants: {'__5'}
Sliced module written to test-onnx-7/xlnet_Opset16/sliced_196.mlir
```

#### Example 2: Extract operations for %288 with depth 2

```bash
python mlir_slice.py test-onnx-7/xlnet_Opset16/model.torch_onnx.mlir %288 \
  -d 2 \
  -o test-onnx-7/xlnet_Opset16/sliced_288.mlir
```

**Output:**
```
Found 545 constant resources
Found target value: %288 = torch.operator "onnx.Einsum"...
Collected 3 operations
Found 2 external inputs
Found 1 required constants: {'_layer.0.rel_attn.o'}
Sliced module written to test-onnx-7/xlnet_Opset16/sliced_288.mlir
```

## How It Works

1. **Resource Mapping**: Scans the input MLIR to build a map of all constant resources
2. **Dependency Analysis**: Uses BFS to traverse backward from the target SSA value
3. **Depth Control**: Limits traversal to the specified depth (if provided)
4. **Constant Inclusion**: Automatically includes constants referenced by collected operations
5. **Topological Sort**: Orders operations to maintain valid SSA form
6. **Resource Filtering**: Includes only the dialect resources actually used
7. **Structure Generation**: Builds a valid MLIR module with properly formatted resources

## Output Structure

The generated MLIR file has the following structure:

```mlir
module {
  func.func @sliced_func(%arg0: !torch.vtensor<...>, ...) -> !torch.vtensor<...> {
    %operation1 = ...
    %operation2 = ...
    %target = ...
    return %target : !torch.vtensor<...>
  }
}
{-#
  dialect_resources: {
    builtin: {
      resource_name: "0x<hex_data>",
      ...
    }
  }
#-}
```

**Key features:**
- Resources are placed **outside** the module block
- Resources are **inside** the `builtin: {}` dictionary
- No trailing comma on the last resource entry
- All operations in topologically sorted order

## Verification

Validate the generated MLIR file using `iree-opt`:

```bash
iree-opt --verify-diagnostics output.mlir
```

Successful verification produces no errors:

```bash
✓ Verification PASSED - No errors found
```

## Use Cases

### Debugging Compilation Failures

When a large model fails to compile, slice to the problematic operation:

```bash
python mlir_slice.py large_model.mlir %failing_op -d 5 -o debug.mlir
```

### Minimizing Test Cases

Create minimal reproducible examples for bug reports:

```bash
python mlir_slice.py full_model.mlir %error_op -d 3 -o minimal_repro.mlir
```

### Analyzing Subgraphs

Study specific operation sequences in isolation:

```bash
python mlir_slice.py model.mlir %intermediate_value -d 10 -o subgraph.mlir
```

## Implementation Details

### Resource Handling

The slicer intelligently filters `dialect_resources` blocks:

- **Before slicing**: May have hundreds of resources (e.g., 545 resources)
- **After slicing**: Only includes resources referenced by collected operations
- **Example**: If only 1 constant is used, only 1 resource is included

### Depth Parameter

- **`-d 1`**: Only the target operation
- **`-d 2`**: Target + immediate dependencies
- **`-d N`**: Target + N levels of dependencies
- **No `-d`**: Include all dependencies (full backward trace)

### Large Files

The slicer handles large neural network weight tensors efficiently:

- Resources can be multi-megabyte hex dumps (e.g., 4.7MB for `_layer.0.rel_attn.o`)
- Each resource is kept as a single line in the output
- No special handling needed - the tool preserves exact byte content

## Troubleshooting

### "SSA value not found"

Ensure the SSA value exists in the input file and is formatted correctly (e.g., `%288` not `288`).

### "torch-mlir import error"

Install torch-mlir bindings:

```bash
pip install --no-deps -r torch_mlir_requirements.txt
```

### "Invalid MLIR output"

If `iree-opt` fails, check:
1. Input file is valid MLIR
2. Target SSA value exists in the input
3. All dependencies are resolvable

## Limitations

- Currently supports MLIR with `torch.operator` and `onnx` dialects
- Requires torch-mlir Python bindings for MLIR parsing
- Resources must be in `{-# dialect_resources: ... #-}` format

## License

Part of the AMDSHARK TestSuite project.
