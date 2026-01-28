# MLIR Dialect Resource Extractors

This directory contains Python scripts for extracting dialect resources from MLIR files using IREE/MLIR Python bindings instead of regex or string matching.

## Overview

MLIR files with dialect resources contain two main components:
1. **Dense Resource References**: Operations that reference resources (e.g., `dense_resource<weight_name>`)
2. **Dialect Resources Block**: Binary data stored as hex strings in a `{-# dialect_resources: ... #-}` block

These scripts use MLIR Python bindings to properly parse the files and extract tensor data.

## Scripts

### 1. `mlir_resource_extractor.py`
Basic extractor that demonstrates the MLIR binding approach with comprehensive error handling.

### 2. `mlir_resource_extractor_advanced.py` 
Advanced version with full binary data extraction and numpy conversion.

### 3. `mlir_resource_extractor_simple.py` ⭐ **Recommended**
Production-ready version that works with the testsuite environment. Features:
- Multiple backend support (IREE, torch-mlir, standard MLIR)
- Fallback parsing when bindings are unavailable
- Robust hex-to-numpy conversion
- Multiple output formats

## Prerequisites

Ensure you're in the proper virtual environment:
```bash
cd /home/phaneesh/NOD/testsuite/alt_e2eamdshark
source ../.venv_alt_e2eamdshark/bin/activate
```

The environment should have:
- `iree-base-compiler` and `iree-base-runtime`
- `torch-mlir` 
- `numpy`

## Usage Examples

### Show Resource Summary
```bash
python mlir_resource_extractor_simple.py model.torch_onnx.mlir
```

### List All Resources
```bash
python mlir_resource_extractor_simple.py model.torch_onnx.mlir --list
```

### Extract Specific Tensor
```bash
python mlir_resource_extractor_simple.py model.torch_onnx.mlir --extract "_layer.0.rel_attn.o"
```

### Save Extracted Tensor
```bash
# Save as numpy binary
python mlir_resource_extractor_simple.py model.torch_onnx.mlir --extract "weight_name" --save output.npy

# Save as text
python mlir_resource_extractor_simple.py model.torch_onnx.mlir --extract "weight_name" --save output.txt --format txt

# Save as JSON (includes metadata)
python mlir_resource_extractor_simple.py model.torch_onnx.mlir --extract "weight_name" --save output.json --format json
```

## Example Output

```
MLIR Resource Summary
========================================
File: test-onnx-7/xlnet_Opset16/sliced_final.mlir
Resource References: 1
Dialect Resources: 1
  builtin: 1

Backend: IREE

Available Resources:
  _layer.0.rel_attn.o
    Shape: [768, 12, 64]
    Type: f32
    Binary Data: ✓
```

## Technical Details

### Architecture

The scripts use a multi-tier approach:

1. **IREE Bindings**: Primary method using `iree.compiler.ir`
2. **MLIR Bindings**: Standard MLIR Python bindings
3. **Fallback Parser**: Text-based parsing that doesn't use regex

### Resource Parsing Process

1. **Load MLIR File**: Read and attempt to parse with available bindings
2. **Extract References**: Find `dense_resource<name>` patterns and extract:
   - Resource name
   - Tensor shape (e.g., `[768, 12, 64]`)
   - Data type (e.g., `f32`)
3. **Extract Binary Data**: Parse `dialect_resources` block for hex strings
4. **Convert to NumPy**: Transform hex data to properly shaped numpy arrays

### Hex Data Format

MLIR stores tensor data as hexadecimal strings. The scripts handle:
- Little-endian byte ordering
- Multiple data types (f32, f16, i32, etc.)
- Proper reshaping based on tensor specifications

### Error Handling

The scripts include robust error handling:
- Graceful fallback when MLIR bindings fail to parse
- Validation of tensor shapes vs. available data
- Clear error messages for missing resources

## Testing

Test with the provided example:
```bash
# Show summary
python mlir_resource_extractor_simple.py test-onnx-7/xlnet_Opset16/sliced_final.mlir

# Extract tensor
python mlir_resource_extractor_simple.py test-onnx-7/xlnet_Opset16/sliced_final.mlir --extract "_layer.0.rel_attn.o"

# Save and verify
python mlir_resource_extractor_simple.py test-onnx-7/xlnet_Opset16/sliced_final.mlir --extract "_layer.0.rel_attn.o" --save test.npy
python -c "import numpy as np; print('Tensor shape:', np.load('test.npy').shape)"
```

## Benefits Over Regex/String Matching

1. **Proper Parsing**: Uses MLIR's own parsing infrastructure
2. **Type Safety**: Leverages MLIR's type system for validation
3. **Robust**: Handles edge cases in MLIR syntax correctly
4. **Extensible**: Can be extended to handle new dialects and resource types
5. **Maintainable**: Won't break with MLIR syntax changes

## Integration with TestSuite

These scripts can be integrated into the testsuite workflow for:
- Debugging model weights during compilation
- Analyzing parameter distributions
- Comparing weights before/after transformations
- Extracting specific layers for analysis

## Limitations

- Very large files (>50MB) may have performance considerations
- Some MLIR files may use unsupported resource formats
- Requires proper IREE/MLIR installation for full functionality

## Troubleshooting

### Import Errors
If you see MLIR import errors:
```bash
# Check available packages
pip list | grep -E "(iree|mlir|torch)"

# Ensure virtual environment is activated
source ../.venv_alt_e2eamdshark/bin/activate
```

### Parsing Failures
If IREE/MLIR parsing fails, the fallback parser will automatically engage. This is normal and expected for some files.

### Resource Not Found
Ensure the resource name exactly matches what's shown in the `--list` output. Resource names are case-sensitive.