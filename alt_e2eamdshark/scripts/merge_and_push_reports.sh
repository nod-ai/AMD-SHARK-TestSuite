#!/bin/bash
# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Script to merge reports and push status artifacts for e2eamdshark test suite
# Usage: ./merge_and_push_reports.sh --backend <rocm|llvm-cpu> [options]

set -e

# Default values
BACKEND=""
REPORTS_DIR="./e2eamdshark-reports"
TEST_SUITE_DIR="./test-suite"
VENV_DIR="./report_venv_alt"
DRY_RUN=false
SKIP_GIT=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 --backend <backend> [options]"
    echo ""
    echo "Required arguments:"
    echo "  --backend, -b <backend>     Backend to use (rocm or llvm-cpu)"
    echo ""
    echo "Optional arguments:"
    echo "  --reports-dir <path>        Path to e2eamdshark-reports directory (default: ./e2eamdshark-reports)"
    echo "  --test-suite-dir <path>     Path to test-suite directory (default: ./test-suite)"
    echo "  --venv-dir <path>           Path to virtual environment (default: ./report_venv_alt)"
    echo "  --dry-run                   Print commands without executing them"
    echo "  --skip-git                  Skip git operations (add, commit, push)"
    echo "  --help, -h                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --backend rocm"
    echo "  $0 --backend llvm-cpu --dry-run"
    echo "  $0 -b rocm --reports-dir /path/to/reports --skip-git"
    exit 1
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend|-b)
            BACKEND="$2"
            shift 2
            ;;
        --reports-dir)
            REPORTS_DIR="$2"
            shift 2
            ;;
        --test-suite-dir)
            TEST_SUITE_DIR="$2"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-git)
            SKIP_GIT=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$BACKEND" ]; then
    log_error "Backend is required"
    usage
fi

if [ "$BACKEND" != "rocm" ] && [ "$BACKEND" != "llvm-cpu" ]; then
    log_error "Invalid backend: $BACKEND. Must be 'rocm' or 'llvm-cpu'"
    exit 1
fi

# Validate directories exist
if [ ! -d "$REPORTS_DIR" ]; then
    log_error "Reports directory not found: $REPORTS_DIR"
    exit 1
fi

if [ ! -d "$TEST_SUITE_DIR" ]; then
    log_error "Test suite directory not found: $TEST_SUITE_DIR"
    exit 1
fi

log_info "Starting merge and push for backend: $BACKEND"
log_info "Reports directory: $REPORTS_DIR"
log_info "Test suite directory: $TEST_SUITE_DIR"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    log_info "Activating virtual environment: $VENV_DIR"
    if [ "$DRY_RUN" = false ]; then
        source "$VENV_DIR/bin/activate"
    else
        echo "[DRY-RUN] source $VENV_DIR/bin/activate"
    fi
else
    log_warn "Virtual environment not found: $VENV_DIR. Assuming Python is available in PATH."
fi

MERGE_SCRIPT="$TEST_SUITE_DIR/alt_e2eamdshark/utils/merge_dicts.py"

if [ ! -f "$MERGE_SCRIPT" ]; then
    log_error "Merge script not found: $MERGE_SCRIPT"
    exit 1
fi

# ============================================================================
# MERGE REPORTS
# ============================================================================
log_info "Starting report merge operations..."

# Merge vai-hf-cnn-fp32 shards
log_info "Merging vai-hf-cnn-fp32 shards..."
run_cmd python "$MERGE_SCRIPT" \
    --sources "$REPORTS_DIR/ci_reports_${BACKEND}_vai-hf-cnn-fp32-shard1_unique_onnx_json/vai-hf-cnn-fp32-shard1_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_vai-hf-cnn-fp32-shard2_unique_onnx_json/vai-hf-cnn-fp32-shard2_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_vai-hf-cnn-fp32-shard3_unique_onnx_json/vai-hf-cnn-fp32-shard3_unique.json" \
    --output "$REPORTS_DIR/vai-hf-cnn-fp32_unique.json" \
    --report --report-file "$REPORTS_DIR/vai-hf-cnn-fp32_unique.md"

# Merge vai-int8-p0p1 shards
log_info "Merging vai-int8-p0p1 shards..."
run_cmd python "$MERGE_SCRIPT" \
    --sources "$REPORTS_DIR/ci_reports_${BACKEND}_vai-int8-p0p1-shard1_unique_onnx_json/vai-int8-p0p1-shard1_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_vai-int8-p0p1-shard2_unique_onnx_json/vai-int8-p0p1-shard2_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_vai-int8-p0p1-shard3_unique_onnx_json/vai-int8-p0p1-shard3_unique.json" \
    --output "$REPORTS_DIR/vai-int8-p0p1_unique.json" \
    --report --report-file "$REPORTS_DIR/vai-int8-p0p1_unique.md"

# Merge nlp shards
log_info "Merging nlp shards..."
run_cmd python "$MERGE_SCRIPT" \
    --sources "$REPORTS_DIR/ci_reports_${BACKEND}_nlp-shard1_unique_onnx_json/nlp-shard1_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_nlp-shard2_unique_onnx_json/nlp-shard2_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_nlp-shard3_unique_onnx_json/nlp-shard3_unique.json" \
    --output "$REPORTS_DIR/nlp_unique.json" \
    --report --report-file "$REPORTS_DIR/nlp_unique.md"

# Merge hf_onnx_model_zoo_non_legacy shards
log_info "Merging hf_onnx_model_zoo_non_legacy shards..."
run_cmd python "$MERGE_SCRIPT" \
    --sources "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_00_onnx_json/hf_onnx_model_zoo_non_legacy_00.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_01_onnx_json/hf_onnx_model_zoo_non_legacy_01.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_02_onnx_json/hf_onnx_model_zoo_non_legacy_02.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_03_onnx_json/hf_onnx_model_zoo_non_legacy_03.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_04_onnx_json/hf_onnx_model_zoo_non_legacy_04.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_05_onnx_json/hf_onnx_model_zoo_non_legacy_05.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_06_onnx_json/hf_onnx_model_zoo_non_legacy_06.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_07_onnx_json/hf_onnx_model_zoo_non_legacy_07.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_08_onnx_json/hf_onnx_model_zoo_non_legacy_08.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_hf_onnx_model_zoo_non_legacy_09_onnx_json/hf_onnx_model_zoo_non_legacy_09.json" \
    --output "$REPORTS_DIR/hf_onnx_model_zoo_non_legacy.json" \
    --report --report-file "$REPORTS_DIR/hf_onnx_model_zoo_non_legacy.md"

# Merge onnx_model_zoo_computer_vision shards
log_info "Merging onnx_model_zoo_computer_vision shards..."
run_cmd python "$MERGE_SCRIPT" \
    --sources "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_computer_vision_1_unique_onnx_json/onnx_model_zoo_computer_vision_1_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_computer_vision_2_unique_onnx_json/onnx_model_zoo_computer_vision_2_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_computer_vision_3_unique_onnx_json/onnx_model_zoo_computer_vision_3_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_computer_vision_4_unique_onnx_json/onnx_model_zoo_computer_vision_4_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_computer_vision_5_unique_onnx_json/onnx_model_zoo_computer_vision_5_unique.json" \
    --output "$REPORTS_DIR/onnx_model_zoo_computer_vision_unique.json" \
    --report --report-file "$REPORTS_DIR/onnx_model_zoo_computer_vision_unique.md"

# Merge all reports into combined report
log_info "Creating combined report..."
run_cmd python "$MERGE_SCRIPT" \
    --sources "$REPORTS_DIR/vai-int8-p0p1_unique.json" \
    "$REPORTS_DIR/vai-hf-cnn-fp32_unique.json" \
    "$REPORTS_DIR/nlp_unique.json" \
    "$REPORTS_DIR/onnx_model_zoo_computer_vision_unique.json" \
    "$REPORTS_DIR/hf_onnx_model_zoo_non_legacy.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_amdshark-test-suite_unique_onnx_json/amdshark-test-suite_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_vai-vision-int8_unique_onnx_json/vai-vision-int8_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_migraphx_unique_onnx_json/migraphx_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_gen_ai_unique_onnx_json/onnx_model_zoo_gen_ai_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_graph_ml_unique_onnx_json/onnx_model_zoo_graph_ml_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_nlp_unique_onnx_json/onnx_model_zoo_nlp_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_validated_text_unique_onnx_json/onnx_model_zoo_validated_text_unique.json" \
    "$REPORTS_DIR/ci_reports_${BACKEND}_onnx_model_zoo_validated_vision_unique_onnx_json/onnx_model_zoo_validated_vision_unique.json" \
    --output "$REPORTS_DIR/combined_reports_unique.json" \
    --report --report-file "$REPORTS_DIR/combined_reports_unique.md"

log_info "Report merge completed!"

# ============================================================================
# PUSH STATUS ARTIFACTS
# ============================================================================
log_info "Starting push status artifacts..."

# Change to reports directory
pushd "$REPORTS_DIR" > /dev/null

# Configure git
if [ "$SKIP_GIT" = false ]; then
    log_info "Configuring git..."
    run_cmd git config user.name "GitHub Actions Bot"
    run_cmd git config user.email "<>"
    run_cmd git pull
fi

# Get current date
DATE=$(date '+%Y-%m-%d')
log_info "Using date: $DATE"

# Define report directories and their source locations
# Format: "target_dir:source_file_prefix:source_type"
# source_type: "merged" = root directory, "ci" = CI artifact subdirectory
REPORT_MAPPINGS=(
    "vai-hf-cnn-fp32_unique:vai-hf-cnn-fp32_unique:merged"
    "vai-int8-p0p1_unique:vai-int8-p0p1_unique:merged"
    "nlp_unique:nlp_unique:merged"
    "hf_onnx_model_zoo_non_legacy:hf_onnx_model_zoo_non_legacy:merged"
    "onnx_model_zoo_computer_vision_unique:onnx_model_zoo_computer_vision_unique:merged"
    "combined-reports_unique:combined_reports_unique:merged"
    "amdshark-test-suite_unique:amdshark-test-suite_unique:ci"
    "vai-vision-int8_unique:vai-vision-int8_unique:ci"
    "migraphx_unique:migraphx_unique:ci"
    "onnx_model_zoo_gen_ai_unique:onnx_model_zoo_gen_ai_unique:ci"
    "onnx_model_zoo_graph_ml_unique:onnx_model_zoo_graph_ml_unique:ci"
    "onnx_model_zoo_nlp_unique:onnx_model_zoo_nlp_unique:ci"
    "onnx_model_zoo_validated_text_unique:onnx_model_zoo_validated_text_unique:ci"
    "onnx_model_zoo_validated_vision_unique:onnx_model_zoo_validated_vision_unique:ci"
)

# Create directory structure and copy files
log_info "Creating directory structure and copying report files..."

for mapping in "${REPORT_MAPPINGS[@]}"; do
    # Parse the mapping
    IFS=':' read -r target_dir source_prefix source_type <<< "$mapping"
    
    # Create target directory
    run_cmd mkdir -p "${DATE}/ci_reports_onnx/${BACKEND}/${target_dir}"
    
    # Determine source path based on type
    if [ "$source_type" = "merged" ]; then
        source_dir="."
        source_file="${source_prefix}.md"
    else
        source_dir="ci_reports_${BACKEND}_${source_prefix}_onnx_md"
        source_file="${source_dir}/${source_prefix}.md"
    fi
    
    # Copy main summary file
    if [ -f "$source_file" ]; then
        run_cmd cp "$source_file" "${DATE}/ci_reports_onnx/${BACKEND}/${target_dir}/summary.md"
    else
        log_warn "Source file not found: $source_file"
    fi
    
    # Copy error report files (*_errors.md)
    if [ "$source_type" = "merged" ]; then
        if ls "${source_prefix}_"*_errors.md 1> /dev/null 2>&1; then
            run_cmd cp "${source_prefix}_"*_errors.md "${DATE}/ci_reports_onnx/${BACKEND}/${target_dir}/"
        fi
    else
        if ls "${source_dir}/${source_prefix}_"*_errors.md 1> /dev/null 2>&1; then
            run_cmd cp "${source_dir}/${source_prefix}_"*_errors.md "${DATE}/ci_reports_onnx/${BACKEND}/${target_dir}/"
        fi
    fi
done

# Git operations
if [ "$SKIP_GIT" = false ]; then
    log_info "Committing and pushing changes..."
    run_cmd git add "$DATE"
    run_cmd git commit -m "add CI status reports for e2eamdshark for ${BACKEND}"
    run_cmd git push origin main
else
    log_warn "Skipping git operations (--skip-git flag set)"
fi

popd > /dev/null

log_info "=========================================="
log_info "Merge and push completed successfully!"
log_info "Backend: $BACKEND"
log_info "Date: $DATE"
log_info "=========================================="

