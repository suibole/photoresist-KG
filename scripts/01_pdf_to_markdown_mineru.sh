#!/bin/bash

set -euo pipefail

INPUT_BASE_DIR="${INPUT_BASE_DIR:-}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"

STORAGE_DIRS=(
    "storage1" "storage2" "storage3" "storage4" "storage5" "storage6"
    "storage7" "storage8" "storage9" "storage10" "storage11" "storage12"
)

usage() {
    echo "Usage: bash scripts/01_pdf_to_markdown_mineru.sh --input-base-dir <pdf_root> --output-base-dir <markdown_root>"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-base-dir)
            INPUT_BASE_DIR="${2:-}"
            shift 2
            ;;
        --output-base-dir)
            OUTPUT_BASE_DIR="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_BASE_DIR" || -z "$OUTPUT_BASE_DIR" ]]; then
    usage
    exit 1
fi

mkdir -p "$OUTPUT_BASE_DIR"

LOG_DIR="$OUTPUT_BASE_DIR/logs"
mkdir -p "$LOG_DIR"

FAILED_PDF_DIR="$OUTPUT_BASE_DIR/failed_pdfs"
mkdir -p "$FAILED_PDF_DIR"

export PARALLEL_HOME="$HOME/.parallel"
mkdir -p "$PARALLEL_HOME"
touch "$PARALLEL_HOME/will-cite"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

process_storage() {
    local storage_dir=$1
    local input_dir="$INPUT_BASE_DIR/$storage_dir"
    local output_dir="$OUTPUT_BASE_DIR/$storage_dir"
    
    mkdir -p "$output_dir"
    
    local storage_log="$LOG_DIR/${storage_dir}_processing.log"
    
    local pdf_count=$(find "$input_dir" -name "*.pdf" | wc -l)
    
    local processed_count=0
    local failed_count=0
    
    while IFS= read -r -d '' pdf_file; do
        local filename=$(basename "$pdf_file" .pdf)
        local article_output_dir="$output_dir/$filename"
        mkdir -p "$article_output_dir"
        
        ((processed_count++))
        
        local temp_log=$(mktemp)
        if mineru -p "$pdf_file" -o "$article_output_dir" --method auto --gpu 2>&1 | tee "$temp_log" >> "$storage_log"; then
            if grep -q "ERROR\|CUDA out of memory\|OutOfMemoryError" "$temp_log"; then
                echo "$pdf_file" >> "$LOG_DIR/failed_files.txt"
                
                local failed_storage_dir="$FAILED_PDF_DIR/$storage_dir"
                mkdir -p "$failed_storage_dir"
                cp "$pdf_file" "$failed_storage_dir/"
                
                rm -rf "$article_output_dir"
                ((failed_count++))
            fi
        else
            echo "$pdf_file" >> "$LOG_DIR/failed_files.txt"
            
            local failed_storage_dir="$FAILED_PDF_DIR/$storage_dir"
            mkdir -p "$failed_storage_dir"
            cp "$pdf_file" "$failed_storage_dir/"
            
            rm -rf "$article_output_dir"
            ((failed_count++))
        fi
        
        rm -f "$temp_log"
        
        echo "$processed_count" > "$LOG_DIR/${storage_dir}_progress.txt"
        echo "$failed_count" > "$LOG_DIR/${storage_dir}_failed.txt"
        
    done < <(find "$input_dir" -name "*.pdf" -print0)
    
    echo "completed $processed_count/$pdf_count, failed $failed_count" > "$LOG_DIR/${storage_dir}_status.txt"
}

export -f process_storage
export INPUT_BASE_DIR OUTPUT_BASE_DIR LOG_DIR FAILED_PDF_DIR

start_time=$(date '+%Y-%m-%d %H:%M:%S')

for storage_dir in "${STORAGE_DIRS[@]}"; do
    echo "0" > "$LOG_DIR/${storage_dir}_progress.txt"
    echo "0" > "$LOG_DIR/${storage_dir}_failed.txt"
    echo "not started" > "$LOG_DIR/${storage_dir}_status.txt"
done

parallel --silent -j6 --joblog "$LOG_DIR/joblog.txt" process_storage ::: "${STORAGE_DIRS[@]}"

end_time=$(date '+%Y-%m-%d %H:%M:%S')

echo "=== Processing Summary ===" > "$LOG_DIR/summary.txt"
total_input=0
total_output=0
total_failed=0
for storage_dir in "${STORAGE_DIRS[@]}"; do
    input_dir="$INPUT_BASE_DIR/$storage_dir"
    pdf_count=$(find "$input_dir" -name "*.pdf" | wc -l)
    output_dir="$OUTPUT_BASE_DIR/$storage_dir"
    processed_count=$(find "$output_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    
    failed_count=0
    if [[ -f "$LOG_DIR/${storage_dir}_failed.txt" ]]; then
        failed_count=$(cat "$LOG_DIR/${storage_dir}_failed.txt")
    fi
    
    echo "$storage_dir: input $pdf_count, success $processed_count, failed $failed_count" >> "$LOG_DIR/summary.txt"
    total_input=$((total_input + pdf_count))
    total_output=$((total_output + processed_count))
    total_failed=$((total_failed + failed_count))
done

echo "Total: input $total_input, success $total_output, failed $total_failed" >> "$LOG_DIR/summary.txt"
cat "$LOG_DIR/summary.txt"

if [[ -f "$LOG_DIR/failed_files.txt" ]]; then
    failed_count=$(wc -l < "$LOG_DIR/failed_files.txt")
    echo "Warning: $failed_count files failed"
    echo "Failed files list: $LOG_DIR/failed_files.txt"
    echo "Failed PDFs saved to: $FAILED_PDF_DIR"
fi