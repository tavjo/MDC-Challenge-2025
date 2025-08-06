#!/bin/bash

# Random Forest Training Script for MDC Challenge
# This script runs the enhanced Random Forest training pipeline with proper error handling

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Default parameters
INPUT_CSV="${PROJECT_ROOT}/Data/train/train_data.csv"
OUTPUT_DIR="${PROJECT_ROOT}/artifacts/models"
DUCKDB_PATH="${PROJECT_ROOT}/artifacts/mdc-challenge.db"
TARGET_COL="target"
GROUP_COL=""
SEED=42
N_ITER=100
USE_BALANCED_RF=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run Random Forest training for MDC Challenge with enhanced error handling.

OPTIONS:
    -i, --input CSV_FILE        Input CSV file (default: Data/train/train_data.csv)
    -o, --output OUTPUT_DIR     Output directory for models (default: artifacts/models)
    -d, --duckdb DB_PATH        DuckDB database path (default: artifacts/mdc-challenge.db)
    -t, --target TARGET_COL     Target column name (default: target)
    -g, --group GROUP_COL       Optional group column for StratifiedGroupKFold
    -s, --seed SEED             Random seed (default: 42)
    -n, --n-iter N_ITER         Number of RandomizedSearch iterations (default: 100)
    -b, --balanced              Use BalancedRandomForestClassifier from imblearn
    -h, --help                  Show this help message
    
EXAMPLES:
    # Basic training with default parameters
    $0
    
    # Training with custom parameters
    $0 --input my_data.csv --output my_models --seed 123 --n-iter 200
    
    # Training with balanced Random Forest
    $0 --balanced
    
    # Training with grouping column
    $0 --group article_id

REQUIREMENTS:
    - Python 3.8+
    - Required packages: pandas, scikit-learn, numpy, joblib
    - Optional: imbalanced-learn (for --balanced option)
    - Optional: duckdb (for metadata storage)
EOF
}

# Function to check if file exists
check_file_exists() {
    if [[ ! -f "$1" ]]; then
        print_error "File not found: $1"
        exit 1
    fi
}

# Function to check if directory exists and create if needed
ensure_directory() {
    if [[ ! -d "$1" ]]; then
        print_status "Creating directory: $1"
        mkdir -p "$1" || {
            print_error "Failed to create directory: $1"
            exit 1
        }
    fi
}

# Function to check Python and required packages
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+."
        exit 1
    fi
    
    # Check Python version
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Found Python $python_version"
    
    # Check required packages
    required_packages=("pandas" "numpy" "scikit-learn" "joblib" "scipy")
    missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        print_status "Install them with: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    # Check optional packages
    if [[ "$USE_BALANCED_RF" == true ]]; then
        if ! python -c "import imblearn" &> /dev/null; then
            print_error "imbalanced-learn is required for --balanced option"
            print_status "Install it with: pip install imbalanced-learn"
            exit 1
        fi
    fi
    
    print_success "All dependencies satisfied"
}

# Function to validate input data
validate_data() {
    print_status "Validating input data..."
    
    check_file_exists "$INPUT_CSV"
    
    # Check if file is not empty
    if [[ ! -s "$INPUT_CSV" ]]; then
        print_error "Input CSV file is empty: $INPUT_CSV"
        exit 1
    fi
    
    # Check if target column exists
    if ! head -1 "$INPUT_CSV" | grep -q "$TARGET_COL"; then
        print_error "Target column '$TARGET_COL' not found in CSV header"
        print_status "Available columns:"
        head -1 "$INPUT_CSV" | tr ',' '\n' | nl
        exit 1
    fi
    
    # Get data dimensions
    num_rows=$(($(wc -l < "$INPUT_CSV") - 1))  # Subtract header
    num_cols=$(head -1 "$INPUT_CSV" | tr ',' '\n' | wc -l)
    
    print_status "Data dimensions: $num_rows rows, $num_cols columns"
    
    # Validate minimum requirements
    if [[ $num_rows -lt 10 ]]; then
        print_error "Insufficient data: need at least 10 rows, found $num_rows"
        exit 1
    fi
    
    if [[ $num_cols -lt 2 ]]; then
        print_error "Insufficient features: need at least 1 feature + target, found $num_cols columns"
        exit 1
    fi
    
    print_success "Data validation passed"
}

# Function to run the training
run_training() {
    print_status "Starting Random Forest training..."
    
    # Build command arguments
    cmd_args=(
        "--input_csv" "$INPUT_CSV"
        "--output_dir" "$OUTPUT_DIR"
        "--duckdb_path" "$DUCKDB_PATH"
        "--target_col" "$TARGET_COL"
        "--seed" "$SEED"
        "--n_iter" "$N_ITER"
    )
    
    if [[ -n "$GROUP_COL" ]]; then
        cmd_args+=("--group_col" "$GROUP_COL")
    fi
    
    if [[ "$USE_BALANCED_RF" == true ]]; then
        cmd_args+=("--use_balanced_rf")
    fi
    
    # Print command being executed
    print_status "Executing: python src/training.py ${cmd_args[*]}"
    
    # Run the training script
    if python src/training.py "${cmd_args[@]}"; then
        print_success "Training completed successfully!"
        
        # Show output files
        if [[ -d "$OUTPUT_DIR" ]]; then
            print_status "Generated artifacts:"
            find "$OUTPUT_DIR" -type f -exec ls -lh {} \; | while read -r line; do
                echo "  $line"
            done
        fi
        
        return 0
    else
        print_error "Training failed!"
        return 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_CSV="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--duckdb)
            DUCKDB_PATH="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_COL="$2"
            shift 2
            ;;
        -g|--group)
            GROUP_COL="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -n|--n-iter)
            N_ITER="$2"
            shift 2
            ;;
        -b|--balanced)
            USE_BALANCED_RF=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Random Forest Training Script for MDC Challenge"
    print_status "============================================="
    
    # Show configuration
    print_status "Configuration:"
    echo "  Input CSV: $INPUT_CSV"
    echo "  Output Dir: $OUTPUT_DIR"
    echo "  DuckDB Path: $DUCKDB_PATH"
    echo "  Target Column: $TARGET_COL"
    [[ -n "$GROUP_COL" ]] && echo "  Group Column: $GROUP_COL"
    echo "  Random Seed: $SEED"
    echo "  RandomizedSearch Iterations: $N_ITER"
    echo "  Use Balanced RF: $USE_BALANCED_RF"
    echo
    
    # Pre-flight checks
    check_dependencies
    validate_data
    
    # Ensure output directories exist
    ensure_directory "$OUTPUT_DIR"
    ensure_directory "$(dirname "$DUCKDB_PATH")"
    
    # Run training
    if run_training; then
        print_success "🎉 Training pipeline completed successfully!"
        print_status "Check the output directory for model artifacts: $OUTPUT_DIR"
        exit 0
    else
        print_error "💥 Training pipeline failed!"
        exit 1
    fi
}

# Execute main function
main "$@"