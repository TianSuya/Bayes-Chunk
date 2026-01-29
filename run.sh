#!/bin/bash

# AnyEdit++ background run script
#
# Usage:
#   1. Edit the parameter configuration section below, then run: ./run.sh
#   2. Or override via command-line: ./run.sh --alg_name=AlphaEdit_ARE --ds_name=unke
#
# ==========================================
# Parameter configuration - edit default values here
# ==========================================

# Algorithm name (options: AlphaEdit, AlphaEdit_ARE, MEMIT, MEMIT_ARE, unke, unke_ARE)
ALG_NAME=""

# Model name
MODEL_NAME=""

# Hyperparameters filename (under hparams/<alg_name>/)
HPARAMS_FNAME=""

# Dataset name (options: unke, cf, mquake, editevery,fake)
DS_NAME=""

# Dataset size limit
DATASET_SIZE_LIMIT=1000

# Number of edits per batch
NUM_EDITS=1

# Sequential edit (true/false)
SEQUENTIAL=false

# Skip generation tests (true/false)
SKIP_GENERATION_TESTS=false

# Conserve memory (true/false)
CONSERVE_MEMORY=false

# Use cache (true/false)
USE_CACHE=false

# Run ID to continue from (leave empty to start fresh)
CONTINUE_FROM_RUN=""

# Generation test interval
GENERATION_TEST_INTERVAL=1

# ==========================================
# Script logic below - usually no need to modify
# ==========================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build arguments array
ARGS=(
    "--alg_name=$ALG_NAME"
    "--model_name=$MODEL_NAME"
    "--hparams_fname=$HPARAMS_FNAME"
    "--ds_name=$DS_NAME"
    "--dataset_size_limit=$DATASET_SIZE_LIMIT"
    "--num_edits=$NUM_EDITS"
)

# Add optional arguments
if [ "$SEQUENTIAL" = "true" ]; then
    ARGS+=("--sequential")
fi

if [ "$SKIP_GENERATION_TESTS" = "true" ]; then
    ARGS+=("--skip_generation_tests")
fi

if [ "$CONSERVE_MEMORY" = "true" ]; then
    ARGS+=("--conserve_memory")
fi

if [ "$USE_CACHE" = "true" ]; then
    ARGS+=("--use_cache")
fi

if [ -n "$CONTINUE_FROM_RUN" ]; then
    ARGS+=("--continue_from_run=$CONTINUE_FROM_RUN")
fi

if [ "$GENERATION_TEST_INTERVAL" != "1" ]; then
    ARGS+=("--generation_test_interval=$GENERATION_TEST_INTERVAL")
fi

# If command-line arguments are provided, they override script configuration
# Format: --alg_name=xxx
if [ $# -gt 0 ]; then
    echo "Command-line arguments detected, overriding script configuration"
    
    # Parse command-line arguments and override variables
    for arg in "$@"; do
        case "$arg" in
            --alg_name=*)
                ALG_NAME="${arg#*=}"
                ;;
            --model_name=*)
                MODEL_NAME="${arg#*=}"
                ;;
            --hparams_fname=*)
                HPARAMS_FNAME="${arg#*=}"
                ;;
            --ds_name=*)
                DS_NAME="${arg#*=}"
                ;;
            --dataset_size_limit=*)
                DATASET_SIZE_LIMIT="${arg#*=}"
                ;;
            --num_edits=*)
                NUM_EDITS="${arg#*=}"
                ;;
            --sequential)
                SEQUENTIAL="true"
                ;;
            --skip_generation_tests)
                SKIP_GENERATION_TESTS="true"
                ;;
            --conserve_memory)
                CONSERVE_MEMORY="true"
                ;;
            --use_cache)
                USE_CACHE="true"
                ;;
            --continue_from_run=*)
                CONTINUE_FROM_RUN="${arg#*=}"
                ;;
            --generation_test_interval=*)
                GENERATION_TEST_INTERVAL="${arg#*=}"
                ;;
        esac
    done
    
    # Rebuild arguments array (using updated variable values)
    ARGS=(
        "--alg_name=$ALG_NAME"
        "--model_name=$MODEL_NAME"
        "--hparams_fname=$HPARAMS_FNAME"
        "--ds_name=$DS_NAME"
        "--dataset_size_limit=$DATASET_SIZE_LIMIT"
        "--num_edits=$NUM_EDITS"
    )
    
    # Re-add optional arguments
    if [ "$SEQUENTIAL" = "true" ]; then
        ARGS+=("--sequential")
    fi
    
    if [ "$SKIP_GENERATION_TESTS" = "true" ]; then
        ARGS+=("--skip_generation_tests")
    fi
    
    if [ "$CONSERVE_MEMORY" = "true" ]; then
        ARGS+=("--conserve_memory")
    fi
    
    if [ "$USE_CACHE" = "true" ]; then
        ARGS+=("--use_cache")
    fi
    
    if [ -n "$CONTINUE_FROM_RUN" ]; then
        ARGS+=("--continue_from_run=$CONTINUE_FROM_RUN")
    fi
    
    if [ "$GENERATION_TEST_INTERVAL" != "1" ]; then
        ARGS+=("--generation_test_interval=$GENERATION_TEST_INTERVAL")
    fi
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Generate log filename (with timestamp and config info)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/evaluate_${ALG_NAME}_${DS_NAME}_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/evaluate_${ALG_NAME}_${DS_NAME}_${TIMESTAMP}.pid"

# Check if a process with the same config is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Warning: Found existing process running (PID: $OLD_PID)"
        echo "To start a new process, remove $PID_FILE first"
        exit 1
    else
        # PID file exists but process does not; remove stale PID file
        rm -f "$PID_FILE"
    fi
fi

# Start Python program (background)
echo "==========================================" | tee -a "$LOG_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "Working dir: $SCRIPT_DIR" | tee -a "$LOG_FILE"
echo "Command: python3 -m experiments.evaluate_uns ${ARGS[*]}" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "PID file: $PID_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo ""

# Show arguments in use
echo "Arguments:"
for arg in "${ARGS[@]}"; do
    echo "  $arg"
done
echo ""

# Run in background, output to log file and terminal
nohup python3 -m experiments.evaluate_uns "${ARGS[@]}" >> "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PID_FILE"
echo "Process started, PID: $PID"
echo "PID saved to: $PID_FILE"
echo "Log file: $LOG_FILE"
echo ""
echo "View log: tail -f $LOG_FILE"
echo "Stop process: kill $PID or use ./stop_evaluate.sh $PID_FILE"

# Wait briefly and check if process started successfully
sleep 2
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Process running OK"
else
    echo "✗ Process failed to start, check log: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi


