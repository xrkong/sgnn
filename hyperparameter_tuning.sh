#!/bin/bash

# Multi-Scale GNN Hyperparameter Tuning Script
# Systematically tests combinations of LAYERS, HIDDEN_DIM, RADIUS_MULTIPLIERS, BATCH_SIZE
# on the Taylor Impact 2D dataset

set -euo pipefail

# Set base parameters
BASE_CMD="CUDA_VISIBLE_DEVICES=0 python gns/multi_scale/train_multi_scale.py"
# NOTE: Do NOT include --layers/--hidden_dim/--batch_size here (they are swept)
BASE_ARGS="--mode=train \
  --data_path=./datasets/taylor_impact_2d/data_processed/ \
  --model_path=./models/Taylor_impact_2d/ \
  --output_path=./rollouts/Taylor_impact_2d/ \
  --ntraining_steps=10000 \
  --noise_std=0.02 \
  --lr_init=0.001 \
  --lr_decay_steps=2000 \
  --dim=2 \
  --project_name=Taylor_impact_2d_MS \
  --nsave_steps=1000 \
  -log=True"

# Hyperparameter ranges to test (focused set for practical tuning)
LAYERS="5 10 15"
HIDDEN_DIM="64 128 256"
RADIUS_MULTIPLIERS="3 4"
BATCH_SIZE="1 4 16 64"

# Additional parameters
NUM_SCALES=3
WINDOW_SIZE=3

# Create results directory
mkdir -p hyperparameter_tuning_results
RESULTS_FILE="hyperparameter_tuning_results/tuning_results.txt"

# Initialize results file
{
  echo "Multi-Scale GNN Hyperparameter Tuning Results"
  echo "============================================="
  echo "Started at: $(date)"
  echo ""
} > "$RESULTS_FILE"

# Counters
total_runs=0
successful_runs=0

echo "Starting hyperparameter tuning..."

# Totals
layers_count=$(echo "$LAYERS" | wc -w)
hidden_count=$(echo "$HIDDEN_DIM" | wc -w)
radius_count=$(echo "$RADIUS_MULTIPLIERS" | wc -w)
batch_count=$(echo "$BATCH_SIZE" | wc -w)
total_combinations=$((layers_count * hidden_count * radius_count * batch_count))
echo "Total combinations to test: $total_combinations"
echo ""

# Loop through all combinations
for layers in $LAYERS; do
  for hidden_dim in $HIDDEN_DIM; do
    for radius_mult in $RADIUS_MULTIPLIERS; do
      for batch in $BATCH_SIZE; do
        total_runs=$((total_runs + 1))

        # Unique run name
        run_name="MS_L${layers}_H${hidden_dim}_B${batch}_RM${radius_mult}"

        # Construct full command
        full_cmd="$BASE_CMD $BASE_ARGS \
          --layers=$layers \
          --hidden_dim=$hidden_dim \
          --batch_size=$batch \
          --radius_multiplier=$radius_mult \
          --num_scales=$NUM_SCALES \
          --window_size=$WINDOW_SIZE \
          --run_name=$run_name"

        echo "=========================================="
        echo "Run $total_runs/$total_combinations: $run_name"
        echo "Layers: $layers | Hidden Dim: $hidden_dim | Batch: $batch | Radius Multiplier: $radius_mult"
        echo "=========================================="

        # Log run details
        {
          echo "Run $total_runs: $run_name"
          echo "  Layers: $layers"
          echo "  Hidden Dim: $hidden_dim"
          echo "  Batch Size: $batch"
          echo "  Radius Multiplier: $radius_mult"
          echo "  Started at: $(date)"
        } >> "$RESULTS_FILE"

        # Execute
        start_time=$(date +%s)
        if eval "$full_cmd"; then
          end_time=$(date +%s)
          duration=$((end_time - start_time))
          successful_runs=$((successful_runs + 1))

          echo "✅ Run $total_runs completed successfully in ${duration}s"
          {
            echo "  Status: SUCCESS"
            echo "  Duration: ${duration}s"
            echo ""
          } >> "$RESULTS_FILE"
        else
          end_time=$(date +%s)
          duration=$((end_time - start_time))

          echo "❌ Run $total_runs failed after ${duration}s"
          {
            echo "  Status: FAILED"
            echo "  Duration: ${duration}s"
            echo ""
          } >> "$RESULTS_FILE"
        fi

        # Small delay to avoid resource conflicts
        sleep 5
      done
    done
  done
done

# Final summary
echo "=========================================="
echo "HYPERPARAMETER TUNING COMPLETED"
echo "=========================================="
echo "Total runs: $total_runs"
echo "Successful runs: $successful_runs"
echo "Failed runs: $((total_runs - successful_runs))"
echo "Success rate: $(( successful_runs * 100 / (total_runs == 0 ? 1 : total_runs) ))%"
echo ""

# Log final summary
{
  echo "=========================================="
  echo "FINAL SUMMARY"
  echo "=========================================="
  echo "Completed at: $(date)"
  echo "Total runs: $total_runs"
  echo "Successful runs: $successful_runs"
  echo "Failed runs: $((total_runs - successful_runs))"
  echo "Success rate: $(( successful_runs * 100 / (total_runs == 0 ? 1 : total_runs) ))%"
} >> "$RESULTS_FILE"

echo "Results saved to: $RESULTS_FILE"
echo "Check WandB dashboard for detailed training metrics and model comparisons."
