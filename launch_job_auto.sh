#!/bin/bash

JOB_NAME="yh_auto_resubmit"
OUTPUT_DIR="slurm_output"
YAML_CONFIG="examples/train_full/qwen2_5_vl_full_sft.yaml"
CHECKPOINT_DIR_PATTERN="saves/qwen2_5_vl-3b/full/sft/checkpoint-*"
SLEEP_DURATION="6h"
CHECK_INTERVAL="10s" # How often to check if the job has started
CONDA_PATH="/gpfs/u/home/LMCG/LMCGhazh/scratch/miniconda3x86/etc/profile.d/conda.sh"
CONDA_ENV="vlm-r1"
WANDB_API_KEY="28b3c634497c0dc6c16767729d4719b1012a94f2"

mkdir -p $OUTPUT_DIR

while true; do
    echo "$(date): Checking for latest checkpoint..."
    # Find the most recently modified checkpoint directory
    LATEST_CHECKPOINT=$(ls -td ${CHECKPOINT_DIR_PATTERN} 2>/dev/null | head -n 1)

    # Base command
    CMD="llamafactory-cli train ${YAML_CONFIG}"

    # Append resume argument if checkpoint exists
    if [[ -d "$LATEST_CHECKPOINT" ]]; then
        echo "$(date): Found checkpoint: $LATEST_CHECKPOINT. Resuming."
        CMD="${CMD} --resume_from_checkpoint ${LATEST_CHECKPOINT}"
    else
        echo "$(date): No checkpoint found. Starting new training."
    fi

    echo "$(date): Submitting job with command: ${CMD}"

    # Submit the job using sbatch and capture the output
    SUBMISSION_OUTPUT=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -o ${OUTPUT_DIR}/sft_%j.out
#SBATCH -e ${OUTPUT_DIR}/sft_%j.err
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8

echo "Job ID: \${SLURM_JOB_ID}"
echo "Running on node: \$(hostname)"
echo "Loading conda environment..."
source ${CONDA_PATH}
conda activate ${CONDA_ENV}
echo "Conda environment activated."
export WANDB_API_KEY=${WANDB_API_KEY}

echo "Executing command: ${CMD}"
${CMD}
echo "Command finished with exit code \$?"
EOF
    )

    # Check if submission was successful by parsing sbatch output
    if [[ $SUBMISSION_OUTPUT =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        JOB_ID=${BASH_REMATCH[1]}
        echo "$(date): Job submitted successfully with ID: $JOB_ID"

        # Wait for the job to start by checking for output/error files
        OUT_FILE="${OUTPUT_DIR}/sft_${JOB_ID}.out"
        ERR_FILE="${OUTPUT_DIR}/sft_${JOB_ID}.err"
        echo "$(date): Waiting for job $JOB_ID to start (checking for $OUT_FILE and $ERR_FILE)..."
        while true; do
            # Check if both files exist. Using -f to check for regular files.
            if [[ -f "$OUT_FILE" && -f "$ERR_FILE" ]]; then
                echo "$(date): Job $JOB_ID appears to have started (output files found)."
                break # Exit the waiting loop
            else
                # Optional: Check job status using squeue
                # squeue_status=$(squeue -j $JOB_ID -h -o %T 2>/dev/null)
                # echo "$(date): Job status: ${squeue_status:-'Not Found/Completed'}. Output files not yet found. Waiting ${CHECK_INTERVAL}..."
                echo "$(date): Output files not yet found. Waiting ${CHECK_INTERVAL}..."
                sleep ${CHECK_INTERVAL}
            fi
        done

        echo "$(date): Sleeping for ${SLEEP_DURATION}..."
        sleep ${SLEEP_DURATION}
        echo "$(date): Woke up, preparing next submission."
    else
        echo "$(date): Error submitting job!"
        echo "sbatch output: $SUBMISSION_OUTPUT"
        echo "$(date): Exiting due to submission error."
        exit 1 # Exit the script if submission fails
    fi
done

echo "$(date): Script finished (this line might not be reached in infinite loop)."