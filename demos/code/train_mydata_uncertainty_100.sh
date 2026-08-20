#!/bin/bash
#SBATCH --job-name=cpn-mydata
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0

PROJECT=/data/cat/ws/mehu311f-cpn_workspace_e1/celldetection
source "$PROJECT/.venv/bin/activate"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

cd "$PROJECT/demos/code"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
START=$(date +%s)
echo "Start time: $(date)"

python - <<'PY'
import torch
import celldetection as cd

print("torch:", torch.__version__)
print("celldetection:", cd.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python mydata_uncertainty_100.py

END=$(date +%s)
echo "End time: $(date)"
echo "Duration: $(( (END - START) / 60 )) minutes"
