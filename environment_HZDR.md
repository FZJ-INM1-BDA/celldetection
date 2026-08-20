

# HZDR RoSI HPC

## Cluster Overview

- **Cluster**: RoSI (Rossendorf Supercomputing Infrastructure)
- **OS**: Ubuntu 22.04.5 LTS
- **Login node**: `rosi5.fz-rossendorf.de`
- **Username**: `hu09`
- **Home quota**: 600G
- **Scheduler**: SLURM
- **Web portal**: [rosi.hzdr.de](https://rosi.hzdr.de/pun/sys/dashboard/) (Open OnDemand)
- **Login node CPU limit**: 300 seconds per process

## SSH Access

### From local machine

```bash
ssh hzdr-rosi
```

### Local SSH config (`~/.ssh/config`)

```
# HZDR RoSI
Host hzdr-rosi
  HostName rosi5.fz-rossendorf.de
  User hu09
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

### VS Code Remote SSH

1. `Cmd+Shift+P` → `Remote-SSH: Connect to Host` → `hzdr-rosi`
2. Open Folder → `/data/home2/hu09/project/celldetection`
3. `Cmd+Shift+P` → `Python: Select Interpreter` → `.venv/bin/python`

------

## GPU Partitions

| Partition  | GPU  | GPUs/node | CPUs   | Memory | Max time |
| ---------- | ---- | --------- | ------ | ------ | -------- |
| `gpu-v100` | V100 | 4         | 48+    | 380G   | 2 days   |
| `gpu-a100` | A100 | 4 or 8    | 32–128 | 1–4TB  | 2 days   |
| `gpu-h100` | H100 | 8         | 192    | 2TB    | 2 days   |
| `gpu-b200` | B200 | 8–32      | 256    | 2.3TB  | 2 days   |

Check partitions:

```bash
sinfo -o "%P %G %c %m %l"
```

------

## Use srun to enter compute node

```bash
srun --partition=gpu-a100 --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash -l
```

------

## Module management on RoSI

```bash
# List available modules
module avail python
module avail cuda

# Load modules
module load python/3.12.4 cuda/12.8

# Other commands (same as Capella)
module list
module --force purge
module unload python/3.12.4
module show python/3.12.4
module spider python
```

------

## Create virtual environment and install dependencies on RoSI

```bash
module load python/3.12.4 cuda/12.8

cd ~/project/celldetection
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
which python
deactivate
pip install -r requirements.txt
pip freeze > requirements.txt
# Install celldetection in editable mode (reads requirements.txt)
pip install -e .
```


------

## Jupyter kernel management on RoSI

```bash
source ~/project/celldetection/.venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name cpn-kernel --display-name="cpn kernel"

jupyter kernelspec list
jupyter kernelspec uninstall cpn-kernel
```

## Submit batch job (sbatch)

```bash
cat > ~/project/train_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=cellseg
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log

module load python/3.12.4 cuda/12.8
source ~/project/celldetection/.venv/bin/activate

cd ~/project/celldetection
python train.py --config config.yaml
EOF

# Submit
sbatch ~/project/train_job.sh

# Monitor
squeue -u $USER

# Cancel
scancel <job_id>

# View output
tail -f <job_id>_out.log
```

------

## Git branch strategy

```bash
# main       → TU Dresden Capella
# hzdr → HZDR RoSI(project)
git branch -r
git branch -a

git push -u origin hzdr

git switch hzdr

git checkout hzdr    # work on HZDR branch
git checkout main          # work on Capella branch

# After making changes on RoSI
git add .
git commit -m "your message"
git push origin hzdr
```

### Git SSH (configured on RoSI)

```bash
# SSH key location: ~/.ssh/id_ed25519
# Remote URL: git@github.com:mengqing-hu/celldetection.git
# Git config:
git config --global user.name "Mengqing Hu"
git config --global user.email "mengqinghu688@gmail.com"
```

------

## Storage

```bash
# Home directory (code, venvs, configs) — 600G quota
du -sh ~

# Bigdata (large datasets, model outputs)
ls /bigdata/

# Cache redirection (add to ~/.bashrc if needed)
export PIP_CACHE_DIR=/bigdata/<group>/hu09/.cache/pip
export HF_HOME=/bigdata/<group>/hu09/.cache/huggingface
export TORCH_HOME=/bigdata/<group>/hu09/.cache/torch
```



{
  "argv": [
    "bash",
    "-lc",
    "module load python/3.12.4 cuda/12.8 && exec /data/home2/hu09/project/celldetection/.venv/bin/python -Xfrozen_modules=off -m ipykernel_launcher -f {connection_file}"
  ],
  "display_name": "cpn kernel",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "kernel_protocol_version": "5.5"
}
