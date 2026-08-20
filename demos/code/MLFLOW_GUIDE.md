# CellDetection Uncertainty Experiments with MLflow

This guide explains how to change the experimental parameters in `mydata_uncertainty.py`, submit training jobs through SLURM, and use MLflow to record, inspect, compare, and share experimental results.

## 1. Relevant files

- `mydata_uncertainty.py`: model configuration, training, validation, plotting, and MLflow logging.
- `train_mydata_uncertainty.sh`: SLURM batch script.
- `../../mlflow.db`: MLflow database containing experiments, parameters, and metrics.
- `mlruns/`: MLflow artifact storage directory.
- `output/mydata_uncertainty_output/<run_id>/`: local output for each experiment.

## 2. Changing experimental parameters

The model and training parameters are defined in the `conf` configuration inside `mydata_uncertainty.py`:

```python
conf = cd.Config(
    uncertainty_head=True,
    uncertainty_nms=True,
    uncertainty_factor=7.0,
    cpn="CpnResNet50UNet",
    score_thresh=0.9,
    nms_thresh=0.5,
    order=6,
    samples=64,
    epochs=20,
    steps_per_epoch=36,
    batch_size=8,
    optimizer={
        "Adam": {
            "lr": 0.0001,
            "betas": (0.9, 0.999),
            "weight_decay": 1e-3,
        }
    },
)
```

Changing these parameters does not require editing the `.sh` file. Every new submission creates a separate MLflow Run and does not overwrite earlier Runs.

The `.sh` file only needs to be changed when:

- adjusting GPU, CPU, or memory resources;
- changing the maximum runtime;
- selecting another SLURM partition;
- changing the Python or CUDA modules;
- running another Python script;
- changing the default MLflow Experiment name.

## 3. Installing and checking MLflow

Load the cluster environment and install MLflow before using it for the first time:

```bash
cd /data/cat/ws/mehu311f-cpn_workspace_e1/celldetection

module purge
module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
source .venv/bin/activate

pip install mlflow
python -c "import mlflow; print(mlflow.__version__)"
```

## 4. Submitting a SLURM experiment

```bash
cd /data/cat/ws/mehu311f-cpn_workspace_e1/celldetection/demos/code
sbatch train_mydata_uncertainty.sh
```

Check the job status:

```bash
squeue --me

squeue -u "$USER"
```

Follow the training log:

```bash
tail -f slurm-cpn-uncertainty-<job_id>.out
```

The SLURM script uses the Job ID to create a readable Run name:

```text
uncertainty-slurm-<job_id>
```

The Python script also obtains the unique MLflow Run ID.

## 5. Separate output for every experiment

Each experiment creates a directory based on its MLflow Run ID:

```text
output/mydata_uncertainty_output/<run_id>/
├── all_figures/
├── config.json
├── run.log
├── loss_curves.png
├── loss_curves_together.png
├── iou_metrics_and_counts.png
└── model_final.pth
```

Consequently, a new experiment cannot overwrite an older experiment or upload stale images from a previous Run.

`model_final.pth` is a PyTorch checkpoint containing:

- the model `state_dict`;
- the experiment configuration;
- the best inference thresholds;
- the final mean F1 score.

## 6. Information recorded by MLflow

### Parameters

- CPN model name;
- epochs, batch size, and dataset sizes;
- optimizer and scheduler;
- learning rate and weight decay;
- Fourier order and sample count;
- uncertainty settings;
- score and NMS thresholds.

### Metrics

- `train_loss` for every epoch;
- `val_loss` for every epoch;
- learning rate for every epoch;
- best validation F1;
- best score threshold;
- best NMS threshold;
- F1 at different IoU thresholds;
- final mean F1;
- FPS and mean inference time.

### Tags

- model name;
- experiment type;
- compute node;
- PyTorch version;
- CellDetection version;
- operating-system information.

### Artifacts

- loss curves;
- IoU and detection metric plots;
- uncertainty prediction images;
- predictions generated during training;
- configuration file;
- execution log;
- final model checkpoint.

A successfully completed script produces a `FINISHED` Run. An uncaught exception marks the Run as `FAILED`.

## 7. Starting the MLflow UI

Load the environment and start MLflow on the allocated compute node. The following example uses node `c29`:

```bash
source /data/cat/ws/mehu311f-cpn_workspace_e1/celldetection/.venv/bin/activate

mlflow ui \
  --backend-store-uri sqlite:////data/cat/ws/mehu311f-cpn_workspace_e1/celldetection/mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "localhost:*,127.0.0.1:*,c29:*" \
  --cors-allowed-origins "http://localhost:*,http://127.0.0.1:*"



mlflow ui \
  --backend-store-uri sqlite:////data/cat/ws/mehu311f-cpn_workspace_e1/celldetection/mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --allowed-hosts "localhost:*,127.0.0.1:*,c147:*" \
  --cors-allowed-origins "http://localhost:*,http://127.0.0.1:*"
```

Keep this terminal running.

Create an SSH tunnel from your local computer:

```bash
ssh -N \
  -o ProxyJump=none \
  -L 5000:c29:5000 \
  mehu311f@login1.capella.hpc.tu-dresden.de

ssh -N \
  -o ProxyJump=none \
  -L 5000:c147:5000 \
  mehu311f@login1.capella.hpc.tu-dresden.de
```

Open the following address in a local browser:

```text
http://localhost:5000
```

Select **Model training**, rather than **GenAI**, in the upper-left corner of the MLflow interface.

## 8. Viewing images and results

Navigate to:

```text
Model training
→ Experiments
→ celldetection-mydata-uncertainty
→ Select a Run
→ Artifacts
→ outputs
```

Important files include:

- `loss_curves.png`: separate training and validation loss plots;
- `loss_curves_together.png`: combined loss plot;
- `iou_metrics_and_counts.png`: IoU metrics and TP/FP/FN counts;
- `all_figures/figure_*.png`: data examples, predictions, and uncertainty plots;
- `config.json`: complete configuration for the Run;
- `run.log`: training log;
- `model_final.pth`: model checkpoint.

## 9. Comparing experiments before and after parameter changes

1. Open **Model training**.
2. Select `celldetection-mydata-uncertainty`.
3. Select the Runs to compare.
4. Click **Compare**.
5. Compare their Parameters, Metrics, Charts, and Artifacts.

It is advisable to change only a small number of parameters in each experiment. This makes it easier to determine which change caused a difference in performance.

## 10. Sharing results with someone

If they have a Capella account, they can create their own SSH tunnel:

```bash
ssh -N \
  -o ProxyJump=none \
  -L 5000:c29:5000 \
  <username>@login1.capella.hpc.tu-dresden.de
```

They can then open:

```text
http://localhost:5000
```

Also provide the compute-node name, port, Experiment name, and relevant Run name.

The current MLflow service runs on a temporary compute node. The UI becomes unavailable when the allocation ends or the node is released, but the database and artifacts remain stored. Restarting the MLflow UI makes them available again.

If they do not have a Capella account, export the plots, configuration, and logs through an institutionally approved sharing service. Never share SSH keys or login credentials.

## 11. Limitations

The current setup uses SQLite:

```text
/data/cat/ws/mehu311f-cpn_workspace_e1/celldetection/mlflow.db
```

SQLite is suitable for personal experimentation and a small number of sequential jobs. It is not recommended for several SLURM jobs that write frequently to the same database at the same time. For long-term multi-user operation, deploy an MLflow Tracking Server backed by PostgreSQL and shared artifact storage.
