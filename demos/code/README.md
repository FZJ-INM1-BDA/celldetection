Overview of files in this folder
--------------------------------

This directory contains example scripts and a demo for using a custom dataset with the Contour Proposal Network (CPN) in the `celldetection` project.

Files
-----

- README.md
	- This file: short overview and usage notes for the `demos/code` folder.

- create_metadata.py
	- Generates dataset split files (`training.txt`, `validation.txt`, `test.txt`) based on PNG images found in `../mydata/img`.
	- Default split ratios: 70% train / 15% val / 15% test. Run with:

		python create_metadata.py

- my_dataset.py
	- A small custom dataset implementation that reads PNG images from `img/` and `mask/` directories under a dataset root.
	- Provides `MyDataset` (base) and convenience subclasses `MyDatasetTrain`, `MyDatasetVal`, and `MyDatasetTest` which read file lists from `metadata/training.txt`, `metadata/validation.txt`, and `metadata/test.txt` respectively.
	- Loads images and masks, generates instance labels using `skimage.measure.label`, and exposes items as `(filename, image, mask, label)`.

- mydata.ipynb
	- The best starting point if you want to understand the workflow step by step.
	- It contains the full notebook-based walkthrough for custom cell detection with Contour Proposal Networks, including data preparation, preprocessing, model configuration, visualization, and training/evaluation logic.
	- The notebook is organized into interactive sections that typically cover:
		- importing dependencies and enabling the runtime environment,
		- defining and printing `celldetection.Config`,
		- loading the custom dataset from `../mydata`,
		- generating and previewing Albumentations transforms,
		- visualizing labels and contour proposals,
		- creating the CPN model and moving it to the selected device,
		- preparing train/validation/test loaders and example plots.
	- `mydata.py` was generated from this notebook with `jupyter nbconvert --to script mydata.ipynb`.

- mydata.py
	- `mydata.py` was generated from this notebook with `jupyter nbconvert --to script mydata.ipynb`.

- train_mydata.sh
	- SLURM batch script for training `mydata.py` on Capella with one GPU.
	- Requests 1 node, 1 GPU, 12 CPU cores, 96G memory, and 14 hours.
	- Loads the Python/CUDA modules, activates the project virtual environment, changes into this folder, and runs:

		python mydata.py

	- Writes SLURM logs to files named like `slurm-cpn-mydata-<jobid>.out` and `slurm-cpn-mydata-<jobid>.err`.
	- Prints the job id, node name, Python path, CUDA availability, GPU name, start/end time, and total duration.

- savemodel/
	- Directory containing saved model checkpoints (e.g., `model_epoch_10.pth`, `model_best.pth`). Use these to resume training or evaluate pretrained results.


Quick start
-----------

1. Prepare your dataset under `../mydata` with the following layout:

	- mydata/
		- img/        <- PNG images
		- mask/       <- PNG masks (same filenames as images)

2. Generate the split metadata files with `create_metadata.py`:

	 python create_metadata.py

   This creates `training.txt`, `validation.txt`, and `test.txt` under `../mydata/metadata`.

3. Use `my_dataset.py` if you want to load the dataset programmatically or adapt the data-loading logic for your own project.

4. Inspect or adjust configuration in `mydata.ipynb` or `mydata.py` (e.g., `conf.directory`, `cpn`, `batch_size`, `device`).

5. Run the demo script or open the notebook for interactive runs:
```
	 python mydata.py
```

Run with sbatch on Capella
--------------------------

Use `sbatch` for longer training jobs that should run on a compute node instead of the login node. You do not need to start an interactive `srun` session before submitting the job.

1. Log in to Capella:

```
ssh ******@login1.capella.hpc.tu-dresden.de
```

2. Enter the code workspace:

```
cd /data/cat/ws/******-cpn_workspace_e1/celldetection/demos/code
```

3. Confirm that the batch script is in this folder:

```
ls *.sh
```

You should see `train_mydata.sh`.

4. If you edited `mydata.ipynb`, convert it to `mydata.py` before submitting:

```
jupyter nbconvert --to script mydata.ipynb
```

5. Submit the fixed training job directly from the login node:

```
sbatch train_mydata.sh
```

6. Check the job status:

```
squeue -u ****** 
```

7. Watch the SLURM log after submission. Replace `<jobid>` with the id printed by `sbatch`:

```
tail -f slurm-cpn-mydata-<jobid>.out
```

The script uses the same resource request as this interactive command:

```
srun --partition=capella --nodes=1 --gres=gpu:1 --cpus-per-task=12 --mem=96G --time=14:00:00 --pty bash -l
```

Training outputs are written by `mydata.py` under:

```
output/mydata_output/
```

If automatic figure and log saving is enabled in the notebook/script, additional files are saved under:

```
output/mydata_output/all_figures/
output/mydata_output/run.log
```

The SLURM job releases its GPU, CPU, and memory allocation automatically when `python mydata.py` finishes or fails.




PS: convert mydata.ipynb to mydata.py with:

```
jupyter nbconvert --to script mydata.ipynb
jupyter nbconvert --to script mydata_uncertainty.ipynb
jupyter nbconvert --to python mydata_uncertainty.ipynb --output mydata_uncertainty.py
jupyter nbconvert --to script 'Cell Detection with Contour Proposal Networks.ipynb'
```





