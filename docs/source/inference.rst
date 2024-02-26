Inference Guide
===============

The ``cpn_inference.py`` script is designed for advanced image processing using Contour Proposal Networks (CPN) for Instance Segmentation. This script is versatile, supporting a range of input types, models, and various processing options to fit different use cases in image analysis.

Usage
-----

Basic Command Line Execution:

.. code-block:: bash

   python cpn_inference.py [arguments]

Using Docker:

.. code-block:: bash

   docker run -v $(pwd):/data ericup/celldetection:latest cpn_inference.py [arguments]

Using Apptainer and Slurm:

.. code-block:: bash

   srun [srun-arguments] apptainer exec --nv celldetection_latest.sif python cpn_inference.py [arguments]

Overview of Inputs and Outputs
------------------------------

The `cpn_inference.py` script is tailored for advanced image processing using Contour Proposal Networks (CPN) for Instance Segmentation. This section provides an overview of the script's inputs and outputs, including the handling of region properties.

Inputs:
```````

1. **Images**: The script processes individual image files, collections of images (using `glob patterns <https://code.visualstudio.com/docs/editor/glob-patterns>`_), or image URLs.

   - Supported Formats: `.tif`, `.jpg`, `.png`, etc.
   - Example: `-i 'images/*.tif'` or `-i 'http://example.com/image.tif'`

2. **Models**: Requires pre-trained models for image segmentation, specified as local files, URLs, or hosted model names.

   - Format: Filename, `glob pattern <https://code.visualstudio.com/docs/editor/glob-patterns>`_, URL, or hosted model name.
   - Example: `-m 'models/model1.pt'` or `-m 'cd://hosted_model_name'`

3. **Additional Options**: Includes masks, point masks, and various processing parameters like tile size, stride, precision, etc.

Outputs:
````````

1. **Segmentation**: The primary output includes contours, scores and classes processed based on the input and models used. These are written to an HDF5 file.

2. **Label Images**: The flags `--labels` and `--flat_labels` can be used to produce label images (with and without overlap-channels, respectively).

3. **Optional Outputs**: Depending on flags like `--overlay`, and `--demo_figure`, the script can output overlays for inspection, and demo figures.

4. **Output Directory**: Users specify the directory for output files.

   - Default: `outputs`
   - Example: `-o 'processed_images'`

5. **Region Properties (CSV Output)**: When specified using the `--properties` flag, the script calculates and outputs region properties as CSV files. These properties are derived from the `skimage.measure.regionprops` and `regionprops_table` functions, providing quantitative measurements of detected regions/objects in the images.

   - Some Common Properties: `label`, `area`, `bbox`, `centroid`, `eccentricity`, `perimeter`, etc.
   - The full list of available properties can be found in the documentation of `skimage.measure.regionprops` (`Skimage Regionprops Documentation <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>`_).

Arguments Explanation
----------------------

- ``-i`` / ``--inputs``:
  Specify input files, patterns (glob), or URLs.
  Default: None (mandatory argument).
  Example: ``-i 'images/*.jpg'``, ``-i 'http://example.com/image.tif'``

- ``-o`` / ``--outputs``:
  Set the output path for processed files.
  Default: 'outputs'.
  Example: ``-o 'processed_images'``

- ``--inputs_method``:
  Method for loading non-hdf5 inputs.
  Default: 'imageio'.
  Example: ``--inputs_method 'pillow'``

- ``--inputs_dataset``:
  Name of the dataset for hdf5 inputs.
  Default: 'image'.
  Example: ``--inputs_dataset 'mydataset'``

- ``-m`` / ``--models``:
  Specify model files, patterns (glob), URLs, or hosted model names.
  Default: None (mandatory argument).
  Example: ``-m 'models/model1.h5'``, ``-m 'cd://hosted_model_name'``

- ``--masks``:
  Specify mask files to define regions of interest.
  Default: None.
  Example: ``--masks 'masks/mask1.tif'``

- ``--point_masks``:
  Define point masks for specifying exact object locations. The model will predict contours at positive locations in the mask.
  Default: None.
  Example: ``--point_masks 'point_masks/mask1.tif'``

- ``--point_mask_exclusive``:
  If set, only objects in point masks are segmented.
  Default: False (not exclusive).
  Example: ``--point_mask_exclusive``

- ``--masks_dataset``, ``--point_masks_dataset``:
  Dataset names for hdf5 mask inputs.
  Default: 'mask' and 'point_mask' respectively.
  Example: ``--masks_dataset 'maskset'``, ``--point_masks_dataset 'pointmaskset'``

- ``--devices``, ``--accelerator``, ``--strategy``:
  Configuration for computation devices and strategies.
  Default: 'auto' for all three.
  Example: ``--devices 'cuda'``, ``--accelerator 'gpu'``, ``--strategy 'ddp'``

- ``--precision``:
  Set computation precision (64, 32, 16, etc.).
  Default: '32-true'.
  Example: ``--precision '32-true'``

- ``--num_workers``, ``--prefetch_factor``, ``--pin_memory``:
  Data loading configurations.
  Default: 0 for ``--num_workers``, 2 for ``--prefetch_factor``, None for ``--pin_memory``.
  Example: ``--num_workers 4``, ``--prefetch_factor 3``, ``--pin_memory``

- ``--batch_size``:
  Number of samples processed per batch.
  Default: 1.
  Example: ``--batch_size 2``

- ``--tile_size``, ``--stride``:
  Settings for processing images in tiles.
  Default: 1024 for ``--tile_size``, 768 for ``--stride``.
  Example: ``--tile_size 1024``, ``--stride 512``

- ``--border_removal``, ``--stitching_rule``, ``--min_vote``:
  Advanced object detection and stitching configurations.
  Default: 4 for ``--border_removal``, 'nms' for ``--stitching_rule``, 1 for ``--min_vote``.
  Example: ``--border_removal 5``, ``--stitching_rule 'max'``, ``--min_vote 2``

- ``--labels``, ``--flat_labels``:
  Options for generating label images with and without channels, respectively.
  Default: False for both.
  Example: ``--labels``, ``--flat_labels``

- ``--demo_figure``, ``--overlay``:
  Generate demonstration figures or overlays.
  Default: False for both.
  Example: ``--demo_figure``, ``--overlay``

- ``--truncated_images``:
  Enable support for partially loaded images.
  Default: False.
  Example: ``--truncated_images``

- ``-p`` / ``--properties``:
  Specify region properties for analysis.
  Default: None.
  Example: ``-p 'area' 'perimeter'``

- ``--spacing``, ``--separator``:
  Configuration for region properties in spatial analyses.
  Default: 1.0 for ``--spacing``, '-' for ``--separator``.
  Example: ``--spacing 0.5``, ``--separator '_'``

- ``--gamma``, ``--contrast``, ``--brightness``:
  Adjust image properties.
  Default: 1.0 for ``--gamma`` and ``--contrast``, 0.0 for ``--brightness``.
  Example: ``--gamma 1.5``, ``--contrast 1.2``, ``--brightness 0.1``

- ``--percentile``:
  Apply percentile normalization.
  Default: None.
  Example: ``--percentile 0.1 0.9``

- ``--model_parameters``:
  Pass additional model parameters.
  Default: ''.
  Example: ``--model_parameters 'threshold=0.5,overlap=0.3'``

Examples
--------

Basic Usage:

.. code-block:: bash

   python cpn_inference.py -i 'images/*.tif' -m 'cd://model_name'

Adjusting Image Properties and Using Masks:

.. code-block:: bash

   python cpn_inference.py -i 'images/*.tif' --masks 'masks/*.tif' -m 'model_file' --gamma 1.2 --contrast 1.1

Processing Multiple Inputs with Different Models:

.. code-block:: bash

   python cpn_inference.py -i image1.tif image2.tif -m model1 model2 --batch_size 2 --tile_size 1024 2048 --stride 512

These examples demonstrate various ways to use the `cpn_inference.py` script. Users can customize the script's execution based on their specific requirements and computational resources.


Using Docker for Running the Script
-----------------------------------

The `cpn_inference.py` script can also be executed within a Docker container. Below is an updated example Docker command, along with a detailed explanation.

.. code-block:: bash

    docker run --rm \
      -v $PWD/docker/inputs/:/inputs/ \
      -v $PWD/docker/outputs:/outputs/ \
      --gpus="device=0" \
      celldetection:latest /bin/bash -c \
      "python cpn_inference.py -i '/images/*.tif' -o '/outputs' -m 'cd://model_name' --tile_size=1024 --stride=768"

Explanation of the Docker Command:

1. **``docker run``**: This is the basic command to run a Docker container.

    - ``--rm``: This flag automatically removes the container once the process exits. It helps in managing resources by cleaning up the temporary container created for this specific task.

2. **Volume Mounts ``-v``**: These options link directories on your local machine to directories in the Docker container. This allows the script within the container to access and output data to your system.

   - ``$PWD/docker/inputs/:/inputs/``: Maps a local directory (`docker/inputs`) to the container's `/inputs/` directory, where the input images should be stored.
   - ``$PWD/docker/outputs:/outputs/``: Maps a local directory (`docker/outputs`) to the container's `/outputs/` directory. The script will write its output files here.

3. **GPU Allocation ``--gpus="device=0"``**: Assigns the first GPU on your machine to the Docker container. This is important for GPU-accelerated processing.

4. **Docker Image**:

   - ``celldetection:latest``: Specifies the Docker image to use, in this case, the latest version of `celldetection`.

5. **Executing the Script**:

   - The command ``/bin/bash -c "python cpn_inference.py -i '/images/*.tif' -o '/outputs' -m 'cd://model_name' --tile_size=1024 --stride=768"`` is executed inside the Docker container. It runs the `cpn_inference.py` script with the specified arguments:
     - ``-i '/images/*.tif'``: Specifies the input images located in the `/images/` directory inside the container.
     - ``-o '/outputs'``: Sets the output directory inside the container to `/outputs/`.
     - ``-m 'cd://model_name'``: Specifies the model for image processing.
     - ``--tile_size=1024`` and ``--stride=768``: These arguments set the tile size and stride, optimizing the script for efficient image processing and memory management.

This Docker command is a template that can be adapted for different use cases. Users can modify the mounted directories, adjust GPU settings, or change the script's arguments to suit their specific requirements.
Note: Always ensure that the paths provided in Docker commands correctly map to your local filesystem for input/output operations.

Using the Script with Apptainer and Slurm in HPC Environments
-------------------------------------------------------------

For users in High-Performance Computing (HPC) environments, the `cpn_inference.py` script can be executed using Apptainer (formerly Singularity) along with the Slurm workload manager. Below is an example command, followed by a detailed explanation:

.. code-block:: bash

    srun --mpi=pspmix --cpu_bind=v --accel-bind=gn --cpus-per-task=64 apptainer exec --nv /path/to/celldetection_latest.sif \
      python /path/to/cpn_inference.py -i 'images/*.tif' -o 'outputs' -m 'cd://model_name' --tile_size=1024 --stride=768

Explanation of the Command:

1. **``srun``**: This is the Slurm command for running jobs. It is configured with various options for CPU, GPU, and memory usage, which may need to be adapted depending on the target system's configuration.

2. **Slurm Options**: These depend entirely on your system. Make sure to adjust them accordingly.

   - ``--mpi=pspmix``: Specifies the MPI configuration.
   - ``--cpu_bind=v``: CPU binding type. Adjust as necessary for your system.
   - ``--accel-bind=gn``: GPU binding settings.
   - ``--cpus-per-task=64``: Allocates 64 CPUs per task. This value should be adjusted based on the available resources and the requirements of the task.

3. **Apptainer Execution**:

   - ``apptainer exec --nv``: Executes a command within an Apptainer container. The ``--nv`` flag enables NVIDIA GPU support, which is crucial for GPU-accelerated processing.
   - ``/path/to/celldetection_latest.sif``: Path to the Apptainer image file (`.sif`). This file contains the environment needed to run the script, including all dependencies. Check our Installation Guide if you want to learn how to create this file.

4. **Running the Python Script**:

   - ``python /path/to/cpn_inference.py``: Executes the `cpn_inference.py` script.
   - **Arguments**:

     - ``-i 'images/*.tif'``: Specifies the input images to be processed.
     - ``-o 'outputs'``: Designates the output directory for the results.
     - ``-m 'cd://model_name'``: Indicates the model to be used for image processing.
     - ``--tile_size=1024`` and ``--stride=768``: Sets the tile size and stride for efficient image processing, particularly important in HPC contexts where managing memory and computational resources is crucial.

It's important to note that the `srun` command may require adaptation to fit the specific configuration and policies of your HPC environment. Users should consult their system administrators or HPC documentation to determine the appropriate settings for their particular system.

Note on Apptainer vs Docker Directory Handling
``````````````````````````````````````````````

Apptainer and Docker handle system directories differently, which is crucial to understand when transitioning from Docker to Apptainer:

- **Apptainer**: It automatically binds several system directories from the host to the container (depending on the host's configuration). This binding facilitates access to the host system's resources and environment, which is particularly useful in HPC settings where access to shared file systems and resources is required.

- **Docker**: In contrast, Docker isolates the container from the host's system directories by default. To access host directories, explicit volume mounts (`-v` option) are required. This isolation is a core feature of Docker, providing a consistent and controlled environment.

Understanding these differences is essential for effectively managing file paths, resource access, and permissions when using containers in different environments.

Adapting to Specific Compute Resources
--------------------------------------

When working with `cpn_inference.py`, it is essential to optimize the script's settings based on the available compute resources, such as CPU/GPU memory and processing power. Two critical parameters to consider are `--tile_size` and `--stride`. Adjusting these can significantly impact the script's memory usage and execution speed.

- ``--tile_size``:
  This parameter defines the size of the tiles or windows the script uses to process large images. A smaller tile size consumes less memory, which is beneficial for systems with limited resources. However, smaller tiles might increase processing time and affect the segmentation quality for larger objects.

  Default: 1024
  Example for Limited Memory: ``--tile_size 512``

- ``--stride``:
  The stride determines the overlap between consecutive tiles during the sliding window processing. A smaller stride increases overlap, which can lead to better stitching of segmented objects across tiles but at the cost of increased computation. A larger stride reduces computation but might miss some objects or parts thereof at the borders between tiles.

  A common strategy is to set the stride to a value slightly less than the tile size. For instance, if the tile size is 1024, setting the stride to `tile_size - 256` (i.e., 768) provides a balance between computational efficiency and overlap for stitching.

  Default: 768
  Example for Balance in Large Images: ``--stride 768`` (when ``--tile_size`` is set to 1024)

Tips for Memory-Constrained Environments
----------------------------------------

1. **Reduce Tile Size**: Lower the `--tile_size` to reduce memory usage per processing step. Be mindful of the object sizes in your images; too small tiles might not capture large objects effectively.

2. **Adjust Stride**: Decrease the `--stride` to increase overlap, which can help in stitching but be cautious of increased computation time.

3. **Batch Processing**: Use a smaller `--batch_size` if memory constraints are an issue. This reduces the number of samples processed simultaneously.

4. **Optimize Model Precision**: Lower the `--precision` setting (e.g., from '32-true' to '16-mixed') to reduce the memory footprint of the model, though this might affect the accuracy.

5. **Resource Allocation**: On multi-GPU systems, distributing the workload (`--strategy` and `--devices` settings) can help manage memory usage more effectively.

By carefully tuning these parameters, users can adapt the `cpn_inference.py` script to fit their system's capabilities, ensuring efficient and effective image processing even under memory constraints.
