Installation Guide
==================

This guide provides detailed instructions for installing the `celldetection` package. Depending on your environment and requirements, you can choose from several installation methods.

Docker and Apptainer Installation
---------------------------------

For users who prefer using Docker or Apptainer, installation of PyTorch or setting up virtual environments is not required, as the Docker image comes with all necessary dependencies.

**Docker Installation**

Pull the latest `celldetection` Docker image:

.. code-block:: bash

    docker pull ericup/celldetection:latest

**Verification for Docker**

After pulling the image, you can run a Docker container to verify the installation:

.. code-block:: bash

    docker run --rm --gpus="device=0" ericup/celldetection:latest python -c "import celldetection; print(celldetection.__version__)"

You may remove ``--gpus="device=0"`` if you do not have GPUs on your system.

**Apptainer Installation**

In HPC environments where Apptainer is preferred:

.. code-block:: bash

    apptainer pull --dir . --disable-cache docker://ericup/celldetection:latest

If your system allows caching, you may remove ``--disable-cache``.
On some systems you may need to `specify a custom cache directory <https://apptainer.org/docs/user/latest/build_env.html>`_ with sufficient disk space.

**Verification for Apptainer**

To verify the installation in Apptainer, run the following command using the downloaded `.sif` file:

.. code-block:: bash

    apptainer exec --nv celldetection_latest.sif python -c "import celldetection; print(celldetection.__version__)"

You may remove ``--nv`` if you do not have GPUs on your system.

Installation in Python Environment
----------------------------------

For users who wish to install `celldetection` directly in their Python environment, follow the steps below. Ensure that you have PyTorch installed as it is a critical dependency for the package. Visit the `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_ for instructions.

**Virtual Environment Setup**

It's recommended to install `celldetection` in a virtual environment.

Using venv:

1. Create a virtual environment:

   .. code-block:: bash

       python -m venv celldetection_env

2. Activate the virtual environment:

   On Windows:

   .. code-block:: bash

       celldetection_env\Scripts\activate

   On macOS and Linux:

   .. code-block:: bash

       source celldetection_env/bin/activate

Using Conda:

1. Create a Conda environment:

   .. code-block:: bash

       conda create -n celldetection_env python=3.x

   Replace `3.x` with the specific Python version you want to use.

2. Activate the Conda environment:

   .. code-block:: bash

       conda activate celldetection_env

**PyPI Installation**

Install the latest stable release from PyPI:

.. code-block:: bash

    pip install -U celldetection

**GitHub Installation**

For the latest development version from GitHub:

.. code-block:: bash

    pip install git+https://github.com/FZJ-INM1-BDA/celldetection.git

Post-Installation
------------------

After installation, you can start using the `celldetection` package for your image processing tasks. If installed in a Python environment, remember to keep your virtual environment active. To exit the virtual environment, use the `deactivate` command.
