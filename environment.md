
#### Use srun to enter compute node:

```
srun --partition=capella --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash -l

```





#### Check workspace on Capella:

```
ws_list -t
ws_list -s
```






#### Module management on Capella:

```
module spider Python/3.12.3

module load release/25.06  GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0

module list
module --force purge
module unload Python/3.12.3
module avail Python/3.12.3
module show Python/3.12.3

```



#### Create virtual environment and install dependencies on Capella:

```
python -m venv .venv
source .venv/bin/activate
pip install ipykernel
which python
deactivate
pip install -r requirements.txt
pip freeze > requirements.txt

```


#### Jupyter kernel management on Capella:

```
pip install ipykernel
pip install --upgrade pip
python -m ipykernel install --user --name cpn-kernel --display-name="cpn kernel"



jupyter kernelspec list
jupyter kernelspec uninstall cpn-kernel

```


```

{
  "argv": [
    "bash",
    "-lc",
    "module load release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0 && exec /data/cat/ws/mehu311f-cpn_workspace/celldetection/.venv/bin/python -Xfrozen_modules=off -m ipykernel_launcher -f {connection_file}"
  ],
  "display_name": "cpn kernel",
  "language": "python",
  "metadata": {
    "debugger": true
  },
  "kernel_protocol_version": "5.5"
}

```

