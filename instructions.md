# Specific training 
```
sbatch <slurm_sscript> -s <bash_script_runs_training>  -c <container> -v <volumes>
```
example: 
```
sbatch slurm/don_volume_aoa_batch -s scripts/run_train_don.sh  -c /p/app/containers/tensorflow/tensorflow-22.03-tf2-py3.sif -v /p/home/jyuko,/p/work1/projects
```


# General HPC 

## slurm

### Commands
- submitting a job: 
    ```
    sbatch <your_batch_script>
    ```

- checking the queue for your jobs: 
    ```
    squeue -u <your user ID>
    ``` 
- you can get an interactive node like this:
    ```
    srun --pty  -A <your account> -q <your qos>  --x11 /bin/bash -i
    ```
     - for the ARL project that looks like this:
        ```
        srun --pty  -A ARLAP44862YFR -q frontier  --x11 /bin/bash -i
        ```
     - then, ssh into the waiting node:

### Notes
- note that the interactive node will not work if you try to get a GPU (it will either throw an error or hang, depending on the method used).

## singularity

### Commands
- singularity shell: 
    ```
    singularity shell -B <directories>,<to>,<mount> --nv </p/your/container/location.sif>
    ```
### Notes
- the `--nv` flag is for nvidia and will throw a warning if the machine you're on doesn't support nvidia.

## Examples
- Running a singularity container on you local host
    ``` 
    singularity shell -B /p/home/your_id ,/p/work1/projects --nv /p/app/containers/tensorflow/tensorflow-22.03-tf2-py3.sif
    ```
- You can run a python script (non-interactive) when calling singularity:
    ```
    singularity exec -B /p/home/your_id,/p/work1/projects  --nv /p/app/containers/tensorflow/tensorflow-22.03-tf2-py3.sif python3 your_script_name.py 
    ```
- You can quickly check that it does the bare minimum (python with numpy):
    ```
    Singularity> python -c "import numpy as np; print(f'random number: {np.random.randint(100)}');"
    ```
- If you have a local package you can install it (but there's no way to reach back to pypi):
    ```
    pip install -e .
    ```
    
# Working through things with Bryan - Notes
- My Questions:
   - How to correctly split up batches per node/GPU and ensure it fits?
   - Ways to integrate all of my scripts/options more seemlessly?
   - Am I doing this right re: python/singularity/slurm
   - Anything special I need to do to get this to run with qsub or other managers.
   - improvements for performance/efficiency.
