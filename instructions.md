# General HPC 

## slurm

### Commands
- submitting a job: 

    ```
    sbatch <your_batch_scipt>
    ```

- checking the queue for your jobs: 

    ```
    squeue -u <your user ID>
    ``` 

### Notes
Insert notes here

## singularity

### Commands
- singularity shell: 
        
    ```
    singularity shell -B <directories>,<to>,<mount> --nv </p/your.container/location.sif>
    ```
### Notes
- the `--nv` flag is for nvidia and will throw a warning if the machine you're on doesn't support nvidia.

## Examples
- Running a singularity container on you local host

    ``` 
    singularity shell -B /p/home/your_id ,/p/work1/projects --nv /p/app/containers/tensorflow/tensorflow-22.03-tf2-py3.sif
    ```
- You can quickly check that it does the bare minimum (python with numpy):
    ```
    Singularity> python -c "import numpy as np; print(f'random number: {np.random.randint(100)}');"
    ```
- You can run a python script (non-interactive) when calling singularity:
    ```
    singularity exec -B /p/home/your_id,/p/work1/projects  --nv /p/app/containers/tensorflow/tensorflow-22.03-tf2-py3.sif python3 your_script_name.py 
    ```