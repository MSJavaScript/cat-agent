# Template for job submission (Only for StepNode)

## run.sh
You should provide a template of submission script `run.sh`. You don't need to set the exact number of MPI process and the initialization of vasp environment. The sub-agent will handle them.
```bash
mpirun -n <np> vasp_std > vasp.stdout
```

## job_description.json
You should generate a `job_description.json` for sub-agent to provider basic information of the job.

```json
{
    "description":"A vasp job for xxx",
    "job_name": "set name here",
    "use_gpu": false,
    "batch_job": ...,
    "n_jobs": ...,
    "workload": "...",
    "placeholders": {
        "np": "The number of mpi processes"
    }
}
```
You need to estimate the workload based on the INCAR settings and the calculating system. For example, the number of atoms, the number of KPOINTS, the value of ENCUT, if the calculation is spin-polarized, DFT+U or hybrid functional, and so on.