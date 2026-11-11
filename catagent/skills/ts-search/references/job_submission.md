# Template for job submission (only for StepNode)

## run.sh

For vasp calculation, use "np" as a placeholder to represent the number of MPI process, subagent will replace it by a correct value. For example, if there are 3 dirs containing NEB, you should generate a `run.sh` for each dir:
```bash
# run.sh
mpirun -n <np> vasp_std > vasp.stdout
```
or generate a single `run.sh` for three dirs:
```bash
# run.sh
for d in neb1 neb2 neb3
do
  cd ${d}
  mpirun -n <np> vasp_std > vasp.stdout
  cd ..
done
```

For ASE, jobs are usually submitted in batches. You should write a python script which run a **single** job, then loop through all the inputs in `run.sh`.  For example, if the job runs a series of NEB using MLFF,
```bash
# Suppose there are files named in format like
# CH3-CH2+H_Pt_step0.extxyz
# Each of them contains an initial NEB path

# run_neb.py is a python script which runs neb with MLFF
# sys.argv[1] is the path to checkpoint file.
# sys.argv[2] is the name of extxyz file containing initial NEB path
# sys.argv[3] is the name prefix of result *.traj and *.log files
for f in *.extxyz
do
    name=${f%.extxyz} 
    python3 run_neb.py frozen_model.pt2 ${f} ${name}
done
```
## job_description.json
You should generate a `job_description.json` for sub-agent to provider basic information of the job. For example:

```json
{
    "description":"10 DFT NEB jobs for NH3 on Fe surface.",
    "job_name": "NH3_Fe",
    "use_gpu": false,
    "batch_job": true,
    "n_jobs": 10,
    "workload": "high",
    "placeholders": {
        "np": "The number of MPI process"
    }
}
```
Because images are relaxed simultaneously, `workload` of VASP NEB are usually "medium" or higher. But for ASE with MLFF, `workload` can be "low" or "mini", because MLFF is very fast on GPU, `use_gpu` should be `true` in this case.
