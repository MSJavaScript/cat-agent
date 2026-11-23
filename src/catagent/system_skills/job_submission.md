# Job submission skill(Only for StepNode)

The details of job submission such as resource allocation and grouping of batch jobs are hidden from the current agent (StepNode). You only need to do 3 tasks: (1) Ensure the integrity of input files within each computation directory. (2) Select appropriate computational platform provider and job environment. (3) Provide a template script `run.sh` for job submission and a json file containing the metadata of the job.

## 1. The integrity of input files

You should guarantee the integrity of inputs in each job dir. For example, if there are 100 single point VASP DFT jobs in dirs from `sp_00` to `sp_99`, then `INCAR`, `POSCAR`, `KPOINTS` and `POTCAR` should exist in all of them. If a input file is the same for all the jobs, you can create soft links pointing to the target file in each subdirectory.

Tips: To avoid a large amount of output, use bash flexibly, for example
```bash
bad=0
for((i=0;i<=99;++i))
do
  a=`printf "%02d" ${i}`
  for f in INCAR POSCAR KPOINTS POTCAR
  do
    if [ ! -e sp_${a}/${f} ]
    then
      bad=1
      echo === sp_${a} lack files ===
      break
    fi
  done
done
if [ ${bad} -eq 0 ]
then 
  echo Inputs are complete in all subdirs.
fi
```


## 2. Get the informations of providers

Tool `get_providers_info` returns the list of metadata of computational platform providers, each item has four fields:
- name: Name of the provider.
- processing_units: Can be "cpu" or "gpu". If you want to finetune MLFF or run MD, structure relaxation and NEB with MLFF, select the provider with gpu.
- job_envs: Job environments and the softwares they include.
- remote_files: Descriptions of remote large files, such as pre-trained models, dataset and cache files. 

You should select a specific provider and job environment for a job. To keep the consistency of computation, same setting should be used for the same type of calculation. For example, if `"vasp-5.4.4"` in `slurm-1` are selected for VASP jobs, then subsequent VASP calculation should use the same. 

## 3. run.sh
You should always generate a `run.sh` as submission script. Only focus on the execution logic, no need to consider settings such as MPI process and job environment initialization, as they will be handled by jobmanager, use placeholders to represent them and explain them in `job_description.json`.  Below are some examples:

### Example 1: Many NEB jobs

Suppose there are a series of NEB jobs in the following dirs
```
CH4-CH3+H  CH3-CH2+H  CO2-CO+O  CO-C+O ...
```
The `run.sh` should be something like
```bash
# No SBATCH derivatives here, no source intel setvars.sh here
# jobmanager will handle all of them 
for d in *
do
  if [ -d ${d} ]
  then
    cd ${d}
    # use <np> to represent the number of MPI processes
    mpirun -n <np> vasp_std > vasp.stdout
    cd ..
  fi
done
```

### Example 2: Finetune machine learning force field (MLFF)
Suppose there are a series of `train_data_*.db` files, for example:
```
train_data_80.db train_data_100.db  train_data_150.db ...
```
We want to finetune the MACE force field using these datasets one by one, `run.sh` should be like:
```bash
# The path to the pre-trained model, represented by placeholders
model=<mace_mode_path>/<mace_mode_name>
# The place holder of absolute path of run_train.py
mace_train=<path_to_run_train.py>
# Loop through all the db files
for train_db in train_data_*.db
do
  ndata=${train_db%.db}
  # Size of train set extracted from filename
  ndata=${ndata##*_} 
  batch_size=`python3 -c "print(int(${ndata}/60))"`
  mkdir log_${ndata} model_${ndata} 
  python3 ${mace_train} --name='ft_mace' \
  --seed=${RANDOM} --foundation_model=${model} \
  --train_file=train_data_${ndata}.db \
  --batch_size=${batch_size} # ... other arguments are omitted 
done
``` 

## 4. job_description.json

You should always generate a `job_description.json` which contains the metadata of a job. This will be read by JobManager and help it to complete the `run.sh`. The format is:
```json
{
  "description":"A batch of vasp jobs to do structure relaxation.",
  "job_name": "NH3-Fe-relax",
  "use_gpu": false,
  "batch_job": true,
  "n_jobs": 500,
  "workload": "medium",
  "placeholders": {
    "np": "The number of mpi processes",
    "model_name": "Name of pre-trained model.",
    "model_path": "Path to the model."
  }
}
```
- description: Short description of the job. 
- job_name: The job name. For a batch job, this is taken as a prefix. 
- use_gpu: If the job requires GPU.
- batch_job: If this is a batch job.
- n_jobs: The total number of jobs. Equals to 1 if `batch_job` is `false`.
- workload: Estimation of the average workload for each job. Possible values are "mini", "low", "medium", "high", "ultra high". 
- placeholders: A dict containing all the placeholders in `run.sh` that must be replaced. Key is the placeholder, value is its meaning.

### The reference for estimating workload
1. The size of the system: Workload depends on the size of the system, such as the number of atoms and mesh density of the finite element calculation. Take VASP job as an example:
    |Number of atoms| Workload|
    |---|---|
    |<16| mini|
    |16 ~ 32| low |
    |32 ~ 64 | medium |
    |64 ~ 200 | high |
    | > 200 |ultra high|

2. Type of calculation: Workload depends on the type of calculation. Here are some examples:
- Workload of NEB is always "medium" or higher because many images are simultaneously relaxed.
- DFT+U and hybrid DFT functional are more expensive than GGA. Workload should be "medium" or higher.
- Workloads of MLFF jobs depends on the model scale, the number of atoms in the training data and the number of epochs. Here are some rough criteria: Workloads of MLFF training and finetuning are usually "medium". MLFF is fast on GPU, if structure relaxation or NEB is run with MLFF, workload should be "low" or "mini". Workload of MD still depeneds on the number of atoms even though MLFF is fast.
  |Number of atoms in MLFF MD | Workload|
    |---|---|
    |<1k| mini|
    |1k ~ 5k| low |
    |5k ~ 1w | medium |
    |1w ~ 2w | high |
    | > 2w |ultra high|

## If you are not sure if a job environment is OK

All the details of job submission are blind to the current agent. If you need to submit a batch of jobs but are not sure if the provider or job environment is OK, just try to submit one or two jobs to see if the job runs successfully.

This also applies to the adjustment of calculation parameters. For example, there is a batch of TiO2 r2SCAN single point DFT. You can just submit a single job and check the result, adjust INCAR until this calculation converges. Then you can safely submit a batch of jobs.  
 
## Conception: Single job vs Batch job
A single job has ONLY one calculation workdir, while a batch job consists of multiple calculation workdirs, each of them or subdirs inside them is a complete job. For both cases, there should be ONLY one `run.sh` and `job_description.json`, that is, the batch jobs share the common `run.sh` and `job_description.json`. 

For example, there are 100 VASP jobs, then you should put only one `run.sh` and `job_description.json` in the workdir of current step. In run.sh, you write as [Example 1: Many NEB jobs](#example-1-many-neb-jobs). And in `job_description.json`, set `batch_job = true`. After you submit it, JobManager will divide them into groups automatically.

If the jobs in a step are indeed very complex, for example, the `run.sh`s in dirA and dirB are very different, or dirA contains vasp jobs, while dirB contains MLFF jobs, then the jobs in these two dirs should be taken as independent jobs. **Just remember one rule, when you call `submit_job`, there should be ONLY one `run.sh` and `job_description.json` in `local_dir`.**