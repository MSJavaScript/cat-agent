
# Template for job submission (only for StepNode)

## run.sh
The pre-trained model is usually located on remote machine, set the path `<model_path>` and name `<model_name>` of `*.pt` file by placeholders, subagent will replace them with corrected values.

```bash
model=<model_path>/<model_name>
# or
# model="./pre-train-model.pt"
# if you have it in job local dir.
dp --pt train input.json --finetune ${model} > train_log 2>&1
dp --pt freeze -c model.ckpt.pt -o frozen 
dp --pt test -m frozen.pt2 -s structs.hdf5 -d pred_val > infer_val_log 2>&1

# You can do something using the resulting `frozen.pt2`
# e.g. python3 run_neb.py frozen.pt2 initial_path.extxyz
# suppose run_neb.py is a script performing NEB.
```

## job_description.json
You should generate a `job_description.json` for sub-agent to provider basic information of the job.

```json
{
    "description":"Describe the job.",
    "job_name": "set name here",
    "use_gpu": true,
    "batch_job": true,
    "n_jobs": 200,
    "workload": "low",
    "placeholders": {
        "model_name": "Name of pre-trained model.",
        "model_path": "Path to the model."
    }
}
```
Some notes:
- description: Should include versions of deepmd-kit and checkpoint file. 
- use_gpu: Always true. 
- workload: Average workload of each job. For MLFF finetuning, training and MD, this can be "medium" or "high". For NEB and structure relaxation, "low" or "mini" is more reasonable, because MLFF is very fast.