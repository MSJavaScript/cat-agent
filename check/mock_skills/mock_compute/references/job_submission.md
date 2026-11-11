# Job submission for the mock-compute skill

Use the local jobmanager provider `local-mock` and the job environment
`mock-compute` for every `mock-relax` step.

The step workdir must contain the input file, `run.sh`, and
`job_description.json`. A minimal `run.sh` is:

```bash
mock-relax --input input_A.json --output relaxed_A.json
```

The job manager adds the environment initialization commands from the
`mock-compute` job environment. Do not add `#SBATCH` directives or machine
selection logic; those belong to the job manager.
