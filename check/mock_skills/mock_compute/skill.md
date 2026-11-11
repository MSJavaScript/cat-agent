---
name: mock-compute
description: A fast fake computation pipeline used to test the cat-agent workflow, router, StepNode, and jobmanager without running real scientific software.
metadata:
  tags:
    - test
    - mock
    - fake-computation
---

# Mock Compute Skill

This skill describes a **fake** computation workflow. It is intentionally not
real science. Every "software" is a short Python script with `sleep()` calls, so
the whole pipeline finishes in a few seconds. Use it to test the agent's
planning, routing, parallel step execution, job submission, and result
reporting.

The available fake software commands are:

| Command | Purpose |
|---|---|
| `mock-prepare --outdir DIR` | Create `input_A.json` and `input_B.json`. |
| `mock-relax --input FILE --output FILE [--seconds N]` | Run one fake relaxation and write `relaxed_<sample>.json` plus a `.log` file. |
| `mock-analyze --input-dir DIR --output FILE` | Compare all `relaxed_*.json` files and write `analysis.json` and `analysis.md`. |
| `mock-report --analysis FILE --output FILE` | Turn `analysis.json` into a small markdown report. |

The commands are on `PATH` when the `mock-compute` job environment is active.
For a local non-job step, they are also available if the test harness has added
`check/mock_software/bin` to `PATH`.

## Recommended workflow

A useful plan has 4-5 steps with a parallel branch:

1. **prepare**: run
   `mock-prepare --outdir <step_workdir>`.
   Outputs: `input_A.json`, `input_B.json`.

2. **relax_A**: run a fake relaxation for sample A:
   `mock-relax --input input_A.json --output relaxed_A.json`.
   This must be submitted through the jobmanager as a job
   (`run.sh` + `job_description.json`).
   Outputs: `relaxed_A.json`, `relax_A.log`.

3. **relax_B**: same as relax_A but for sample B:
   `mock-relax --input input_B.json --output relaxed_B.json`.
   This is independent of relax_A and can run in parallel with it.
   Outputs: `relaxed_B.json`, `relax_B.log`.

4. **analyze**: after both relax steps are done, run
   `mock-analyze --input-dir <dir containing both relaxed files> --output analysis.json`.
   Outputs: `analysis.json`, `analysis.md`.

5. **report**: run
   `mock-report --analysis analysis.json --output final_report.md`.
   Output: `final_report.md`.

The dependences are therefore:

```text
prepare ──┬──> relax_A ──┐
          └──> relax_B ──┴──> analyze ──> report
```

## Job submission

For the relaxation steps, use provider `local-mock` and job environment
`mock-compute`. The jobmanager will run `run.sh` locally. A minimal `run.sh` is:

```bash
mock-relax --input input_A.json --output relaxed_A.json
```

Put it together with the input file in the step workdir and provide a
`job_description.json` such as:

```json
{
  "description": "Fake mock-relax job for sample A.",
  "job_name": "mock-relax-A",
  "use_gpu": false,
  "batch_job": false,
  "n_jobs": 1,
  "workload": "mini",
  "placeholders": {}
}
```

Do not put resource settings, `#SBATCH`, or job-environment activation commands
in `run.sh`; the jobmanager handles those.

## Notes

- The fake energies are deterministic. Sample A normally ends with a lower
  energy than sample B.
- A step is not finished until its `report.md` contains a summary section.
- The `mock-report` step is optional, but including it makes the final result
  easy to inspect.
