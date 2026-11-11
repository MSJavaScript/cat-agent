# Mock-compute test workflow

This reference repeats the expected artifact flow for the fake pipeline.

| Step | Command | Required inputs | Outputs |
|---|---|---|---|
| prepare | `mock-prepare --outdir .` | none | `input_A.json`, `input_B.json` |
| relax_A | `mock-relax --input input_A.json --output relaxed_A.json` | `input_A.json` | `relaxed_A.json`, `relax_A.log` |
| relax_B | `mock-relax --input input_B.json --output relaxed_B.json` | `input_B.json` | `relaxed_B.json`, `relax_B.log` |
| analyze | `mock-analyze --input-dir . --output analysis.json` | both relaxed files | `analysis.json`, `analysis.md` |
| report | `mock-report --analysis analysis.json --output final_report.md` | `analysis.json` | `final_report.md` |

The two relax steps are independent. The analysis step must wait for both of
them; the report step must wait for the analysis.
