# TABLE III — Antwerp GRU, 1M training steps

Status: COMPLETE; 9000/9000 episodes.
Updated (UTC): 2026-10-15T20:15:10.590852+00:00

| Difficulty | Method | Episodes | Success (%) | Collision (%) | No Merge / timeout (%) | Avg Return | Avg Ego Vel (m/s) | Avg Jerk (m/s^3) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | DAROM-GRU w/o SC | 1500/1500 | 18.93 ± 1.21 | 73.60 ± 0.53 | 0.33 ± 0.31 | 145.10 ± 7.17 | 10.359 ± 0.063 | 0.0021 ± 0.0013 |
| Easy | DAROM-GRU + SC | 1500/1500 | 16.00 ± 3.27 | 40.73 ± 0.42 | 0.07 ± 0.12 | 64.84 ± 11.85 | 7.956 ± 0.191 | 0.0025 ± 0.0013 |
| Medium | DAROM-GRU w/o SC | 1500/1500 | 15.60 ± 2.25 | 74.47 ± 1.81 | 0.33 ± 0.23 | 70.76 ± 6.59 | 9.767 ± 0.041 | 0.0029 ± 0.0009 |
| Medium | DAROM-GRU + SC | 1500/1500 | 7.20 ± 1.22 | 20.87 ± 1.94 | 0.47 ± 0.42 | -41.47 ± 5.50 | 6.577 ± 0.117 | 0.0016 ± 0.0007 |
| Hard | DAROM-GRU w/o SC | 1500/1500 | 42.20 ± 1.40 | 42.33 ± 2.32 | 8.13 ± 1.50 | -306.74 ± 19.74 | 9.639 ± 0.093 | 0.0023 ± 0.0013 |
| Hard | DAROM-GRU + SC | 1500/1500 | 38.93 ± 3.63 | 38.00 ± 2.08 | 15.73 ± 0.99 | -340.30 ± 17.26 | 8.416 ± 0.025 | 0.0023 ± 0.0013 |

- Each row: one checkpoint, three batches of 500 episodes; training seed 579071.
- Deterministic actions, uniform delay up to 20 steps; existing baseline safety controller.
- Final ± values are sample standard deviations across three evaluation batches, not training-seed variation or a confidence interval.
- Partial rows show pooled means only; no final spread estimate until all three batches are complete.
- Success excludes any ego teleport. No Merge follows the legacy timeout proxy; true unmerged and merged timeouts are separate in JSON/CSV.
- Motion uses equal episode weight across all outcomes; signed jerk may cancel. Absolute/RMS jerk, missing counts and success Wilson 95% intervals are retained in JSON.
- This covers the GRU safety-on/off rows only, not the paper's other algorithms or its 7M training budget.
- Evaluation episode seeds: 710001–710500, 720001–720500, 730001–730500; paired across safety settings.
