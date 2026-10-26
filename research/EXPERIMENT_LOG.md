# AI Experiment Log

Last updated: 2026-10-26

Active scope: Antwerp `E76 -> D3` single-ego exit task

Active branch at update: `codex/e76-d3-gamma099-800k`

Controller-fix commit: `f16be9f`

## How an AI should use this file

This is the compact handoff and decision log for active experiments. Read it
before proposing another E76 -> D3 run. The README remains the implementation
contract; ignored JSON and JSONL files are the detailed evidence. Do not treat
an idea in this file as an approved experiment.

Evidence labels:

- **Fact**: directly measured, tested, or read from a saved artifact.
- **Inference**: best explanation supported by the facts, but not isolated as
  a causal variable.
- **Hypothesis**: a proposed explanation or next investigation.
- **Decision**: an explicit constraint on future work.

When adding an experiment, record its purpose, single intended variable,
complete configuration, artifact paths and hashes, result, limitations, and
decision. Keep failed experiments: append or amend them instead of rewriting
history to show only successful runs.

## Current state in one page

- **Fact:** The four gamma-0.99 baseline checkpoints at 200k, 400k, 600k and
  800k each achieved `0/25` strict D3 arrivals. The 100 cases are balanced at
  five evaluation cases for each E76 starting lane at each checkpoint.
- **Fact:** The controller had one real numerical-boundary bug. A fully
  contained 1.8 m vehicle in a 3.2 m lane can settle near a 0.7 m lateral
  offset, but the former exact comparison rejected values such as 0.7005 m.
- **Fact:** Environment version 3 allows 0.001 m beyond the analytic clearance
  only when the mapped target lane matches and lateral speed is at most
  0.05 m/s. A 0.7011 m offset, continued lateral motion, or a wrong target lane
  still fails completion.
- **Fact:** One `changeLane(..., 0)` call continues to produce lateral motion in
  SUMO 1.18.0 in the tested scene. Durations 0 and 4 had the same measured
  motion and settling steps. Production control remains at duration 4.
- **Fact:** The pending target stayed locked across all nine observed active
  J105 crossings. Four crossings reached the mapped target and were affected by
  the numerical boundary; five were genuinely incomplete target mismatches.
- **Fact:** On five directed cases using the frozen 800k checkpoint, the 1 mm
  fix changed completed/expired lane changes from `15/7` to `18/4`, but strict
  D3 arrival remained `0/5`.
- **Fact:** The diagnostic route-following rule arrived in `4/5` primary cases.
  Its lane-3 case collided. A second lane-3 seed missed the exit after 628
  safety interventions, so rule-policy success is not robust.
- **Inference:** The remaining SAC failure is primarily policy behavior, not
  the 1 mm controller boundary. At 800k over 25 cases, the actor requested
  movement away from D3 13,063 times versus 1,211 toward D3. On the five
  post-fix cases the ratio was 2,622 versus 155.
- **Decision:** Preserve all four version-2 checkpoints and their logs. Do not
  spend more large-scale evaluation budget on these old models.
- **Decision:** No training is running. Do not start training or a large
  evaluation without a new explicit instruction and a configuration/risk
  review.
- **Decision:** Do not reopen terminal in-place lateral movement as a fix. Do
  not silently change gamma, potential shaping, reward weights, or the formal
  observation contract.

## Experiment lineage

### Historical context: earlier E76 -> D3 models

The earlier 1M `e76-d3-easy-1m-nolc-20261024` run used the raw lane-index
potential definition. EXP-005, `E76D3-GAMMA0999-S42-200K`, isolated gamma
0.999 for 200k steps and historically recorded 18/25 arrivals, but 11 of those
arrivals reached the E77 lane end before completing the required lane change.
Both belong to earlier controller/reward contracts and must not be used as
direct controls for environment version 3.

**Conclusion:** these runs are useful historical evidence, not valid estimates
of current-controller performance. Their checkpoints and logs remain retained.

### EXP-E76-099-800K: corrected-control gamma-0.99 baseline

**Status:** training complete; checkpoints frozen as environment version 2.

**Purpose:** determine whether a fresh gamma-0.99 SAC policy learns the strict
E76 -> D3 exit under the corrected tracked-lane-change controller.

**Run name:** `E76D3-GAMMA099-S42-800K-V2`.

Configuration:

| Field | Value |
|---|---|
| Task / traffic | `exit_e76_d3` / Easy |
| Route | E76 -> D3 |
| Training seed | 42 |
| Training steps | 800,000 fresh steps |
| Checkpoint interval | 200,000 steps |
| Algorithm | SAC with `paper_raw` GRU encoder |
| Gamma | 0.99 |
| Delay | uniform 0-20 control steps |
| Training safety | off |
| Lane-change controller | `tracked_lane_change_v2`, 4 s command duration |
| Potential | original raw lane-index E76 -> D3 potential |
| Lane-command penalty | disabled for this task |
| Evaluation | deterministic, safety off, 25 fixed cases/checkpoint |
| Evaluation coverage | five cases for each E76 starting lane 0-4 |
| Runtime used for diagnosis | SUMO 1.18.0 |

Frozen checkpoint identity and outcome:

| Step | SHA-256 | Strict D3 result | Completed / expired changes | Boundary-like expiries |
|---:|---|---:|---:|---:|
| 200,000 | `0a1b9b14d56031ed31bd7e5f316b52a9021ecb4e4b63b36bd89526e8b753cc34` | 0/25 | 93 / 45 | 21 |
| 400,000 | `381886e0e94a6a3543fb7b63a6fd2a704b4f47ade55929aa0a3a3da8ba1af984` | 0/25 | 77 / 44 | 17 |
| 600,000 | `0d330b91ddaca04921f2ab1a3a7ab47de28db8dc6e72fcbd15117e314b6195a5` | 0/25 | 107 / 48 | 22 |
| 800,000 | `0b2570860713607da611756c078640426e47e102aa379e87700ac5e22aa4417c` | 0/25 | 73 / 44 | 17 |

The 77 boundary-like expiries are diagnostic events, not 77 failed episodes;
one episode may contain multiple lane-change attempts.

**Conclusion:** more training steps within this run did not produce any strict
success at the four saved checkpoints. The actor strongly favored the wrong
lateral direction, including from lane 0, where all five 800k cases accepted a
move away from the direct D3 path.

**Limitations:** this is one training seed and one traffic level. Repeated
checkpoint evaluation estimates episode behavior, not independent-training
variance. The checkpoints predate the version-3 completion tolerance.

**Decision:** retain the checkpoint bytes and logs, but stop evaluating these
four models at scale. Version checks intentionally prevent treating them as
version-3 checkpoints.

### DIAG-CTRL-001: low-cost controller diagnosis

**Status:** complete; no training.

**Purpose:** separate controller defects from learned-policy defects before
changing rewards, observations, or training settings.

#### Completion boundary

The analytic lateral clearance is `(3.2 - 1.8) / 2 = 0.7 m`. Stable SUMO
values included approximately 0.7000-0.7005 m with zero lateral velocity.
Environment version 3 therefore uses a 0.001 m numerical tolerance.

**Fact:** contract tests accept 0.7005 m and reject 0.7011 m. They also reject
0.7005 m when lateral speed is 0.051 m/s and prevent a wrong target lane from
accumulating stable completion steps.

**Conclusion:** this was a real but narrow controller bookkeeping bug. The fix
does not relax route topology, safety, target matching, or terminal behavior.

#### Single `changeLane` command and pending lock

On the same seed and scene, duration 0 and duration 4 both produced:

| Measurement | Duration 0 | Duration 4 |
|---|---:|---:|
| First lateral-motion step | 1 | 1 |
| Last lateral-motion step | 34 | 34 |
| First target-lane-index step | 21 | 21 |
| First settled step | 35 | 35 |

**Conclusion:** in this SUMO 1.18.0 setup, a single duration-0 request did not
cancel motion after one simulation step. The pending record is only an internal
lock; it does not repeatedly issue TraCI movement commands. No evidence was
found that the pending target changed or cleared unexpectedly at J105.

#### Frozen 800k before/after check

Seeds `200017`, `200004`, `200003`, `200005`, and `200001` cover starting
lanes 0, 1, 2, 3, and 4 respectively.

| Metric | Before 1 mm fix | After 1 mm fix |
|---|---:|---:|
| Completed lane changes | 15 | 18 |
| Expired lane changes | 7 | 4 |
| Strict D3 arrivals | 0/5 | 0/5 |

**Conclusion:** the fix corrects completion accounting, but it is insufficient
to fix the learned route choice.

### DIAG-RULE-001: route-following diagnostic rule

**Status:** complete; diagnostic only; no training.

**Purpose:** test whether the same low-level controller and normal safety
constraints can physically execute the D3 route when the high-level direction
is correct.

**Rule:** from E76, request only legal adjacent changes toward a lane that can
reach D3. Use the SAC controller's target lock, room checks, timing, and safety
interventions; do not allow terminal in-place lateral movement.

Primary results:

| Start lane | Seed | Outcome |
|---:|---:|---|
| 0 | 200017 | Arrived |
| 1 | 200004 | Arrived |
| 2 | 200003 | Arrived |
| 3 | 200005 | Collision |
| 4 | 200001 | Arrived |

An additional lane-3 case, seed 200008, ended `Missed Exit` in `E77_2` after
628 safety interventions.

**Conclusion:** the route and controller are demonstrably feasible for some
traffic realizations and all five starting-lane geometries are represented, but
the simple rule is not a robust solution. Lane 3 is the clearest stress case.

## Validation fixed with environment version 3

- Local: `python3 -m unittest tests.test_antwerp_contract` -> 50/50 passed.
- Remote targeted controller suite under SUMO 1.18.0 -> 21/21 passed.
- `git diff --check` passed before commit `f16be9f`.
- The version-3 code changes only completion tolerance and associated contract
  documentation/tests; it does not change gamma, potential, reward weights, or
  formal observations.

## Known problems and open questions

1. **Wrong-direction SAC actions.** The dominant failure is not a lack of
   lane-change commands, but a large excess of commands away from D3.
2. **Lane-3 traffic interaction.** Even a route-aware rule can collide or be
   blocked by safety interventions. Feasibility in a few seeds is not a robust
   success estimate.
3. **Topology knowledge versus learning.** The current formal observation may
   not make the route consequence of a lane index sufficiently learnable. This
   is a hypothesis, not permission to change the observation.
4. **Reward/action semantics.** The action sign, conditional action statistics
   by starting lane, and critic preference for toward/away actions need an
   offline audit before attributing failure to reward shaping.
5. **Single-seed evidence.** The baseline cannot establish training stability
   across initializations. That limitation does not justify spending more
   evaluation budget on its four already-failed checkpoints.

## Suggested next work, not yet approved

These are ordered from cheapest to most expensive.

1. Perform an offline, no-SUMO audit of the saved 800k replay/log data:
   condition actor outputs and accepted actions on start lane, edge, lane index,
   distance to J105, and whether the action points toward D3. Check action-sign
   consistency from policy output through controller execution.
2. Use the frozen actor and critic only for a small counterfactual diagnostic:
   compare actor action and critic values for matched toward/away actions at
   the same recorded states. This tests whether the bias lives primarily in the
   actor, critic, or downstream execution. Do not call it causal without a
   controlled intervention.
3. Improve the diagnostic rule's lane-3 decision timing using existing safety
   inputs and constraints, then test only a small declared seed set. Keep it a
   diagnostic baseline, not a replacement training policy.
4. Before any new version-3 training, write a one-page configuration and risk
   proposal specifying the hypothesis, one intended variable, budget, stop
   conditions, checkpoint schedule, and evaluation seeds. Do not reuse the old
   25 cases for checkpoint selection and final reporting simultaneously.

## Artifact map

Detailed artifacts are intentionally ignored by Git and must not be moved into
tracked source directories.

| Evidence | Path |
|---|---|
| Four-checkpoint comparison | `tmp/ai/e76-d3/gamma099-baseline-800k-download/gamma099_baseline_800k/eval25_paired/comparison.json` |
| Completion-boundary audit | `tmp/ai/e76-d3/gamma099-baseline-800k-download/gamma099_baseline_800k/eval25_paired/completion_audit.json` |
| Downloaded training log | `tmp/ai/e76-d3/gamma099-baseline-800k-download/gamma099_baseline_800k/E76D3-GAMMA099-S42-800K-V2.jsonl` |
| Controller/rule diagnostic | `tmp/ai/e76-d3/controller-diagnostic-v1/report.json` |
| Before-fix and extra lane-3 cases | `tmp/ai/e76-d3/controller-diagnostic-v1/supplemental.json` |
| Diagnostic script | `tmp/ai/e76-d3/controller_diagnostic.py` |

Checkpoint and runtime-log bytes remain outside Git under their existing
experiment storage. Never delete or overwrite them as part of updating this
document.
