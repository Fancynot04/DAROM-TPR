# AI Experiment Log

Last updated: 2026-10-26

Active scope: Antwerp `E76 -> D3` single-ego exit task

Active branch at update: `codex/e76-d3-gamma099-800k`

Controller-fix commit: `f16be9f`

## EXP-E76-CONTROL134-200K: running

**Question:** how much of the NAV136 result is attributable to adding live
D3 distance and required-lane navigation, rather than only to fresh training
under the v3 controller? Train a fresh 134-dimensional SAC control with the
same seed and 200k budget; do not reuse any old checkpoint or replay.

**Paired configuration:** Easy E76→D3, SUMO 1.18.0/environment v3 with 1 mm
containment tolerance, seed 42, γ=0.99, unchanged potential/reward/safety and
lane-change logic, `paper_raw` GRU/SAC, uniform 0–20-step delay, training
safety off. The sole intended input difference from NAV136 is no two-value
navigation block: observation `[vehicle 93, action history 40, delay 1]`.
This also removes two columns from the GRU fusion layer, so parameter count
is not exactly matched. Schedule: 200,000 environment steps; model and replay
at 100k and 200k. Use a distinct run name and output directory; record local
JSONL and offline W&B metrics (Forge was unreachable for NAV136). Run the same
five fixed development seeds at 100k and same balanced 25 seeds at 200k,
Easy/uniform/no safety/deterministic. Compare strict arrival, collision,
missed exit and E76/E77 actor request direction by starting lane. The matched
seed reduces one source of variation but one training seed is still not a
robust causal estimate.

**Risk and launch gate:** about 95 minutes of GPU training at NAV136's
measured 35.16 steps/s, plus saving and evaluation; isolate the remote
checkout so no old outputs are touched. Run 134-shape/profile/compatibility
tests, verify SUMO 1.18.0, GPU, disk and exact code revision. Stop on
nonfinite observations, contract drift or failed checkpoint save.

**Launch and current status (2026-10-26):** committed code revision
`9546c3acf475c4f3961734eb8f0d4347192e63dc` in separate remote checkout
`/home/cyl/project/DAROM-TPR-control134-200k`; run name
`E76D3-CONTROL134-GAMMA099-S42-200K`. The remote 92-test suite passed
(12 skipped), SUMO reported 1.18.0, and an actual reset yielded 134 finite
inputs. GPU and disk preflight passed. Background training PID `2705923` used
`--wandb_mode offline`, with offline W&B ID `jdxnmtp4`; no Forge sync or model
upload. Its ignored JSONL is
`tmp/ai/e76-d3/control_134_200k/E76D3-CONTROL134-GAMMA099-S42-200K.jsonl`.
At the 100k checkpoint spot check the process was alive, scalar losses were
present, and training continued past `env_steps=108,000`. A 15-minute heartbeat
will inspect the final checkpoint and run the remaining fixed-seed evaluation.

**100k development diagnostic:** the model and replay checkpoints were saved.
Their SHA-256 values are respectively
`a734dd34ba067f6c757f69d2d723af614879a49b70a378e05a45bc58a5349a55`
and `afa85c1aa44163c7c9374a5be4395f49e77fd507520a2eb8e69467c3c1de9fb1`.
The five fixed seeds `200017, 200004, 200003, 200005, 200001` covered start
lanes 0–4 and yielded **3/5 strict arrivals, 2/5 missed exits, 0 collisions**.
Lanes 0, 1, 4 arrived; lanes 2, 3 missed. Raw E76/E77 actor directions were
1,994 toward D3 and 185 away; these are requests, not completed changes.
The ignored report is `tmp/ai/e76-d3/control_134_200k/eval_100k_5.json`
(SHA-256 `6a57997da368434db5af2202aa1fada117d5983cc9829d55254136daadf4a1cc`).
NAV136 also scored 3/5 on its 100k development set, so this small diagnostic
does not show a success-count separation. It is not the final 25-case result.

## EXP-E76-NAV136-200K: completed

**Purpose:** test whether explicit live D3 distance and required road-network
lane changes reduce wrong-direction requests and produce strict arrivals under
the existing 1 mm containment controller. This is a fresh random-initialized
policy; the four old checkpoints are historical reference only, not a matched
causal control.

**Configuration:** Easy E76→D3, seed 42, `paper_raw` GRU/SAC, 136 inputs
`[vehicle 93, navigation 2, actions 40, delay 1]`, uniform 0–20-step neighbor
delay, training safety off, γ=0.99, unchanged potential/reward/controller,
200,000 environment steps. Navigation is current-time
`[clip(d_D3/1260.74,0,1), n_required/5]`; the pure-network count follows
E76/E77 lane indices and the actual J105 target lane. A nonterminal invalid
driving distance is a hard error. Model and replay save at 100k and 200k.

**Planned monitoring:** one online `DAROM-OUT` Forge run, `env_steps`
for all `train/*` curves; 100-step losses and entropy metrics, per-episode
outcomes and start lane, latest-20-episode strict success/missed/collision
rates, and 1k-step toward/away request counts and nonempty-window away ratio.
Local JSONL remains under ignored `tmp/ai/e76-d3/nav_200k/`. Verify run ID,
first loss curves, continued metric updates at both checkpoints, and final
step parity. Network lag is reported without restarting training. At 100k,
diagnose seeds `200017, 200004, 200003, 200005, 200001` (historically lanes
0–4); at 200k, report the existing 25 balanced seeds
`200001–200012, 200014, 200016–200020, 200036–200037, 200045, 200047,
200050, 200063–200064`. Verify actual start lanes in each new report. Report
fixed cases separately from training curves. Do not upload model artifacts or
sync duplicate TensorBoard curves.

**Executed configuration and deviation:** code commit
`b87098077a4853415db53a6a8d2798b0d91474a2`, remote checkout
`/home/cyl/project/DAROM-TPR-nav136-200k`, run
`E76D3-NAV136-GAMMA099-S42-200K`. Remote SUMO was 1.18.0, CUDA device
was an RTX 4090, and the remote 91-test suite passed (12 skipped). A real
SUMO reset returned a finite 136-value observation. Forge API verification
timed out despite the remote proxy responding to a basic HTTP request. With
the user's subsequent approval to proceed without Forge, training used
`--wandb_mode offline`: **no online Forge run was created**, no model or
TensorBoard files were uploaded. The offline W&B run ID was `64eavsbr`;
JSONL and TensorBoard remain local to the remote checkout. The prior four
checkpoints and their logs were not changed.

**Training fact:** 200,000 environment steps and 199,899 SAC updates were
recorded in the JSONL; elapsed training time was 5,687.97 s (35.16
steps/s). There were 229 completed training episodes: 31 strict arrivals,
187 missed exits and 11 collisions. These training outcomes are *not* the
fixed-seed evaluation. Model and replay checkpoints exist at both 100k and
200k, plus final files. SHA-256 of the 100k model checkpoint:
`3b9e8c07dcc7017ca340d6ac411f66da207c7366b64e5fb0c040fd57df571380`;
200k model checkpoint:
`208297efdf20e9196ed0fbbaa726cf64f4a538fd8d9b335d30cc83500317be08`;
final model:
`29cc4da724bf508dee59559f338d75f4831c721fe5cfe8358ae7f538dc19c5e7`;
final replay:
`3a4add54d65d0200d1fdd25b0b84d471b04fbfe5336f294b0d4b66c42a5b00da`.

**Fixed-seed evaluations, Easy/uniform/no safety/deterministic:** at 100k,
the five development seeds covered lanes 0–4 and gave 3/5 strict arrivals,
2/5 missed exits and 0 collisions. At 200k, the 25 historical balanced
seeds again gave five cases per lane: **16/25 strict arrivals, 9/25 missed
exits, 0/25 collisions**. Per-lane results and raw actor request directions:

| E76 start lane | Strict arrivals | Missed exits | Collisions | Toward D3 | Away from D3 |
|---:|---:|---:|---:|---:|---:|
| 0 | 3/5 | 2/5 | 0/5 | 3,763 | 169 |
| 1 | 5/5 | 0/5 | 0/5 | 3,667 | 149 |
| 2 | 3/5 | 2/5 | 0/5 | 3,796 | 142 |
| 3 | 2/5 | 3/5 | 0/5 | 3,334 | 233 |
| 4 | 3/5 | 2/5 | 0/5 | 3,690 | 137 |
| **Total** | **16/25** | **9/25** | **0/25** | **18,250** | **830** |

The away-request share among nonzero E76/E77 actor directions was 830/19,080
= 4.35%; the policy is no longer *predominantly* requesting away from D3,
although wrong-direction requests remain, especially in lane 3. Requests
include commands ignored by pending-maneuver or room checks; they are not
accepted lane changes. The 100k five-case result was a development diagnostic,
not a model-selection step.

**Additional 100k checkpoint evaluation on the same 25 development cases
(2026-10-26):** frozen 100k model SHA-256
`3b9e8c07dcc7017ca340d6ac411f66da207c7366b64e5fb0c040fd57df571380`;
same Easy/uniform/no-safety/deterministic protocol and exact balanced seeds as
the 200k report. The 100k model reached **10/25** strictly, missed **15/25**,
and collided in **0/25**. The 200k model reached **16/25**, a descriptive gain
of 6/25 (24 percentage points); 10 individual failures became arrivals, but
4 arrivals regressed to misses. The comparison by starting lane is:

| E76 start lane | 100k strict | 200k strict | 100k toward / away requests | 200k toward / away requests |
|---:|---:|---:|---:|---:|
| 0 | 3/5 | 3/5 | 952 / 231 | 3,763 / 169 |
| 1 | 2/5 | 5/5 | 926 / 401 | 3,667 / 149 |
| 2 | 3/5 | 3/5 | 1,386 / 377 | 3,796 / 142 |
| 3 | 1/5 | 2/5 | 1,550 / 397 | 3,334 / 233 |
| 4 | 1/5 | 3/5 | 1,257 / 216 | 3,690 / 137 |
| **Total** | **10/25** | **16/25** | **6,071 / 1,622** | **18,250 / 830** |

The away share among nonzero requests dropped from 1,622/7,693 = 21.08% to
830/19,080 = 4.35%. Request totals also changed substantially, so counts
alone do not identify a physical lane-change mechanism. This reused 25-case
set measures development trajectory, not independent generalization or proof
that added navigation inputs caused improvement. The ignored 100k report is
`tmp/ai/e76-d3/nav_200k/eval_100k_25.json` (SHA-256
`dc461be762e487cb010026969d2730b8a073862c4563ddee1493ea2519493e69`).
The previous five-case 100k diagnostic remains valid but is superseded for
100k→200k comparison by this paired 25-case report. No continuation training
was launched as part of this evaluation.

**Interpretation:** the new checkpoint reaches D3 in these cases and requests
away far less often than the historically measured 800k actor (13,063 away
versus 1,211 toward over its 25 cases). This is
descriptive only: the old checkpoints differ in training history and budget,
so they do not isolate the causal effect of the two navigation inputs. Lane 3
remains the weakest in this small evaluation. A matched fresh 134-dimensional
control is now running (above). Its result will improve the comparison, but
one training seed still cannot establish a robust causal effect.

**Frozen 200k failure diagnostic (five cases):** the deterministic evaluation
was replayed step-by-step without training or modifying the controller. All
five final outcomes and toward/away request counts matched the saved 25-case
report. The three E76_3 missed-exit seeds were `200008`, `200009`, `200010`;
additional missed-exit seeds `200002` (lane 4) and `200036` (lane 0) were
checked for pattern coverage.

| Seed | Observed decisive sequence | Classification |
|---:|---|---|
| 200008 | Reached E76_0 with about 515 m left and E77_0 with 172 m left; accepted E77_0→E77_1 **away** at 55 m; missed on E77_1. | C primary |
| 200009 | Reached E76_0 with about 541 m left; accepted E76_0→E76_1 **away** at 66 m, then J105 mapped to E77_2. Later reached E77_1; the final toward E77_0 request was accepted at 96 m but recentered and timed out. | C primary, B secondary |
| 200010 | Began toward D3 with about 814 m left and reached E76_0 with 313 m left. Later accepted two away changes on E76 at 104 m and 73 m, then two more on E77, and missed on E77_5. Earlier E76_1→0 pending attempts stalled before one succeeded. | C primary, B earlier contributor |
| 200002 | Reached E76_0 and E77_0, then oscillated between E77_0 and E77_1 near the exit. | C primary |
| 200036 | Started in E76_0, accepted an away change about 53 m before J105, then took the wrong J105/E77 lane path. | C primary |

Here A means too-late first planning toward D3; B means a toward request was
made but physical completion was blocked or timed out; C means unnecessary or
repeated away changes. **A was not supported in these five cases.** For B,
SUMO's live lane-change state on seed `200010` reported adjacent target-lane
leader/follower blockage or overlap on 123 of 180 sampled steps across the
three stalled E76 pending windows. On seed `200009`, the final E77 pending
window reported leader blockage on 18 of 60 steps and follower blockage on 8;
the car moved laterally but recentered before the pending timeout. This
supports a real traffic/safety-gap contribution, not merely a pending-state
counter artifact. These counts are diagnostic flags, not an isolated causal
test of gap policy or lane-change duration. In all three lane-3 failures, the
policy's late **accepted away** changes are the clearest shared mechanism;
do not reopen terminal in-place lateral movement as a workaround.

**Diagnostic artifacts (ignored, retained):**
`tmp/ai/e76-d3/nav_200k/failure_diagnostic_200k_5.json` (SHA-256
`f813c4030357e91466b999deaa2a45f87b6e68bb448faa31a6dd2d7f301f4dfd`)
and `tmp/ai/e76-d3/nav_200k/failure_lane_state_200k_2.json` (SHA-256
`dd1b404092964167fbeadeda154ba99933e357ecbd491f6ddcad128ce57c7d91`).

**Artifacts (remote, ignored):**
`tmp/ai/e76-d3/nav_200k/E76D3-NAV136-GAMMA099-S42-200K.jsonl`,
`tmp/ai/e76-d3/nav_200k/eval_100k_5.json` (SHA-256
`b6c849ca8d7ebcd383fdd5609a05e75db230a5fe7af013e5cd017af93be8666f`),
and `tmp/ai/e76-d3/nav_200k/eval_200k_25.json` (SHA-256
`8d7fb2843300a34a49a0870979d6a1b00ffcffdd1a3eeb76ddfaf7fcbeb82a87`).
Models and replay are under `models_experiments/e76-d3/` in that checkout.

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
- **Fact:** NAV136 completed at 16/25 strict arrivals; a paired 134-input,
  200k control is now training in a separate checkout. The five NAV136 missed
  cases inspected so far chiefly involve accepted away changes near J105 or
  the exit; two also showed target-lane blockage during pending changes.
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

1. **Wrong-direction SAC actions in old checkpoints.** Their dominant failure
   was a large excess of commands away from D3. The new NAV136 policy has a
   much lower away-request fraction, but still has 9/25 missed exits.
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
