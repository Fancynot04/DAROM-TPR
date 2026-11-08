# DAROM-TPR: E76→D3 single-agent driving

This repository contains the single-agent DAROM SAC/GRU environment for an Ego
vehicle that starts on E76 and reaches D3 through E77. The task is fixed:

- Ego origin: E76; destination: D3; departure lane: random legal E76 lane
- scenarios: `easy`, `medium`, `hard`
- SUMO step: 0.1 s; episode limit: 400 s / 4,000 steps
- road limit: 120 km/h (`33.3333333333 m/s`) in `sumo_new/Antwerp.net.xml`
- one Ego agent, `laneChangeMode=0`, custom safety interception disabled
- pending lane-change requests are observational only and never block a new command
- original DAROM delayed observation, action semantics, SAC and GRU encoder are retained

## Layout

```text
train_sac_delay.py          fresh SAC training entry
evaluate_sac_delay.py      checkpoint evaluation entry
src/merging.py             single-agent SUMO environment
src/wrapper.py             delay/action wrappers
src/delay_encoder.py       DAROM delayed-state encoders
src/exit_e76_d3_config.py  E76→D3 experiment profiles
src/geometry.py            traffic geometry helpers
src/utils.py               shared utilities
src/road_boundary.py       road-boundary diagnostics
src/rewards/events.py      terminal-event rewards
sumo_new/Antwerp.net.xml   Antwerp network with the fixed corridor limit
sumo_new/Antwerp_*_exit_e76_d3.rou.xml
                            Easy/Medium/Hard E76→D3 routes
sumo_new/Antwerp.sumocfg   Easy E76→D3 GUI configuration
```

Old entry, legacy O3→D1, dual-agent routes, detector definitions, root
`config.yaml`, old model archives, and continuation/replay-resume paths are not
part of the current task.

## Environment setup

Install the dependencies in `requirements.txt`, install SUMO, and export
`SUMO_HOME` so that its Python tools are available:

```bash
export SUMO_HOME=/path/to/sumo
python3 -m pip install -r requirements.txt
```

The training and evaluation code starts SUMO directly with the selected route
file. It does not load detector `additional` files and does not modify lane
speed limits through TraCI.

## Training

Every run starts from fresh random weights and an empty replay buffer. Models
and logs should use a new run name and remain in ignored output directories.

```bash
python3 train_sac_delay.py \
  --scenario easy \
  --seed 42 \
  --total_timesteps 200000 \
  --checkpoint_freq 200000 \
  --gamma 0.99 \
  --reward-config configs/rewards/events_only_v1.json \
  --wandb_mode disabled
```

The command-line values are final; no historical Phase or reward file can
override seed, scenario, budget, checkpoint interval, or gamma. The B0 reward
is zero on non-terminal steps and settles exactly one event: success `+50`,
collision/missed-exit/teleport/timeout `-100`.

## Evaluation

```bash
python3 evaluate_sac_delay.py \
  models_experiments/e76-d3/<run>.zip \
  --scenario easy \
  --episodes 50 --seed 200001 \
  --no-safety \
  --output tmp/ai/e76-d3/eval.json
```

Evaluation uses the checkpoint's E76→D3 reward metadata and the selected
scenario. Safety is off for the matched task contract unless explicitly
requested for a separate diagnostic.

## GUI inspection

`sumo_new/Antwerp.sumocfg` opens the Easy E76→D3 route with a 600-second GUI
end time and 0.1-second step. The 400-second/4,000-step task timeout is owned
by the environment and is independent of this GUI end time. For Medium or Hard inspection, replace the route file
in the GUI configuration with the corresponding
`Antwerp_<scenario>_exit_e76_d3.rou.xml` file.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The real SUMO integration suite is opt-in:

```bash
python3 -m tests.test_sumo_integration
```

It checks all three scenarios, the fixed route contract, 0.1-second stepping,
4,000-step episode limits, `laneChangeMode=0`, and the 120 km/h network speed.
