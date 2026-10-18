"""Opt-in real SUMO/SAC integration test. All generated files stay under /tmp.

Run: .venv/bin/python -B -m tests.test_long_resume --device cpu
The test does not upload metrics or start formal long training.
"""

import argparse
import io
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.checkpoints import validate_checkpoint
from experiments.run import ROOT


def equal(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu()), f'Tensor mismatch: {path}; max={float((a.cpu()-b.cpu()).abs().max())}'
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (left, right) in enumerate(zip(a, b)):
            equal(left, right, f'{path}/{i}')
    else:
        assert a == b, (path, a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix=f'darom-long-resume-{args.device}-'))
    print(f'Evidence: {root}', flush=True)

    def run(name, extra, expected=0):
        output = root / name
        command = [sys.executable, '-B', '-m', 'experiments.long_run', 'train', '--output', str(output), *extra]
        with (root / f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if expected == 0 and result.returncode != 0:
            raise AssertionError((root / f'{name}.log').read_text())
        if expected != 0:
            assert result.returncode != 0, 'Corrupt checkpoint unexpectedly accepted'
            return
        return next(output.glob('D-seed42-*'))

    verification = ['--verification', '--verification-device', args.device]
    continuous = run('continuous', verification)
    warmup = run('warmup', [*verification, '--stop-at', '73'])
    warmup_checkpoint = warmup / 'checkpoints' / 'step-000000073'
    assert json.loads((warmup / 'status.json').read_text())['gradient_updates'] == 0
    paused = run('paused', ['--resume', str(warmup_checkpoint), '--stop-at', '273'])
    cp = paused / 'checkpoints' / 'step-000000273'
    assert json.loads((paused / 'status.json').read_text())['status'] == 'paused'
    with (cp / 'state.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    assert len(saved['environment']['episode']['actions']) > 0, 'Need a mid-episode checkpoint'
    assert saved['callback']['completed_evals'] == [256]
    resumed = run('resumed', ['--resume', str(cp)])
    left, right = [d / 'checkpoints' / 'step-000000512' for d in (continuous, resumed)]
    for d in (continuous, resumed):
        status = json.loads((d / 'status.json').read_text())
        assert status['status'] == 'completed' and status['steps'] == 512 and status['gradient_updates'] == 412
    with zipfile.ZipFile(left / 'model.zip') as a, zipfile.ZipFile(right / 'model.zip') as b:
        for name in a.namelist():
            if name.endswith('.pth'):
                equal(torch.load(io.BytesIO(a.read(name)), map_location='cpu', weights_only=False),
                      torch.load(io.BytesIO(b.read(name)), map_location='cpu', weights_only=False), name)
    with (left / 'replay-buffer.pkl').open('rb') as f:
        a = pickle.load(f)
    with (right / 'replay-buffer.pkl').open('rb') as f:
        b = pickle.load(f)
    assert a.full and b.full and a.pos == b.pos
    for name in ['observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts']:
        equal(getattr(a, name), getattr(b, name), name)
    equal(json.loads((continuous / 'eval-512/results.json').read_text()),
          json.loads((resumed / 'eval-512/results.json').read_text()), 'evaluation')
    with (left / 'state.pkl').open('rb') as f:
        a = pickle.load(f)
    with (right / 'state.pkl').open('rb') as f:
        b = pickle.load(f)
    equal(a['rng'], b['rng'], 'rng')
    for key in ['actions', 'trace', 'simulation_time', 'vehicles', 'space_rng']:
        equal(a['environment']['episode'][key], b['environment']['episode'][key], key)
    for key in ['rewards', 'episode_reward', 'episode_length', 'completed_evals', 'last_processed_step']:
        equal(a['callback'][key], b['callback'][key], key)
    # Historical checkpoint branching cannot alter or overwrite the original run.
    assert json.loads((paused / 'status.json').read_text())['status'] == 'paused'
    manifest = json.loads((cp / 'manifest.json').read_text())
    with patch('experiments.checkpoints.source_hashes', return_value={}):
        try:
            validate_checkpoint(cp, manifest['environment'])
        except RuntimeError as error:
            assert 'source differs' in str(error)
        else:
            raise AssertionError('Source drift accepted')
    corrupt = root / 'corrupt'
    shutil.copytree(cp, corrupt)
    with (corrupt / 'replay-buffer.pkl').open('ab') as f:
        f.write(b'corruption')
    run('corrupt-rejected', ['--resume', str(corrupt)], expected=1)
    assert 'checksum mismatch' in (root / 'corrupt-rejected.log').read_text()
    result = {'device': args.device, 'continuous': str(continuous), 'resumed': str(resumed),
              'pause_steps': [73, 273], 'final_steps': 512, 'gradient_updates': 412,
              'mid_episode_actions': len(saved['environment']['episode']['actions']),
              'bitwise_equal_weights_optimizers_rng_replay': True,
              'equal_evaluation': True, 'corruption_and_source_drift_rejected': True}
    (root / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
