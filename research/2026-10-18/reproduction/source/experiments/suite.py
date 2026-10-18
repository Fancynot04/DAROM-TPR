"""Bounded first-round queue; each SUMO environment lives in its own process."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

from experiments.run import ROOT, preflight, write_json


def run_suite(cfg, output=None, smoke=False, smoke_steps=128):
    preflight(cfg, strict=not smoke)
    root = Path(output or cfg['runtime']['output_root'])
    if not root.is_absolute():
        root = ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    # Keep this descriptor open throughout the queue, including all active children.
    with (root / '.suite.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'A suite is already running under {root}') from None
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = root / f"{'smoke-' if smoke else ''}suite-{stamp}-{uuid4().hex[:8]}"
        directory.mkdir()
        # Queued jobs all use the same configuration, even if the source JSON changes.
        config_path = directory / 'config.json'
        write_json(config_path, cfg)
        jobs = [{'group': group, 'seed': seed, 'state': 'queued'}
                for seed in cfg['training_seeds'] for group in cfg['groups']]
        state = {'status': 'running', 'pid': os.getpid(), 'parallel_runs': cfg['runtime']['parallel_runs'],
                 'started_at': datetime.now(timezone.utc).isoformat(), 'jobs': jobs}

        def save():
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            pending = directory / 'queue.json.pending'
            write_json(pending, state)
            pending.replace(directory / 'queue.json')

        save()
        print(f'Suite state: {directory / "queue.json"}', flush=True)
        active = []
        failed = False
        try:
            while active or any(job['state'] == 'queued' for job in jobs):
                # Observe all exits before refilling slots: failure stops new launches.
                for entry in active[:]:
                    process, job, log = entry
                    code = process.poll()
                    if code is not None:
                        log.close()
                        job.update(state='completed' if code == 0 else 'failed', returncode=code,
                                   finished_at=datetime.now(timezone.utc).isoformat())
                        failed |= code != 0
                        active.remove(entry)
                        save()
                if failed:
                    for job in jobs:
                        if job['state'] == 'queued':
                            job['state'] = 'blocked_by_failure'
                    save()
                while not failed and len(active) < cfg['runtime']['parallel_runs']:
                    job = next((job for job in jobs if job['state'] == 'queued'), None)
                    if job is None:
                        break
                    job_dir = directory / f"{job['group']}-seed{job['seed']}"
                    job_dir.mkdir()
                    command = [sys.executable, '-B', '-m', 'experiments.run', 'train',
                               '--config', str(config_path), '--group', job['group'],
                               '--seed', str(job['seed']), '--output', str(job_dir)]
                    if smoke:
                        command.extend(['--smoke', '--smoke-steps', str(smoke_steps),
                                        '--tracking', cfg['tracking']['mode']])
                    log = (job_dir / 'process.log').open('w')
                    try:
                        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT)
                    except BaseException:
                        log.close()
                        raise
                    job.update(state='running', pid=process.pid, output=str(job_dir), command=command,
                               started_at=datetime.now(timezone.utc).isoformat())
                    active.append((process, job, log))
                    save()
                if active:
                    time.sleep(2)
            state['status'] = 'failed' if failed else 'completed'
            save()
        except BaseException as error:
            state.update(status='interrupted', error=str(error))
            # Do not leave unnoticed children or start replacements after interruption.
            # Already active runs finish normally; exact simulator resume is unsupported.
            save()
            for process, job, log in active:
                code = process.wait()
                log.close()
                job.update(state='completed' if code == 0 else 'failed', returncode=code)
            for job in jobs:
                if job['state'] == 'queued':
                    job['state'] = 'blocked_by_interruption'
            save()
            raise
        if failed:
            raise RuntimeError(f'A training job failed; see {directory / "queue.json"}')
