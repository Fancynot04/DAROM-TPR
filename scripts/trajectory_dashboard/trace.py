"""Validate a published evaluation and index its trajectory by seed."""

import argparse
import gzip
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path


EGO_IDS = ("ego_1", "ego_2")
TRACE_FORMAT = "dual-ego-eval-jsonl-gzip-v1"
CACHE_VERSION = 1
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "tmp/ai/trajectory-dashboard/index"


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_total(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label}: trajectory total {actual} differs from {expected}")


def _read_episodes(trace):
    """Yield one episode at a time and require contiguous file and episode steps."""
    seed = None
    rows = []
    last_global_step = 0
    episode_number = 0
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at trajectory line {line_number}") from error
            event = row.get("event")
            if event == "reset":
                if seed is not None:
                    yield seed, rows
                seed = row["seed"]
                rows = []
                episode_number += 1
                if row["episode"] != episode_number or row["global_step"] != last_global_step:
                    raise ValueError(f"Invalid reset at trajectory line {line_number}")
                continue
            last_global_step += 1
            if (event != "step" or seed is None or row["seed"] != seed
                    or row["episode"] != episode_number or row["step"] != len(rows) + 1
                    or row["global_step"] != last_global_step):
                raise ValueError(f"Discontinuous trajectory at line {line_number}")
            rows.append(row)
    if seed is not None:
        yield seed, rows


def _validate_episode(seed, rows, expected):
    if len(rows) != expected["global_steps"]:
        raise ValueError(f"Seed {seed}: incorrect number of steps")
    team_rewards = [row["team_reward"] for row in rows]
    if sum(value != 0 for value in team_rewards) > 1:
        raise ValueError(f"Seed {seed}: multiple team reward payments")
    _verify_total(math.fsum(team_rewards), expected["team_reward"], f"Seed {seed}, team")
    for ego in EGO_IDS:
        observations = [row["egos"][ego] for row in rows]
        terminal = [obj["terminal"] for obj in observations if obj["terminal"] is not None]
        if terminal != [expected["outcomes"][ego]]:
            raise ValueError(f"Seed {seed}, {ego}: terminal events {terminal} do not match summary")
        label = f"Seed {seed}, {ego}"
        _verify_total(math.fsum(obj["reward"] for obj in observations), expected["rewards"][ego], label)
        component_totals = expected["reward_components"][ego]
        for obj in observations:
            if set(obj["components"]) != set(component_totals):
                raise ValueError(f"{label}: unexpected reward components")
            _verify_total(math.fsum(obj["components"].values()), obj["reward"], f"{label}, step reward")
        for key, expected_value in component_totals.items():
            total = math.fsum(obj["components"][key] for obj in observations)
            _verify_total(total, expected_value, f"{label}, {key}")


@dataclass(frozen=True)
class TraceIndex:
    """A read-only handle to a verified SQLite cache."""

    path: Path
    metadata: dict

    def episode(self, seed):
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as conn:
            row = conn.execute("SELECT steps FROM episodes WHERE seed = ?", (seed,)).fetchone()
        if row is None:
            raise KeyError(f"Seed {seed} is not in this evaluation")
        return json.loads(gzip.decompress(row[0]))


def _read_index(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        return TraceIndex(path, json.loads(conn.execute("SELECT value FROM metadata").fetchone()[0]))


def build_index(trace, summary=None, cache_dir=DEFAULT_CACHE):
    """Use both source file hashes as the cache key, publishing only verified indexes."""
    trace = Path(trace).resolve()
    summary = Path(summary or trace.with_name("summary.json")).resolve()
    summary_bytes = summary.read_bytes()
    report = json.loads(summary_bytes)
    descriptor = report["trajectory"]
    if not descriptor or descriptor["format"] != TRACE_FORMAT:
        raise ValueError(f"Expected trajectory format {TRACE_FORMAT}")
    trace_stat = trace.stat()
    trace_hash = file_sha256(trace)
    if trace_hash != descriptor["sha256"]:
        raise ValueError("Trajectory SHA-256 does not match summary.json")
    evaluation = report["environment"]
    if tuple(evaluation["agent_ids"]) != EGO_IDS:
        raise ValueError("Summary does not contain the expected two Ego IDs")
    dt, gamma = evaluation["step_length_s"], report["trainer_config"]["gamma"]
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(gamma) and 0 < gamma <= 1):
        raise ValueError("Invalid SUMO time step or discount factor")
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    target = cache_dir / f"v{CACHE_VERSION}-{trace_hash}-{summary_hash}.sqlite3"
    if target.exists():
        return _read_index(target)

    records = report["records"]
    seeds = [row["seed"] for row in records]
    if len(seeds) != len(set(seeds)) or len(seeds) != report["episodes"] or not seeds:
        raise ValueError("Invalid evaluation seed list in summary.json")
    with tempfile.TemporaryDirectory(prefix="building-", dir=cache_dir) as work:
        partial = Path(work) / target.name
        step_count = 0
        observed_seeds = []
        with closing(sqlite3.connect(partial)) as conn, conn:
            conn.execute("CREATE TABLE episodes (seed INTEGER PRIMARY KEY, steps BLOB NOT NULL)")
            conn.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
            for index, (seed, rows) in enumerate(_read_episodes(trace)):
                if index >= len(records) or seed != seeds[index]:
                    raise ValueError(f"Unexpected trajectory seed {seed}")
                _validate_episode(seed, rows, records[index])
                observed_seeds.append(seed)
                step_count += len(rows)
                payload = gzip.compress(json.dumps(rows, separators=(",", ":")).encode(), compresslevel=3)
                conn.execute("INSERT INTO episodes VALUES (?, ?)", (seed, payload))
            if (observed_seeds != seeds or len(seeds) != descriptor["episodes"]
                    or step_count != descriptor["step_records"]
                    or step_count != report["global_steps"]):
                raise ValueError("Trajectory episode or step counts do not match summary.json")
            metadata = {
                "checkpoint": report["training_step"], "checkpoint_sha256": report["checkpoint_sha256"],
                "trace_sha256": trace_hash, "summary_sha256": summary_hash,
                "step_records": step_count, "step_length_s": dt, "gamma": gamma,
                "shared_actor": report.get("training_config", {}).get("shared_actor"),
                "training_seed": report.get("training_seed"),
                "episodes": [{"seed": row["seed"], "outcomes": row["outcomes"],
                              "steps": row["global_steps"]} for row in records],
            }
            conn.execute("INSERT INTO metadata VALUES (?)", (json.dumps(metadata),))
        newer = trace.stat()
        if (newer.st_mtime_ns, newer.st_size, newer.st_ino) != (
            trace_stat.st_mtime_ns, trace_stat.st_size, trace_stat.st_ino,
        ) or summary.read_bytes() != summary_bytes:
            raise ValueError("Evaluation source changed during indexing")
        os.replace(partial, target)
    return TraceIndex(target, metadata)


def published_evaluations(directory):
    """Find complete trace/report pairs in the evaluation output layout."""
    pairs = {}
    for summary in Path(directory).glob("step_*/summary.json"):
        step = summary.parent.name.removeprefix("step_")
        trace = summary.with_name("eval_trajectory.jsonl.gz")
        if not step.isascii() or not step.isdecimal() or not trace.is_file():
            continue
        checkpoint = int(step)
        if checkpoint in pairs:
            raise ValueError(f"Multiple evaluation directories for checkpoint {checkpoint}")
        pairs[checkpoint] = (trace.resolve(), summary.resolve())
    if not pairs:
        raise ValueError(f"No complete evaluation trace/report pairs in {directory}")
    return dict(sorted(pairs.items()))


def argument_parser(description, *, allow_eval_dir=False, allow_catalog=False):
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group(required=not allow_catalog)
    source.add_argument("--trace", type=Path, help="Published eval_trajectory.jsonl.gz")
    if allow_eval_dir:
        source.add_argument("--eval-dir", type=Path, help="Parent of step_<checkpoint> evaluation directories")
    parser.add_argument("--summary", type=Path, help="Defaults to summary.json next to the trajectory")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    return parser


def main():
    args = argument_parser(__doc__).parse_args()
    archive = build_index(args.trace, args.summary, args.cache_dir)
    print(json.dumps(archive.metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
