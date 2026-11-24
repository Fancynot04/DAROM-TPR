"""Publish completed evaluation bundles and fetch them through OpenSSH."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid


ROOT = ".local/share/darom-trajectories"
TRACE = "eval_trajectory.jsonl.gz"
SUMMARY = "summary.json"
FILES = (SUMMARY, TRACE, "episodes.csv")
MANIFEST = "manifest.json"


def component(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid archive path component: {value!r}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    with partial.open("wb") as stream:
        write(stream)
    partial.replace(path)


def encode(row):
    return (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()


def relative(row):
    step = row["step"]
    if row["version"] != 1 or type(step) is not int or step < 0:
        raise ValueError("Invalid trajectory archive manifest")
    digests = row["sha256"]
    if set(digests) != set(FILES) or any(not re.fullmatch(r"[a-f0-9]{64}", d) for d in digests.values()):
        raise ValueError("Invalid trajectory archive checksums")
    batch = component(row["batch"])
    identity = hashlib.sha256(encode(digests)).hexdigest()
    return Path(component(row["experiment"]), f"step_{step}", f"{batch}-{identity}")


def publish(summary, experiment, root, batch="eval"):
    summary = Path(summary)
    sources = {SUMMARY: summary, **{name: summary.with_name(name) for name in FILES[1:]}}
    report_bytes = summary.read_bytes()
    report = json.loads(report_bytes)
    digests = {name: sha256(sources[name]) for name in FILES[1:]}
    digests[SUMMARY] = hashlib.sha256(report_bytes).hexdigest()
    trajectory = report.get("trajectory") or {}
    if trajectory.get("sha256") != digests[TRACE] or trajectory.get("file") != TRACE:
        raise ValueError("The trajectory does not match the evaluation report")
    row = dict(version=1, experiment=experiment, step=report["training_step"], batch=batch,
               sha256=digests, checkpoint_sha256=report["checkpoint_sha256"], episodes=report["episodes"],
               seed_start=report["seed_start"])
    destination = Path(root).expanduser() / relative(row)
    if (destination / MANIFEST).is_file():
        if read_manifest(destination / MANIFEST) != row or not complete(destination, row, verify=True):
            raise ValueError(f"Archive failed verification: {destination}")
        return destination
    for name, source in sources.items():
        def copy(stream):
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, stream)
            stream.flush()
            if sha256(Path(stream.name)) != digests[name]:
                raise ValueError(f"Source changed during publication: {source}")

        atomic_write(destination / name, copy)
    atomic_write(destination / MANIFEST, lambda stream: stream.write(encode(row)))
    return destination


def read_manifest(path):
    row = json.loads(path.read_bytes())
    relative(row)
    return row


def complete(directory, row, *, verify=False):
    return all((directory / name).is_file() and (not verify or sha256(directory / name) == row["sha256"][name])
               for name in FILES)


def local_entries(cache, archive):
    entries = {}
    for path in cache.glob(f"*/step_*/*/{MANIFEST}"):
        try:
            row = read_manifest(path)
            if cache / relative(row) == path.parent and complete(path.parent, row):
                entries[str(relative(row))] = (row, path.parent)
        except (OSError, ValueError, TypeError, KeyError):
            continue
    for summary in sorted(archive.glob("*/evidence/evaluations/step_*/**/summary.json")):
        trace = summary.with_name(TRACE)
        try:
            step = int(summary.relative_to(archive).parts[3].removeprefix("step_"))
        except ValueError:
            continue
        if trace.is_file():
            experiment = summary.relative_to(archive).parts[0]
            batch = str(summary.parent.relative_to(archive / experiment / "evidence" / "evaluations"))
            entries[str(summary)] = (dict(experiment=experiment, step=step, batch=batch), summary.parent)
    return entries


def ssh(host, port, command):
    # The terminal that launched Streamlit supplies any OpenSSH password prompt.
    return ["ssh", "-o", "ConnectTimeout=8", "-p", str(port), "--", host, command]


def remote_entries(host, port, root):
    remote = shlex.quote(root)
    command = (f"if test -d {remote}; then "
               f"find {remote} -mindepth 4 -maxdepth 4 -type f -name {MANIFEST} -exec cat {{}} +; fi")
    payload = subprocess.check_output(ssh(host, port, command), timeout=120)
    entries = {}
    for line in payload.splitlines():
        row = json.loads(line)
        entries[str(relative(row))] = row
    return entries


def download(host, port, root, cache, row):
    location = relative(row)
    destination = cache / location
    if (destination / MANIFEST).is_file() and complete(destination, row, verify=True):
        return destination
    remote = shlex.quote(str(Path(root) / location))
    command = f"tar --format=ustar -C {remote} -cf - {' '.join(FILES)}"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=cache) as payload:
        subprocess.run(ssh(host, port, command), stdout=payload, check=True, timeout=600)
        payload.seek(0)
        with tarfile.open(fileobj=payload, mode="r:") as bundle:
            members = bundle.getmembers()
            if sorted(member.name for member in members) != sorted(FILES) or any(not m.isfile() for m in members):
                raise ValueError("Unexpected files in the remote evaluation bundle")
            for name in FILES:
                def extract(stream):
                    with bundle.extractfile(name) as incoming:
                        shutil.copyfileobj(incoming, stream)
                    stream.flush()
                    if sha256(Path(stream.name)) != row["sha256"][name]:
                        raise ValueError(f"Downloaded file failed SHA-256 verification: {name}")

                atomic_write(destination / name, extract)
    atomic_write(destination / MANIFEST, lambda stream: stream.write(encode(row)))
    return destination


def add_publish_arguments(parser):
    parser.add_argument("--publish-experiment-id", help="Full exp ID without the darom/ prefix")
    parser.add_argument("--trajectory-archive-root", type=Path, default=Path.home() / ROOT)
    parser.add_argument("--trajectory-batch", default="eval")


def publish_requested(summary, args):
    if args.publish_experiment_id:
        print("Trajectory archive:", publish(summary, args.publish_experiment_id,
                                              args.trajectory_archive_root, args.trajectory_batch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--archive-root", type=Path, default=Path.home() / ROOT)
    parser.add_argument("--batch", default="eval")
    args = parser.parse_args()
    print(publish(args.summary, args.experiment_id, args.archive_root, args.batch))


if __name__ == "__main__":
    main()
