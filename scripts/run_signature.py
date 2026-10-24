#!/usr/bin/env python3
"""Create or verify the immutable inputs for a resumable training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_command(command: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(command):
        if command[index] == "--resume_checkpoint":
            if index + 1 >= len(command):
                raise ValueError("--resume_checkpoint is missing its path")
            index += 2
            continue
        normalized.append(command[index])
        index += 1
    return normalized


def labeled_hash(value: str) -> tuple[str, dict[str, str]]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=PATH, got {value!r}")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not label or not path.is_file():
        raise ValueError(f"invalid artifact {value!r}")
    return label, {"path": str(path), "sha256": sha256_file(path)}


def build_signature(args: argparse.Namespace) -> dict:
    sources = {}
    for raw_path in sorted(args.source):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"missing source file: {path}")
        sources[str(path)] = sha256_file(path)
    artifacts = dict(labeled_hash(value) for value in args.artifact)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("training command is empty")
    return {
        "schema_version": 1,
        "command_without_resume": normalize_command(command),
        "source_sha256": sources,
        "artifacts": artifacts,
    }


def describe_difference(previous: dict, current: dict) -> str:
    lines = []
    for key in ("command_without_resume", "source_sha256", "artifacts"):
        if previous.get(key) != current.get(key):
            lines.append(f"- {key} changed")
    return "\n".join(lines) or "- signature schema changed"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-existing", action="store_true")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        current = build_signature(args)
    except ValueError as error:
        raise SystemExit(f"RUN SIGNATURE FAILED: {error}") from error

    output = args.output.resolve()
    if args.require_existing and not output.is_file():
        raise SystemExit(
            f"RUN SIGNATURE FAILED: refusing to resume without {output}"
        )
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != current:
            raise SystemExit(
                "RUN SIGNATURE FAILED: refusing to mix incompatible training runs:\n"
                + describe_difference(previous, current)
            )

    atomic_write_json(output, current)
    print(f"Run signature verified: {output}")


if __name__ == "__main__":
    main()
