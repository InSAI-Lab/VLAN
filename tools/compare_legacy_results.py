from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            out.update(flatten(v, key))
    else:
        out[prefix] = d
    return out


def compare(new_path: Path, old_path: Path) -> dict[str, Any]:
    new = load_json(new_path)
    old = load_json(old_path)
    fnew = flatten(new)
    fold = flatten(old)
    keys = sorted(set(fnew) | set(fold))
    mismatches = []
    for key in keys:
        if fnew.get(key) != fold.get(key):
            mismatches.append({"key": key, "new": fnew.get(key), "old": fold.get(key)})
    return {
        "new_path": str(new_path),
        "old_path": str(old_path),
        "match": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", required=True)
    args = parser.parse_args()
    result = compare(Path(args.new), Path(args.old))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
