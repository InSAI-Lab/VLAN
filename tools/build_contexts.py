from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlan_final.tasks.contexts import build_context
from vlan_final.utils.dataset_io import iter_traj_names
from vlan_final.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/test")
    parser.add_argument("--artifacts_root", type=str, default="artifacts")
    parser.add_argument("--scene_names", type=str, default="")
    args = parser.parse_args()

    data_root = REPO_ROOT / args.data_root
    artifacts_root = REPO_ROOT / args.artifacts_root
    scene_names = [s.strip() for s in args.scene_names.split(",") if s.strip()] if args.scene_names else sorted(p.name for p in data_root.iterdir() if p.is_dir())

    for scene_name in scene_names:
        scene_dir = data_root / scene_name
        for traj_name in iter_traj_names(scene_dir):
            targets = load_json(artifacts_root / "targets" / scene_name / traj_name / "agent_targets.json")
            payload = build_context(scene_dir, traj_name, targets)
            out_path = artifacts_root / "contexts" / scene_name / f"{traj_name}.json"
            save_json(out_path, payload)
            print(f"[OK] contexts {scene_name}/{traj_name}")


if __name__ == "__main__":
    main()
