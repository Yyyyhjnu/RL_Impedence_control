"""
Evaluate learned force-impedance PPO policy against a fixed-admittance baseline.

Examples:
  python eval_force_impedance.py --model runs/ppo_force_impedance/ppo_force_impedance_final.zip
  python eval_force_impedance.py --baseline_only --render
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public"))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public" / "stable-baselines3-acmpc"))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public" / "training_modules"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from ur5e_force_impedance_env import Ur5eForceImpedanceEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="", help="PPO .zip model path.")
    p.add_argument("--baseline_only", action="store_true")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "runs" / "force_impedance_eval"))
    p.add_argument(
        "--scene_xml",
        type=str,
        default=str(ROOT / "model" / "universal_robots_ur5e" / "scene-30.xml"),
    )
    p.add_argument("--episode_steps", type=int, default=2200)
    p.add_argument("--line_speed_y", type=float, default=-0.01)
    p.add_argument("--line_total_disp", type=float, default=0.30)
    p.add_argument("--target_force_z", type=float, default=-50.0)
    p.add_argument("--baseline_b_z", type=float, default=1000.0)
    p.add_argument("--baseline_k_z", type=float, default=0.0)
    return p.parse_args()


def make_env(args: argparse.Namespace, seed: int, render: bool) -> Ur5eForceImpedanceEnv:
    return Ur5eForceImpedanceEnv(
        scene_xml=args.scene_xml,
        episode_steps=args.episode_steps,
        line_speed_y=args.line_speed_y,
        line_total_disp=args.line_total_disp,
        base_target_force_z=args.target_force_z,
        render=render,
        seed=seed,
    )


def fixed_action_for(env: Ur5eForceImpedanceEnv, b_z: float, k_z: float) -> np.ndarray:
    b_min, b_max = env.b_z_range
    k_min, k_max = env.k_z_range
    b_norm = 2.0 * (float(b_z) - b_min) / max(b_max - b_min, 1e-9) - 1.0
    k_norm = 2.0 * (float(k_z) - k_min) / max(k_max - k_min, 1e-9) - 1.0
    return np.clip(np.array([b_norm, k_norm, 0.0], dtype=np.float32), -1.0, 1.0)


def run_episode(
    env: Ur5eForceImpedanceEnv,
    policy: Optional[PPO],
    label: str,
    episode_idx: int,
    baseline_action: Optional[np.ndarray] = None,
) -> list[dict[str, float | int | str]]:
    obs, _ = env.reset(seed=episode_idx)
    rows: list[dict[str, float | int | str]] = []
    done = False
    step = 0
    while not done:
        if policy is None:
            assert baseline_action is not None
            action = baseline_action
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        rows.append(
            {
                "label": label,
                "episode": episode_idx,
                "step": step,
                "reward": float(reward),
                "force_z": float(info.get("force_z", 0.0)),
                "target_force_z": float(info.get("target_force_z", 0.0)),
                "force_error_z": float(info.get("force_error_z", 0.0)),
                "b_z": float(info.get("b_z", 0.0)),
                "k_z": float(info.get("k_z", 0.0)),
                "progress": float(info.get("progress", 0.0)),
                "ik_success": int(bool(info.get("ik_success", True))),
            }
        )
        step += 1
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not rows:
        return {"rms_force_error": float("nan"), "max_abs_force_error": float("nan"), "mean_reward": float("nan")}
    force_error = np.array([float(r["force_error_z"]) for r in rows], dtype=np.float64)
    reward = np.array([float(r["reward"]) for r in rows], dtype=np.float64)
    return {
        "rms_force_error": float(np.sqrt(np.mean(np.square(force_error)))),
        "max_abs_force_error": float(np.max(np.abs(force_error))),
        "mean_reward": float(np.mean(reward)),
        "final_progress": float(rows[-1]["progress"]),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_force(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({str(r["label"]) for r in rows})
    plt.figure(figsize=(10, 5))
    for label in labels:
        selected = [r for r in rows if str(r["label"]) == label and int(r["episode"]) == 0]
        if not selected:
            continue
        steps = np.array([int(r["step"]) for r in selected], dtype=np.int32)
        force = np.array([float(r["force_z"]) for r in selected], dtype=np.float64)
        target = np.array([float(r["target_force_z"]) for r in selected], dtype=np.float64)
        plt.plot(steps, force, label=f"{label}: force_z")
        plt.plot(steps, target, linestyle="--", label=f"{label}: target")
    plt.xlabel("step")
    plt.ylabel("force z (N)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, float | int | str]] = []

    baseline_env = make_env(args, args.seed, args.render)
    baseline_action = fixed_action_for(baseline_env, args.baseline_b_z, args.baseline_k_z)
    for ep in range(int(args.episodes)):
        all_rows.extend(
            run_episode(
                baseline_env,
                policy=None,
                label="fixed_admittance",
                episode_idx=args.seed + ep,
                baseline_action=baseline_action,
            )
        )
    baseline_env.close()

    if not args.baseline_only:
        if not args.model:
            raise ValueError("未指定 --model；如只评估固定导纳，请添加 --baseline_only。")
        model = PPO.load(args.model)
        policy_env = make_env(args, args.seed + 1000, args.render)
        for ep in range(int(args.episodes)):
            all_rows.extend(
                run_episode(
                    policy_env,
                    policy=model,
                    label="ppo_impedance",
                    episode_idx=args.seed + ep,
                )
            )
        policy_env.close()

    csv_path = out_dir / "force_impedance_eval.csv"
    png_path = out_dir / "force_tracking_episode0.png"
    write_csv(csv_path, all_rows)
    plot_force(png_path, all_rows)

    for label in sorted({str(r["label"]) for r in all_rows}):
        label_rows = [r for r in all_rows if str(r["label"]) == label]
        stats = summarize(label_rows)
        print(
            f"[{label}] rms_force_error={stats['rms_force_error']:.3f} "
            f"max_abs_force_error={stats['max_abs_force_error']:.3f} "
            f"mean_reward={stats['mean_reward']:.3f} final_progress={stats['final_progress']:.3f}"
        )
    print(f"[eval] wrote {csv_path}")
    print(f"[eval] wrote {png_path}")


if __name__ == "__main__":
    main()
