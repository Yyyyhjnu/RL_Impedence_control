"""
仅在波浪地形（scene_45.xml + hfield 注入）上评估 PPO 力阻抗策略。

复用 eval_force_impedance_three_way_wave.py 中的 Ur5eWaveForceImpedanceEnv；
不包含 fixed_admittance / adaptive_sigma_phi。

Example:
  python eval_ppo_force_impedance_wave_only.py --model runs/ppo_force_impedance/ppo_force_impedance_final.zip
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public"))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public" / "stable-baselines3-acmpc"))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public" / "training_modules"))

from eval_force_impedance_three_way import (  # noqa: E402
    plot_force_comparison,
    run_episode,
    summarize,
    write_csv,
)
from eval_force_impedance_three_way_wave import Ur5eWaveForceImpedanceEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="波浪地形上仅评估 PPO 力阻抗模型")
    p.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "runs" / "ppo_force_impedance" / "ppo_force_impedance_final.zip"),
        help="PPO .zip 路径",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "runs" / "ppo_force_impedance_wave_eval"),
    )
    p.add_argument(
        "--scene_xml",
        type=str,
        default=str(ROOT / "model" / "universal_robots_ur5e" / "scene_45.xml"),
    )
    p.add_argument("--episode_steps", type=int, default=1500)
    p.add_argument("--line_speed_y", type=float, default=-0.01)
    p.add_argument("--line_total_disp", type=float, default=0.30)
    p.add_argument("--target_force_z", type=float, default=-50.0)
    return p.parse_args()


def make_env_wave(args: argparse.Namespace, seed: int, render: bool) -> Ur5eWaveForceImpedanceEnv:
    return Ur5eWaveForceImpedanceEnv(
        scene_xml=args.scene_xml,
        episode_steps=args.episode_steps,
        line_speed_y=args.line_speed_y,
        line_total_disp=args.line_total_disp,
        base_target_force_z=args.target_force_z,
        render=render,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    if not str(args.model).strip():
        raise ValueError("请指定 --model 指向 PPO .zip")
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(model_path))
    policy_env = make_env_wave(args, args.seed + 1000, args.render)
    all_rows: list[dict[str, float | int | str]] = []
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

    csv_path = out_dir / "ppo_force_impedance_wave_eval.csv"
    png_path = out_dir / "ppo_force_tracking_wave_episode0.png"
    write_csv(csv_path, all_rows)
    plot_force_comparison(png_path, all_rows, episode=args.seed)

    stats = summarize(all_rows)
    print(
        f"[ppo_wave_only] model={model_path} episodes={args.episodes} scene={args.scene_xml}\n"
        f"  rms_force_error={stats['rms_force_error']:.3f} "
        f"max_abs_force_error={stats['max_abs_force_error']:.3f} "
        f"mean_reward={stats['mean_reward']:.3f} final_progress={stats['final_progress']:.3f}"
    )
    print(f"[ppo_wave_only] wrote {csv_path}")
    print(f"[ppo_wave_only] wrote {png_path}")


if __name__ == "__main__":
    main()
