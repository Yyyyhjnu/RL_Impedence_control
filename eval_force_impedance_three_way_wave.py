"""
三方法力跟踪评估：波浪地形（scene_45.xml + hfield 注入逻辑与 adapted_control_wave.py 一致）。

在 Ur5eForceImpedanceEnv 加载 scene_45 后，将单向 Y 方向 sin 波浪写入首个 hfield，
再执行 reset，保证与 MuJoCo 中 wave 脚本一致的地形形状。

方法：
  fixed_admittance / ppo_impedance / adaptive_sigma_phi
  与 eval_force_impedance_three_way.py 相同；adaptive 实现复用该模块中的 AdaptiveSigmaPhiUr5eEnv。

Examples:
  python eval_force_impedance_three_way_wave.py --model runs/ppo_force_impedance/ppo_force_impedance_final.zip
  python eval_force_impedance_three_way_wave.py --methods fixed,adaptive
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public"))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public" / "stable-baselines3-acmpc"))
sys.path.insert(0, str(ROOT / "ac_mpc_428" / "acmpc_public" / "training_modules"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from eval_force_impedance_three_way import (  # noqa: E402
    AdaptiveSigmaPhiUr5eEnv,
    EvaluationInterrupted,
    fixed_action_for,
    plot_impedance_params,
    plot_force_comparison,
    print_impedance_delta_summary,
    run_episode,
    summarize,
    write_impedance_params_csv,
    write_csv,
    zero_z_ref_action,
)
from ur5e_force_impedance_env import Ur5eForceImpedanceEnv  # noqa: E402


def inject_wave_hfield_data(model: mujoco.MjModel) -> None:
    """
    与 adapted_control_wave.py 中 inject_hfield_data 相同：
    Z 仅随 Y 变化，sin(Y)/2+0.5，写入 hfield_id=0。
    """
    if int(model.nhfield) <= 0:
        raise RuntimeError("当前模型无 hfield，请使用 scene_45.xml 等含 hfield 的场景。")
    hfield_id = 0
    adr = int(model.hfield_adr[hfield_id])
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])

    y_wave_frequency = 3 * np.pi
    y = np.linspace(0, y_wave_frequency, ncol)
    x = np.linspace(0, 1, nrow)
    _X, Y = np.meshgrid(x, y)
    Z = (np.sin(Y)) / 2.0 + 0.5
    model.hfield_data[adr : adr + nrow * ncol] = Z.flatten()


class Ur5eWaveForceImpedanceEnv(Ur5eForceImpedanceEnv):
    """在父类加载模型后注入波浪 hfield，再 reset（避免首次 reset 仍用 XML 默认高度场）。"""

    def __init__(self, **kwargs: Any) -> None:
        seed = kwargs.get("seed")
        if seed is not None:
            kwargs = dict(kwargs)
            kwargs["seed"] = None
        super().__init__(**kwargs)
        inject_wave_hfield_data(self.model)
        mujoco.mj_forward(self.model, self.data)
        if seed is not None:
            self.reset(seed=seed)


class Ur5eWaveAdaptiveSigmaPhiEnv(AdaptiveSigmaPhiUr5eEnv):
    """波浪地形上的 adaptive_sigma_phi。"""

    def __init__(self, **kwargs: Any) -> None:
        seed = kwargs.get("seed")
        if seed is not None:
            kwargs = dict(kwargs)
            kwargs["seed"] = None
        super().__init__(**kwargs)
        inject_wave_hfield_data(self.model)
        mujoco.mj_forward(self.model, self.data)
        if seed is not None:
            self.reset(seed=seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="三方法力跟踪评估：scene_45 波浪地形（hfield 注入同 adapted_control_wave.py）。"
    )
    p.add_argument(
        "--methods",
        type=str,
        default="fixed,ppo,adaptive",
        help="Comma list: fixed, ppo, adaptive.",
    )
    p.add_argument("--model", type=str, default="", help="PPO .zip（含 ppo 方法时必填）。")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "runs" / "force_impedance_three_way_wave_eval"),
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
    p.add_argument("--baseline_b_z", type=float, default=1000.0)
    p.add_argument("--baseline_k_z", type=float, default=0.0)
    p.add_argument("--ac_k1", type=float, default=5.0)
    p.add_argument("--ac_k2", type=float, default=3.0)
    p.add_argument("--ac_c0", type=float, default=-8.0)
    p.add_argument("--ac_md", type=float, default=1.0)
    p.add_argument("--ac_bd", type=float, default=1000.0)
    p.add_argument("--ac_phi_clip", type=float, default=0.1)
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


def make_adaptive_wave(args: argparse.Namespace, seed: int, render: bool) -> Ur5eWaveAdaptiveSigmaPhiEnv:
    return Ur5eWaveAdaptiveSigmaPhiEnv(
        scene_xml=args.scene_xml,
        episode_steps=args.episode_steps,
        line_speed_y=args.line_speed_y,
        line_total_disp=args.line_total_disp,
        base_target_force_z=args.target_force_z,
        adaptive_k1=args.ac_k1,
        adaptive_k2=args.ac_k2,
        adaptive_c0=args.ac_c0,
        adaptive_md=args.ac_md,
        adaptive_bd=args.ac_bd,
        adaptive_phi_clip=args.ac_phi_clip,
        render=render,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
    valid = {"fixed", "ppo", "adaptive"}
    unknown = methods - valid
    if unknown:
        raise ValueError(f"Unknown --methods: {unknown}; allowed: {valid}")
    if not methods:
        raise ValueError("--methods is empty")
    if "ppo" in methods and not str(args.model).strip():
        raise ValueError("选择 ppo 时需提供 --model 指向 .zip。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | int | str]] = []

    try:
        if "fixed" in methods:
            baseline_env = make_env_wave(args, args.seed, args.render)
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

        if "ppo" in methods:
            model = PPO.load(args.model)
            policy_env = make_env_wave(args, args.seed + 1000, args.render)
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

        if "adaptive" in methods:
            adaptive_env = make_adaptive_wave(args, args.seed + 2000, args.render)
            adaptive_action = zero_z_ref_action()
            for ep in range(int(args.episodes)):
                all_rows.extend(
                    run_episode(
                        adaptive_env,
                        policy=None,
                        label="adaptive_sigma_phi",
                        episode_idx=args.seed + ep,
                        baseline_action=adaptive_action,
                    )
                )
            adaptive_env.close()
    except EvaluationInterrupted as exc:
        all_rows.extend(exc.rows)
        print(f"[three_way_wave] interrupted; saving {len(all_rows)} collected rows.")

    csv_path = out_dir / "force_impedance_three_way_wave_eval.csv"
    png_path = out_dir / "force_tracking_three_way_wave_episode0.png"
    impedance_csv_path = out_dir / "impedance_params_three_way_wave_eval.csv"
    impedance_png_path = out_dir / "impedance_params_three_way_wave_episode0.png"
    write_csv(csv_path, all_rows)
    plot_force_comparison(png_path, all_rows, episode=args.seed)
    write_impedance_params_csv(impedance_csv_path, all_rows)
    plot_impedance_params(impedance_png_path, all_rows, episode=args.seed)

    print("[three_way_wave] summary (all episodes per label)")
    for label in sorted({str(r["label"]) for r in all_rows}):
        label_rows = [r for r in all_rows if str(r["label"]) == label]
        stats = summarize(label_rows)
        print(
            f"  [{label}] rms_force_error={stats['rms_force_error']:.3f} "
            f"max_abs_force_error={stats['max_abs_force_error']:.3f} "
            f"mean_reward={stats['mean_reward']:.3f} final_progress={stats['final_progress']:.3f}"
        )
    print_impedance_delta_summary(all_rows)
    print(f"[three_way_wave] wrote {csv_path}")
    print(f"[three_way_wave] wrote {png_path}")
    print(f"[three_way_wave] wrote {impedance_csv_path}")
    print(f"[three_way_wave] wrote {impedance_png_path}")


if __name__ == "__main__":
    main()
