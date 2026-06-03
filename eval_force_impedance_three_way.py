"""
Compare three force-tracking controllers on the same Ur5eForceImpedanceEnv:

1) fixed_admittance  — constant Bz/Kz (same as eval_force_impedance.py)
2) ppo_impedance     — SB3 PPO policy (.zip)
3) adaptive_sigma_phi — Z-axis adaptive admittance from adapted_control_45.py
   (sigma(eta), phi integral, md/bd dynamics) embedded in a Gym env subclass
   so physics / IK / scanning match the training environment.

Examples:
  python eval_force_impedance_three_way.py \\
    --model runs/ppo_force_impedance/ppo_force_impedance_final.zip

  python eval_force_impedance_three_way.py --methods fixed,adaptive

  python eval_force_impedance_three_way.py --model path/to.zip --render
"""
from __future__ import annotations

import argparse
import csv
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

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402 — used by AdaptiveSigmaPhiUr5eEnv.step (subclass body lives in this module)
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from ur5e_force_impedance_env import Ur5eForceImpedanceEnv  # noqa: E402


class EvaluationInterrupted(Exception):
    """Carry partially collected rows so Ctrl+C can still save evaluation data."""

    def __init__(self, rows: list[dict[str, float | int | str]]):
        super().__init__("evaluation interrupted")
        self.rows = rows


class AdaptiveSigmaPhiUr5eEnv(Ur5eForceImpedanceEnv):
    """
    Same task as Ur5eForceImpedanceEnv, but Z-axis admittance follows
    adapted_control_45.py (sigma from contact force, phi integral, dd_z from md/bd).
    Action still supplies z_ref_delta via channel 2; Bz/Kz channels are ignored for Z dynamics.
    """

    def __init__(
        self,
        *,
        adaptive_k1: float = 5.0,
        adaptive_k2: float = 3.0,
        adaptive_c0: float = -8.0,
        adaptive_md: float = 1.0,
        adaptive_bd: float = 1000.0,
        adaptive_phi_clip: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self._ac_k1 = float(adaptive_k1)
        self._ac_k2 = float(adaptive_k2)
        self._ac_c0 = float(adaptive_c0)
        self._ac_md = float(adaptive_md)
        self._ac_bd = float(adaptive_bd)
        self._ac_phi_clip = float(adaptive_phi_clip)
        self._ac_phi = 0.0
        self._ac_sigma = 0.0
        super().__init__(**kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._ac_phi = 0.0
        self._ac_sigma = 0.0
        return obs, info

    def step(self, action: np.ndarray):  # noqa: C901 — mirrors base env step with Z branch replaced
        action_clipped = np.clip(
            np.asarray(action, dtype=np.float64).reshape(self.ACTION_DIM), -1.0, 1.0
        )
        _, _, z_ref_delta = self._map_action(action_clipped)
        b_z = self._ac_bd
        k_z = 0.0
        f_target_z = self.base_target_force_z

        mujoco.mj_forward(self.model, self.data)
        curr_tf = self._get_ee_pose_matrix()
        now_pos = curr_tf[:3, 3]
        now_rot = R.from_matrix(curr_tf[:3, :3])

        v_pos = (now_pos - self.last_ee_pos) / self.dt
        diff_rot_obj = now_rot * self.last_ee_rot.inv()
        v_rot = diff_rot_obj.as_rotvec() / self.dt
        ee_vel = np.concatenate([v_pos, v_rot])
        ee_vel_filter = self.vel_filter.update(ee_vel)
        self.last_ee_pos = now_pos.copy()
        self.last_ee_rot = now_rot

        force = self._read_force()
        err_pos = now_pos - self.desired_pos
        err_rot = (now_rot * self.desired_rot.inv()).as_rotvec()
        ee_pos_err = np.concatenate([err_pos, err_rot])

        fz_pre = float(force[2])
        fd = float(f_target_z)
        eta = fz_pre
        exponent = self._ac_k1 * (eta - self._ac_c0)
        sigma = 1.0 / (np.exp(exponent) + self._ac_k2)
        sigma_max = (self._ac_bd * self.dt) / (self._ac_md + self._ac_bd * self.dt + 1e-12)
        sigma = float(np.clip(sigma, 5e-2, sigma_max))
        self._ac_sigma = sigma

        current_force_error = fz_pre - fd
        self._ac_phi += sigma * current_force_error / self._ac_bd
        self._ac_phi = float(np.clip(self._ac_phi, -self._ac_phi_clip, self._ac_phi_clip))

        F_error_z = current_force_error + self._ac_bd * self._ac_phi
        dd_ee_z = (F_error_z - self._ac_bd * ee_vel_filter[2]) / self._ac_md

        dd_ee = np.zeros(6, dtype=np.float64)
        dd_ee[0] = -self.fixed_xy_damping * ee_vel_filter[0] / self.m_z
        dd_ee[2] = -dd_ee_z

        self.admittance_vel = np.clip(
            dd_ee * self.dt,
            -self.admittance_velocity_limit,
            self.admittance_velocity_limit,
        )
        delta_step_admittance = self.admittance_vel * self.dt
        delta_step_nominal = np.zeros(3, dtype=np.float64)
        delta_step_nominal[1] = self.line_speed_y * self.dt

        self.desired_pos += delta_step_nominal + delta_step_admittance[:3]
        delta_rot_step = R.from_rotvec(delta_step_admittance[3:])
        self.desired_rot = delta_rot_step * self.desired_rot

        target_tf = np.eye(4, dtype=np.float64)
        target_tf[:3, :3] = self.desired_rot.as_matrix()
        target_tf[:3, 3] = self.desired_pos

        ik_success = True
        try:
            dof, info_ik = self._kin.ik(target_tf, current_arm_motor_q=self.last_q_des)
            q_des = np.asarray(dof[:6], dtype=np.float64).reshape(6)
            ik_success = bool(info_ik.get("success", True))
        except Exception:
            q_des = self.last_q_des.copy()
            ik_success = False

        if ik_success and np.all(np.isfinite(q_des)):
            q_des = np.clip(q_des, self._jnt_lo, self._jnt_hi)
            self.last_q_des = q_des.copy()
            self.q_ref = q_des.copy()
        self.data.ctrl[: self.POS_ACT_DIM] = self.q_ref

        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        force_after = self._read_force()
        obs = self._build_obs(force_after, ee_vel_filter, b_z, k_z, f_target_z)

        force_error = f_target_z - float(force_after[2])
        force_pen = 0.02 * abs(force_error)
        desired_y = self.start_pos[1] + np.sign(self.line_speed_y) * self.line_total_disp
        y_track_pen = 2.0 * abs(float(self.desired_pos[1] - self.data.body(self.ee_body_name).xpos[1]))
        action_smooth_pen = 0.02 * float(np.linalg.norm(action_clipped - self.last_action))
        vel_pen = 0.01 * float(np.linalg.norm(ee_vel_filter[:3]))
        reward = -force_pen - y_track_pen - action_smooth_pen - vel_pen

        progress = self._progress(float(self.data.body(self.ee_body_name).xpos[1]))
        in_contact_by_force = abs(float(force_after[2])) >= self.contact_force_threshold
        if in_contact_by_force:
            reward += 0.25 * progress
        elif progress > 0.05:
            reward -= 0.5

        terminated = False
        truncated = False
        done_reason = None
        if abs(float(force_after[2])) > self.max_force_abs:
            reward -= 5.0
            terminated = True
            done_reason = "force_limit"
        if not ik_success:
            reward -= 1.0
        if progress >= 1.0 or (
            self.line_speed_y < 0.0 and self.desired_pos[1] <= desired_y
        ) or (
            self.line_speed_y > 0.0 and self.desired_pos[1] >= desired_y
        ):
            if in_contact_by_force and abs(force_error) <= self.force_success_tolerance:
                reward += 3.0
            else:
                reward -= 3.0
            terminated = True
            done_reason = "progress_or_desired_y"

        q_meas = np.asarray(self.data.qpos[:6], dtype=np.float64)
        q_dot = np.asarray(self.data.qvel[:6], dtype=np.float64)
        out_of_range = bool(
            np.any(q_meas < self._jnt_lo - 1e-3) or np.any(q_meas > self._jnt_hi + 1e-3)
        )
        nan_state = bool(not np.all(np.isfinite(q_meas)) or not np.all(np.isfinite(q_dot)))
        if out_of_range or nan_state:
            reward -= 5.0
            terminated = True
            done_reason = "bad_state"

        self._step += 1
        if self._step >= self.episode_steps:
            truncated = True
            done_reason = "time_limit"

        self.last_action = action_clipped.copy()
        self.last_params = np.array([b_z, k_z, f_target_z], dtype=np.float64)

        info = {
            "force_z": float(force_after[2]),
            "target_force_z": float(f_target_z),
            "force_error_z": float(force_error),
            "b_z": float(b_z),
            "k_z": float(k_z),
            "z_ref_delta": float(z_ref_delta),
            "progress": float(progress),
            "reward": float(reward),
            "in_contact_by_force": bool(in_contact_by_force),
            "ik_success": bool(ik_success),
            "reward_force": float(-force_pen),
            "adaptive_phi": float(self._ac_phi),
            "adaptive_sigma": float(self._ac_sigma),
        }
        self._last_render_info = info.copy()

        if self._render_enabled:
            self.render()

        return obs, float(reward), bool(terminated), bool(truncated), info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare fixed admittance, PPO, and adapted_control_45-style adaptive Z admittance."
    )
    p.add_argument(
        "--methods",
        type=str,
        default="fixed,ppo,adaptive",
        help="Comma list: fixed, ppo, adaptive (order does not affect grouping in CSV).",
    )
    p.add_argument("--model", type=str, default="", help="PPO .zip path (required if ppo is in --methods).")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "runs" / "force_impedance_three_way_eval"))
    p.add_argument(
        "--scene_xml",
        type=str,
        default=str(ROOT / "model" / "universal_robots_ur5e" / "scene-30.xml"),
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


def make_adaptive_env(args: argparse.Namespace, seed: int, render: bool) -> AdaptiveSigmaPhiUr5eEnv:
    return AdaptiveSigmaPhiUr5eEnv(
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


def fixed_action_for(env: Ur5eForceImpedanceEnv, b_z: float, k_z: float) -> np.ndarray:
    b_min, b_max = env.b_z_range
    k_min, k_max = env.k_z_range
    b_norm = 2.0 * (float(b_z) - b_min) / max(b_max - b_min, 1e-9) - 1.0
    k_norm = 2.0 * (float(k_z) - k_min) / max(k_max - k_min, 1e-9) - 1.0
    return np.clip(np.array([b_norm, k_norm, 0.0], dtype=np.float32), -1.0, 1.0)


def zero_z_ref_action() -> np.ndarray:
    """Mid B/K from normalized 0,0 and zero z_ref_delta (used for adaptive controller)."""
    return np.zeros(3, dtype=np.float32)


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
    prev_b_z: Optional[float] = None
    prev_k_z: Optional[float] = None
    prev_z_ref_delta: Optional[float] = None
    try:
        while not done:
            if policy is None:
                assert baseline_action is not None
                action = baseline_action
            else:
                action, _ = policy.predict(obs, deterministic=True)
            action_arr = np.asarray(action, dtype=np.float64).reshape(env.ACTION_DIM)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            b_z = float(info.get("b_z", 0.0))
            k_z = float(info.get("k_z", 0.0))
            z_ref_delta = float(info.get("z_ref_delta", 0.0))
            delta_b_z = 0.0 if prev_b_z is None else b_z - prev_b_z
            delta_k_z = 0.0 if prev_k_z is None else k_z - prev_k_z
            delta_z_ref_delta = 0.0 if prev_z_ref_delta is None else z_ref_delta - prev_z_ref_delta
            rows.append(
                {
                    "label": label,
                    "episode": episode_idx,
                    "step": step,
                    "reward": float(reward),
                    "force_z": float(info.get("force_z", 0.0)),
                    "target_force_z": float(info.get("target_force_z", 0.0)),
                    "force_error_z": float(info.get("force_error_z", 0.0)),
                    "action_b_norm": float(action_arr[0]),
                    "action_k_norm": float(action_arr[1]),
                    "action_z_ref_norm": float(action_arr[2]),
                    "b_z": b_z,
                    "k_z": k_z,
                    "z_ref_delta": z_ref_delta,
                    "delta_b_z": float(delta_b_z),
                    "delta_k_z": float(delta_k_z),
                    "delta_z_ref_delta": float(delta_z_ref_delta),
                    "abs_delta_b_z": float(abs(delta_b_z)),
                    "abs_delta_k_z": float(abs(delta_k_z)),
                    "abs_delta_z_ref_delta": float(abs(delta_z_ref_delta)),
                    "progress": float(info.get("progress", 0.0)),
                    "ik_success": int(bool(info.get("ik_success", True))),
                    "adaptive_phi": float(info.get("adaptive_phi", 0.0)),
                    "adaptive_sigma": float(info.get("adaptive_sigma", 0.0)),
                }
            )
            prev_b_z = b_z
            prev_k_z = k_z
            prev_z_ref_delta = z_ref_delta
            step += 1
    except KeyboardInterrupt as exc:
        raise EvaluationInterrupted(rows) from exc
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


def plot_force_comparison(path: Path, rows: list[dict[str, float | int | str]], episode: int = 0) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({str(r["label"]) for r in rows})
    plt.figure(figsize=(10, 5))
    for label in labels:
        selected = [r for r in rows if str(r["label"]) == label and int(r["episode"]) == episode]
        if not selected:
            continue
        steps = np.array([int(r["step"]) for r in selected], dtype=np.int32)
        force = np.array([float(r["force_z"]) for r in selected], dtype=np.float64)
        target = np.array([float(r["target_force_z"]) for r in selected], dtype=np.float64)
        plt.plot(steps, force, label=f"{label}: Fz")
        plt.plot(steps, target, linestyle=":", alpha=0.35)
    plt.xlabel("step")
    plt.ylabel("force z (N)")
    plt.title(f"Force tracking (episode {episode})")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_impedance_params_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "episode",
        "step",
        "progress",
        "action_b_norm",
        "action_k_norm",
        "action_z_ref_norm",
        "b_z",
        "k_z",
        "z_ref_delta",
        "delta_b_z",
        "delta_k_z",
        "delta_z_ref_delta",
        "abs_delta_b_z",
        "abs_delta_k_z",
        "abs_delta_z_ref_delta",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_impedance_params(path: Path, rows: list[dict[str, float | int | str]], episode: int = 0) -> None:
    selected = [r for r in rows if int(r["episode"]) == episode]
    if not selected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({str(r["label"]) for r in selected})
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label in labels:
        label_rows = [r for r in selected if str(r["label"]) == label]
        steps = np.array([int(r["step"]) for r in label_rows], dtype=np.int32)
        b_z = np.array([float(r.get("b_z", 0.0)) for r in label_rows], dtype=np.float64)
        k_z = np.array([float(r.get("k_z", 0.0)) for r in label_rows], dtype=np.float64)
        z_ref_delta = np.array([float(r.get("z_ref_delta", 0.0)) for r in label_rows], dtype=np.float64)
        axes[0].plot(steps, b_z, label=label)
        axes[1].plot(steps, k_z, label=label)
        axes[2].plot(steps, z_ref_delta, label=label)
    axes[0].set_ylabel("B_z")
    axes[1].set_ylabel("K_z")
    axes[2].set_ylabel("z_ref_delta (m)")
    axes[2].set_xlabel("step")
    for ax in axes:
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Impedance parameters (episode {episode})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_impedance_delta_summary(rows: list[dict[str, float | int | str]]) -> None:
    for label in sorted({str(r["label"]) for r in rows}):
        label_rows = [r for r in rows if str(r["label"]) == label]
        if not label_rows:
            continue
        db = np.array([float(r.get("abs_delta_b_z", 0.0)) for r in label_rows], dtype=np.float64)
        dk = np.array([float(r.get("abs_delta_k_z", 0.0)) for r in label_rows], dtype=np.float64)
        dz = np.array([float(r.get("abs_delta_z_ref_delta", 0.0)) for r in label_rows], dtype=np.float64)
        print(
            f"  [{label}] impedance_delta "
            f"mean|dB|={np.mean(db):.6g} max|dB|={np.max(db):.6g} "
            f"mean|dK|={np.mean(dk):.6g} max|dK|={np.max(dk):.6g} "
            f"mean|dz_ref|={np.mean(dz):.6g} max|dz_ref|={np.max(dz):.6g}"
        )


def main() -> None:
    args = parse_args()
    methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
    valid = {"fixed", "ppo", "adaptive"}
    unknown = methods - valid
    if unknown:
        raise ValueError(f"Unknown --methods entries: {unknown}; allowed: {valid}")
    if not methods:
        raise ValueError("--methods is empty")

    if "ppo" in methods and not str(args.model).strip():
        raise ValueError("选择 ppo 时需提供 --model 指向 .zip 模型。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | int | str]] = []

    try:
        if "fixed" in methods:
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

        if "ppo" in methods:
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

        if "adaptive" in methods:
            adaptive_env = make_adaptive_env(args, args.seed + 2000, args.render)
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
        print(f"[three_way] interrupted; saving {len(all_rows)} collected rows.")

    csv_path = out_dir / "force_impedance_three_way_eval.csv"
    png_path = out_dir / "force_tracking_three_way_episode0.png"
    impedance_csv_path = out_dir / "impedance_params_three_way_eval.csv"
    impedance_png_path = out_dir / "impedance_params_three_way_episode0.png"
    write_csv(csv_path, all_rows)
    plot_force_comparison(png_path, all_rows, episode=args.seed)
    write_impedance_params_csv(impedance_csv_path, all_rows)
    plot_impedance_params(impedance_png_path, all_rows, episode=args.seed)

    print("[three_way] summary (all episodes per label)")
    for label in sorted({str(r["label"]) for r in all_rows}):
        label_rows = [r for r in all_rows if str(r["label"]) == label]
        stats = summarize(label_rows)
        print(
            f"  [{label}] rms_force_error={stats['rms_force_error']:.3f} "
            f"max_abs_force_error={stats['max_abs_force_error']:.3f} "
            f"mean_reward={stats['mean_reward']:.3f} final_progress={stats['final_progress']:.3f}"
        )
    print_impedance_delta_summary(all_rows)
    print(f"[three_way] wrote {csv_path}")
    print(f"[three_way] wrote {png_path}")
    print(f"[three_way] wrote {impedance_csv_path}")
    print(f"[three_way] wrote {impedance_png_path}")


if __name__ == "__main__":
    main()
