"""
根据评估 CSV 汇总「平面 / 斜面 / 正弦面」下 PPO（或其它 label）力跟踪指标：

  - 稳定时间 (s)：接触后优先「尾段误差上界≤δ」的首个 step×dt；否则持久落入 δ；
    再否则「首次 |e|≤δ」×dt。δ=max(0.05|Fref|, abs_tol)。
  - 超调 (%)：max(σ_过猛%, σ_不足%)（与 analyze_ppo_wave_force_metrics.py 一致）。
  - 力误差：接触后 RMS / MAE / Max|e|，各回合平均。

默认：
  平面   → runs/force_impedance_eval/force_impedance_eval.csv（含 ppo_impedance，scene-30）
  正弦面 → runs/ppo_force_impedance_wave_eval/ppo_force_impedance_wave_eval.csv
  斜面   → 无默认；--csv_slope 指定否则为 —

dt 默认 0.005 s（ur5e_2.xml option timestep）。

输出 runs/terrain_force_metrics_table.md 与 .csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def settling_time_persistent(err: np.ndarray, delta: float, i_start: int) -> float:
    n = len(err)
    if i_start >= n:
        return float("nan")
    for t in range(i_start, n):
        if bool(np.all(np.abs(err[t:]) <= delta + 1e-9)):
            return float(t)
    return float("nan")


def suffix_settle_step(err: np.ndarray, delta: float, i_start: int) -> float:
    n = len(err)
    for t in range(i_start, n):
        if float(np.max(np.abs(err[t:]))) <= delta + 1e-9:
            return float(t)
    return float("nan")


def first_in_band_step(err: np.ndarray, delta: float, i_start: int) -> float:
    for t in range(i_start, len(err)):
        if abs(float(err[t])) <= delta + 1e-9:
            return float(t)
    return float("nan")


def contact_start(F: np.ndarray, thr: float) -> int:
    for i in range(len(F)):
        if abs(float(F[i])) >= thr:
            return i
    return 0


def episode_metrics(
    F: np.ndarray,
    F_ref: float,
    *,
    contact_threshold: float,
    delta_ratio: float,
    abs_tol: float,
) -> dict[str, float]:
    F_ref = float(F_ref)
    eps = 1e-9
    mag = max(abs(F_ref), eps)
    ic = contact_start(F, contact_threshold)
    F_seg = F[ic:]
    f_min, f_max = float(np.min(F_seg)), float(np.max(F_seg))

    sigma_minus = 0.0
    if F_ref < 0.0 and f_min < F_ref - eps:
        sigma_minus = 100.0 * max(0.0, F_ref - f_min) / mag
    sigma_plus = 0.0
    if f_max > F_ref + eps:
        sigma_plus = 100.0 * max(0.0, f_max - F_ref) / mag

    err = (F - F_ref).astype(np.float64)
    delta = max(delta_ratio * mag, abs_tol)

    ts_persist = settling_time_persistent(err, delta, ic)
    ts_suffix = suffix_settle_step(err, delta, ic)
    ts_first = first_in_band_step(err, delta, ic)

    ts_step = ts_suffix
    if np.isnan(ts_step):
        ts_step = ts_persist
    if np.isnan(ts_step):
        ts_step = ts_first

    e_post = err[ic:]
    rms = float(np.sqrt(np.mean(e_post**2))) if len(e_post) else float("nan")
    mae = float(np.mean(np.abs(e_post))) if len(e_post) else float("nan")
    emax = float(np.max(np.abs(e_post))) if len(e_post) else float("nan")

    return {
        "overshoot_pct": max(sigma_minus, sigma_plus),
        "stable_step": float(ts_step),
        "rms_N": rms,
        "mae_N": mae,
        "max_abs_err_N": emax,
    }


def load_grouped(path: Path, label: str) -> dict[int, tuple[np.ndarray, float]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    buckets: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for r in rows:
        if str(r.get("label", "")) != label:
            continue
        ep = int(r["episode"])
        st = int(r["step"])
        fz = float(r["force_z"])
        tr = float(r["target_force_z"])
        buckets[ep].append((st, fz, tr))
    out: dict[int, tuple[np.ndarray, float]] = {}
    for ep, items in buckets.items():
        items.sort(key=lambda x: x[0])
        F = np.array([x[1] for x in items], dtype=np.float64)
        targets = np.array([x[2] for x in items], dtype=np.float64)
        if len(targets) == 0:
            continue
        if not np.allclose(targets, targets[0], rtol=0, atol=1e-3):
            raise ValueError(f"{path} episode {ep}: target_force_z 不一致")
        out[ep] = (F, float(targets[0]))
    return out


def mean_agg(
    ep_dict: dict[int, tuple[np.ndarray, float]],
    *,
    dt: float,
    contact_threshold: float,
    delta_ratio: float,
    abs_tol: float,
) -> dict[str, float]:
    if not ep_dict:
        return {
            "stable_time_s": float("nan"),
            "overshoot_pct": float("nan"),
            "rms_N": float("nan"),
            "mae_N": float("nan"),
            "max_abs_err_N": float("nan"),
        }
    mkw = dict(contact_threshold=contact_threshold, delta_ratio=delta_ratio, abs_tol=abs_tol)
    sts: list[float] = []
    ovs: list[float] = []
    rms: list[float] = []
    maes: list[float] = []
    emaxs: list[float] = []
    for _ep, (F, Fref) in sorted(ep_dict.items()):
        m = episode_metrics(F, Fref, **mkw)
        sts.append(m["stable_step"] * dt)
        ovs.append(m["overshoot_pct"])
        rms.append(m["rms_N"])
        maes.append(m["mae_N"])
        emaxs.append(m["max_abs_err_N"])
    return {
        "stable_time_s": float(np.nanmean(sts)),
        "overshoot_pct": float(np.nanmean(ovs)),
        "rms_N": float(np.nanmean(rms)),
        "mae_N": float(np.nanmean(maes)),
        "max_abs_err_N": float(np.nanmean(emaxs)),
    }


def fmt(x: float, nd: int = 3) -> str:
    if x != x:
        return "—"
    return f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="平面/斜面/正弦面 力跟踪指标表")
    p.add_argument(
        "--csv_plane",
        type=str,
        default=str(ROOT / "runs" / "force_impedance_eval" / "force_impedance_eval.csv"),
        help="平面/scene-30 类评估 CSV（需含 label=ppo_impedance）",
    )
    p.add_argument("--csv_slope", type=str, default="", help="斜面专用 CSV；空则无斜面列数据")
    p.add_argument(
        "--csv_sine",
        type=str,
        default=str(ROOT / "runs" / "ppo_force_impedance_wave_eval" / "ppo_force_impedance_wave_eval.csv"),
    )
    p.add_argument("--label", type=str, default="ppo_impedance")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--contact_threshold", type=float, default=5.0)
    p.add_argument("--delta_ratio", type=float, default=0.05)
    p.add_argument("--abs_tol", type=float, default=1.0)
    args = p.parse_args()

    plane_path = Path(args.csv_plane)
    sine_path = Path(args.csv_sine)
    slope_path = Path(args.csv_slope) if str(args.csv_slope).strip() else None

    plane_d = load_grouped(plane_path, args.label) if plane_path.is_file() else {}
    sine_d = load_grouped(sine_path, args.label) if sine_path.is_file() else {}
    slope_d = load_grouped(slope_path, args.label) if slope_path and slope_path.is_file() else {}

    mkw = dict(
        dt=args.dt,
        contact_threshold=args.contact_threshold,
        delta_ratio=args.delta_ratio,
        abs_tol=args.abs_tol,
    )
    col_plane = mean_agg(plane_d, **mkw)
    col_sine = mean_agg(sine_d, **mkw)
    col_slope = mean_agg(slope_d, **mkw)

    out_dir = ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "terrain_force_metrics_table.md"
    csv_path = out_dir / "terrain_force_metrics_table.csv"

    lines_md = [
        "# 平面 / 斜面 / 正弦面 力跟踪指标（由 CSV 统计）",
        "",
        f"- 标签: `{args.label}`，仿真步长 dt={args.dt} s",
        f"- 平面 CSV: `{plane_path}`（存在: {plane_path.is_file()}）",
        f"- 正弦面 CSV: `{sine_path}`（存在: {sine_path.is_file()}）",
        f"- 斜面 CSV: `{slope_path or '（未指定）'}`（存在: {bool(slope_path and slope_path.is_file())}）",
        f"- 接触阈值 {args.contact_threshold} N；δ = max({args.delta_ratio}×|F_ref|, {args.abs_tol}) N",
        "",
        "**说明**：`scene-30` 场景中同时含水平与斜面几何体；下表「平面」列对应上述平面默认 CSV（整段扫描任务），并非 `scene_pingmian` 纯平面。斜面列需单独跑仅斜面场景并导出同格式 CSV 后传入 `--csv_slope`。",
        "",
        "## 主表（各回合平均）",
        "",
        "| 指标 | 平面 | 斜面 | 正弦面 |",
        "|------|------|------|--------|",
        f"| 稳定时间 (s) | {fmt(col_plane['stable_time_s'])} | {fmt(col_slope['stable_time_s'])} | {fmt(col_sine['stable_time_s'])} |",
        f"| 超调 (%) | {fmt(col_plane['overshoot_pct'])} | {fmt(col_slope['overshoot_pct'])} | {fmt(col_sine['overshoot_pct'])} |",
        f"| 力误差 RMS (N) | {fmt(col_plane['rms_N'])} | {fmt(col_slope['rms_N'])} | {fmt(col_sine['rms_N'])} |",
        f"| 力误差 MAE (N) | {fmt(col_plane['mae_N'])} | {fmt(col_slope['mae_N'])} | {fmt(col_sine['mae_N'])} |",
        f"| 力误差 Max|e| (N) | {fmt(col_plane['max_abs_err_N'])} | {fmt(col_slope['max_abs_err_N'])} | {fmt(col_sine['max_abs_err_N'])} |",
        "",
    ]
    md_path.write_text("\n".join(lines_md), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "plane", "slope", "sine"])
        w.writerow(["stable_time_s", col_plane["stable_time_s"], col_slope["stable_time_s"], col_sine["stable_time_s"]])
        w.writerow(["overshoot_pct", col_plane["overshoot_pct"], col_slope["overshoot_pct"], col_sine["overshoot_pct"]])
        w.writerow(["rms_force_error_N", col_plane["rms_N"], col_slope["rms_N"], col_sine["rms_N"]])
        w.writerow(["mae_force_error_N", col_plane["mae_N"], col_slope["mae_N"], col_sine["mae_N"]])
        w.writerow(["max_abs_force_error_N", col_plane["max_abs_err_N"], col_slope["max_abs_err_N"], col_sine["max_abs_err_N"]])

    print(f"[terrain_metrics] wrote {md_path}")
    print(f"[terrain_metrics] wrote {csv_path}")


if __name__ == "__main__":
    main()
