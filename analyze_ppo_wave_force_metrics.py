"""
从 ppo_force_impedance_wave_eval（或任意同格式 CSV）计算力跟踪的
超调量与调节时间（按 step 索引）。

定义（目标力 F_ref = target_force_z，通常为负；误差 e = F_z - F_ref）：

1) 接触起点 i_c：首个满足 |F_z| >= contact_threshold 的 step；若无则 i_c = 0。

2) 超调（向「更负」——接触力超过目标幅值）：
   σ_minus% = max(0, F_ref - min_{k>=i_c} F_z) / max(|F_ref|, eps) * 100
   仅当 F_ref < 0 且 min F_z < F_ref 时非零。

3) 超调（向「较不负」——力不足相对目标的峰值）：
   σ_plus% = max(0, max_{k>=i_c} F_z - F_ref) / max(|F_ref|, eps) * 100
   仅当 max F_z > F_ref 时非零。

4) 调节时间 t_s（step）：从 i_c 起，首个 step t，使得对所有 k∈[t, T-1] 均有 |e(k)|<=δ；
   δ = max(ratio*|F_ref|, abs_tol)。若全程不满足则 nan。

若 CSV 含多种 label，按 label + episode 分别统计。

Example:
  python analyze_ppo_wave_force_metrics.py
  python analyze_ppo_wave_force_metrics.py --csv runs/force_impedance_three_way_wave_eval/force_impedance_three_way_wave_eval.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def settling_time_persistent(err: np.ndarray, delta: float, i_start: int) -> float:
    """First t >= i_start such that |err[k]|<=delta for all k in [t, n-1]."""
    n = len(err)
    if i_start >= n:
        return float("nan")
    for t in range(i_start, n):
        ok = True
        for k in range(t, n):
            if abs(float(err[k])) > delta:
                ok = False
                break
        if ok:
            return float(t)
    return float("nan")


def metrics_one_episode(
    F: np.ndarray,
    F_ref_scalar: float,
    *,
    contact_threshold: float,
    delta_ratio: float,
    abs_tol: float,
) -> dict[str, float]:
    F_ref = float(F_ref_scalar)
    eps = 1e-9
    mag = max(abs(F_ref), eps)

    ic = 0
    for i in range(len(F)):
        if abs(float(F[i])) >= contact_threshold:
            ic = i
            break

    F_seg = F[ic:]
    f_min = float(np.min(F_seg))
    f_max = float(np.max(F_seg))

    sigma_minus_pct = 0.0
    if F_ref < 0.0 and f_min < F_ref - eps:
        sigma_minus_pct = 100.0 * max(0.0, F_ref - f_min) / mag

    sigma_plus_pct = 0.0
    if f_max > F_ref + eps:
        sigma_plus_pct = 100.0 * max(0.0, f_max - F_ref) / mag

    err = F - F_ref
    delta = max(delta_ratio * mag, abs_tol)
    ts = settling_time_persistent(err.astype(np.float64), delta, ic)

    return {
        "contact_step": float(ic),
        "sigma_minus_pct": float(sigma_minus_pct),
        "sigma_plus_pct": float(sigma_plus_pct),
        "settling_step": float(ts),
        "delta_N": float(delta),
        "f_min": f_min,
        "f_max": f_max,
    }


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="力跟踪：超调量与调节时间（step）")
    p.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "runs" / "ppo_force_impedance_wave_eval" / "ppo_force_impedance_wave_eval.csv"),
    )
    p.add_argument("--contact_threshold", type=float, default=5.0, help="|Fz| 达到该值视为开始接触段")
    p.add_argument("--delta_ratio", type=float, default=0.05, help="误差带 δ = max(ratio*|Fref|, abs_tol)")
    p.add_argument("--abs_tol", type=float, default=1.0, help="误差带绝对下限 (N)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    rows = load_csv_rows(csv_path)
    by_key: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for r in rows:
        lab = str(r.get("label", "default"))
        ep = int(r["episode"])
        st = int(r["step"])
        fz = float(r["force_z"])
        tr = float(r["target_force_z"])
        by_key[(lab, ep)].append((st, fz, tr))

    out_rows: list[dict[str, float | int | str]] = []
    for (lab, ep), items in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        items.sort(key=lambda x: x[0])
        steps = np.array([t[0] for t in items], dtype=np.int32)
        forces = np.array([t[1] for t in items], dtype=np.float64)
        targets = np.array([t[2] for t in items], dtype=np.float64)
        if len(targets) == 0:
            continue
        if not np.allclose(targets, targets[0], rtol=0, atol=1e-3):
            raise ValueError(f"episode {lab}/{ep} 内 target_force_z 不一致，无法定义单参考调节时间")
        m = metrics_one_episode(
            forces,
            float(targets[0]),
            contact_threshold=args.contact_threshold,
            delta_ratio=args.delta_ratio,
            abs_tol=args.abs_tol,
        )
        out_rows.append(
            {
                "label": lab,
                "episode": ep,
                "n_steps": len(forces),
                "contact_step": int(m["contact_step"]),
                "sigma_minus_pct": round(m["sigma_minus_pct"], 4),
                "sigma_plus_pct": round(m["sigma_plus_pct"], 4),
                "settling_step": m["settling_step"],
                "delta_band_N": round(m["delta_N"], 4),
                "f_min": round(m["f_min"], 4),
                "f_max": round(m["f_max"], 4),
            }
        )

    out_csv = csv_path.parent / "wave_force_overshoot_settling.csv"
    if out_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

    print(f"[metrics] 输入: {csv_path}")
    print(f"[metrics] 误差带 δ = max({args.delta_ratio}*|F_ref|, {args.abs_tol}) N；调节时间=自接触步起全程落入 |e|<=δ 的首个 step（无则 nan）")
    print()
    for r in out_rows:
        ts = r["settling_step"]
        ts_str = "nan" if (isinstance(ts, float) and np.isnan(ts)) else str(int(ts))
        print(
            f"  [{r['label']}] ep={r['episode']}  "
            f"σ_过猛%={r['sigma_minus_pct']:.2f}  σ_不足%={r['sigma_plus_pct']:.2f}  "
            f"接触@step={r['contact_step']}  调节时间@step={ts_str}  "
            f"f∈[{r['f_min']:.1f},{r['f_max']:.1f}]N"
        )

    if out_rows:
        sm = np.nanmean([r["sigma_minus_pct"] for r in out_rows])
        sp = np.nanmean([r["sigma_plus_pct"] for r in out_rows])
        sts = [r["settling_step"] for r in out_rows]
        sts_valid = [s for s in sts if not (isinstance(s, float) and np.isnan(s))]
        print()
        print(f"  各回合平均: σ_过猛%={sm:.2f}  σ_不足%={sp:.2f}")
        if sts_valid:
            print(f"  调节时间(有解回合) 均值 step={float(np.mean(sts_valid)):.1f}  min={min(sts_valid):.0f}  max={max(sts_valid):.0f}")
        else:
            print("  调节时间: 无回合满足「自接触后全程落入误差带」")
        print(f"[metrics] 已写: {out_csv}")


if __name__ == "__main__":
    main()
