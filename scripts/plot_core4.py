#!/usr/bin/env python3
"""핵심 4개 모델 비교 그림 생성 (core-4 comparison figure).

다른 베이스라인(TransE, RotatE, DistMult, ComplEx, TKRL)은 제외하고,
온톨로지 결합형 직접 베이스 2종과 본 연구 손실 적용 결과 2종, 총 4개만 비교한다.

  (1) TransO (base)        : 직접 비교 대상(선행 연구)
  (2) TransC (base)        : 직접 비교 대상(선행 연구)
  (3) Ours (TransO base)   : 본 연구 손실(L_dir+L_hier_attn+L_attr+L_type)을 TransO 베이스에 적용
  (4) Ours (TransC base)   : 동일 손실을 TransC 점수 함수에 적용 (추가 튜닝 없음)

각 모델의 seed별 raw JSON(experiments/<run>/raw/main_<key>_s*.json)을 읽어
5-seed 평균과 표준편차로 막대그래프를 그린다. 재학습이 필요 없다.

사용법:
    python scripts/plot_core4.py
    python scripts/plot_core4.py --run-dir experiments/final_run
    python scripts/plot_core4.py --out figures/my_core4.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# 표시 라벨 -> raw 파일 접두어(main_<KEY>_s*.json) 매핑
MODELS = [
    ("TransO\n(base)", "TransO"),
    ("TransC\n(base)", "TransC"),
    ("Ours\n(TransO base)", "Ours"),
    ("Ours\n(TransC base)", "OursC"),
]
# 모델별 고정 색상(두 패널에서 동일 의미 유지)
COLORS = {
    "TransO": "#90A4AE",
    "TransC": "#BCAAA4",
    "Ours":   "#1F77B4",
    "OursC":  "#D62728",
}

# 패널 B에 표시할 metric (라벨, JSON 경로)
METRICS_B = [
    ("Hits@1",   ("overall", "hits@1")),
    ("Hits@10",  ("overall", "hits@10")),
    ("Dir H@1",  ("groups", "directional", "hits@1")),
    ("Hier MRR", ("groups", "hierarchy", "mrr")),
]


def _seed_values(run_dir, key, path):
    """raw/main_<key>_s*.json 들에서 path 경로의 metric 값을 seed별로 수집."""
    vals = []
    for fp in sorted(glob.glob(str(run_dir / "raw" / ("main_%s_s*.json" % key)))):
        cur = json.load(open(fp, encoding="utf-8"))
        for p in path:
            cur = cur[p]
        vals.append(float(cur))
    if not vals:
        raise FileNotFoundError("no raw files for model key '%s' in %s" % (key, run_dir / "raw"))
    return vals


def _mean_std(run_dir, key, path):
    vals = _seed_values(run_dir, key, path)
    return float(np.mean(vals)), float(np.std(vals))  # ddof=0 (repo 규약과 동일)


def make_figure(run_dir, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for _, k in MODELS]
    labels = [lab for lab, _ in MODELS]
    colors = [COLORS[k] for k in keys]
    n_seeds = len(_seed_values(run_dir, keys[0], ("overall", "mrr")))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.7))

    # 패널 A : Overall MRR (base -> +ours 향상 강조)
    mrr_mu = [_mean_std(run_dir, k, ("overall", "mrr"))[0] for k in keys]
    mrr_sd = [_mean_std(run_dir, k, ("overall", "mrr"))[1] for k in keys]
    order = [0, 2, 1, 3]                 # 시각 배치: TransO, Ours(TransO), TransC, Ours(TransC)
    xpos = [0.0, 1.0, 2.5, 3.5]
    for xp, idx in zip(xpos, order):
        axA.bar(xp, mrr_mu[idx], width=0.8, yerr=mrr_sd[idx], capsize=4,
                color=colors[idx], edgecolor="white")
        axA.text(xp, mrr_mu[idx] + mrr_sd[idx] + 0.012, "%.3f" % mrr_mu[idx],
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    # base -> ours 향상폭(%p): 동일 점수함수 쌍끼리 비교
    for li, ri in [(0, 1), (2, 3)]:
        b, o = order[li], order[ri]
        diff = (mrr_mu[o] - mrr_mu[b]) * 100
        ytop = max(mrr_mu[b], mrr_mu[o]) + 0.06
        axA.annotate("", xy=(xpos[ri], ytop), xytext=(xpos[li], ytop),
                     arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.6))
        axA.text((xpos[li] + xpos[ri]) / 2, ytop + 0.006, "+%.1f%%p" % diff,
                 ha="center", va="bottom", fontsize=10, color="#2E7D32", fontweight="bold")
    axA.set_xticks(xpos)
    axA.set_xticklabels([labels[i] for i in order], fontsize=9)
    axA.set_ylabel("Filtered MRR")
    axA.set_ylim(0, max(mrr_mu) + 0.16)
    axA.set_title("Overall MRR  -  base vs. + ours loss", fontsize=11, fontweight="bold")
    axA.grid(axis="y", alpha=0.3)

    # 패널 B : 핵심 지표 그룹 비교
    x = np.arange(len(METRICS_B))
    w = 0.2
    for i, k in enumerate(keys):
        mus = [_mean_std(run_dir, k, p)[0] for _, p in METRICS_B]
        sds = [_mean_std(run_dir, k, p)[1] for _, p in METRICS_B]
        axB.bar(x + (i - 1.5) * w, mus, w, yerr=sds, capsize=2,
                color=COLORS[k], edgecolor="white",
                label=labels[i].replace("\n", " "))
    axB.set_xticks(x)
    axB.set_xticklabels([m for m, _ in METRICS_B])
    axB.set_ylabel("score")
    axB.set_title("Key metrics (5 seeds, std)", fontsize=11, fontweight="bold")
    axB.grid(axis="y", alpha=0.3)
    axB.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.9)

    fig.suptitle("Core comparison  -  defense-law KG (7,354 triples, %d seeds)" % n_seeds,
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print("[saved] %s  (%d seeds)" % (out_path, n_seeds))
    for lab, k in zip(labels, keys):
        mu, sd = _mean_std(run_dir, k, ("overall", "mrr"))
        print("  %-20s MRR=%.3f+-%.3f" % (lab.replace("\n", " "), mu, sd))


def main():
    ap = argparse.ArgumentParser(description="core-4 comparison figure")
    ap.add_argument("--run-dir", default="experiments/final_run2",
                    help="experiment dir (raw/ contains per-seed JSON)")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <run-dir>/figures/core4_comparison.png)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "figures" / "core4_comparison.png"
    make_figure(run_dir, out)


if __name__ == "__main__":
    main()
