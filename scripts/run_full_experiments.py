"""전체 실험 러너 — 중단·재개(stepper) 방식 (졸업논문 가이드라인 §3 전체 스펙).

샌드박스의 호출당 시간 제한(45s) 안에서 전체 실험을 완주하도록,
작업 큐(jobs.json)와 학습 체크포인트로 진행 상태를 저장하고 반복 호출로 재개한다.

    python scripts/run_full_experiments.py --out full_run --budget 34
    (DONE 파일이 생길 때까지 반복 호출)

Stage A : 메인 비교  — 6 모델 (TransE/RotatE/DistMult/ComplEx/TransO/Ours) × 5 seed
Stage B : Ablation   — Ours 변형 4종 (−L_dir/−L_hier_attn/−L_attr/−L_type) × 5 seed
Stage C : 강건성(H3) — TransE·Ours × train {25,50,75}% × 3 seed
Stage D : 가설 검정  — H1/H2/H3 + 집계(CSV/PNG/report.md)

산출물 (experiments/<run>/):
  results.json / main_results.csv / per_relation.csv / ablation.csv / robustness.csv
  hypotheses.json / report.md / figures/*.png / manifest.json / progress.log
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.np_engine.data import load_kg, subsample_train
from src.np_engine.evaluate import evaluate
from src.np_engine.stats import paired_bootstrap, test_H3
from src.np_engine.train import config_for, train_chunk

REL_EN = {
    "type": "type", "소속": "belongsTo", "포함": "contains",
    "외부인용": "citesExternal", "인용": "cites", "소관기관": "competentAuthority",
    "관할법령": "governingLaw", "위임": "delegates", "위임받음": "delegatedBy",
    "상위법령": "parentLaw", "하위법령": "childLaw", "발령기관": "issuedBy",
}
MAIN_MODELS = ["TransE", "RotatE", "DistMult", "ComplEx", "TransO", "Ours"]
ABLATIONS = {
    "Ours-L_dir":       {"delta_dir": 0.0},
    "Ours-L_hier_attn": {"beta_hier": 0.0},
    "Ours-L_attr":      {"gamma_attr": 0.0},
    "Ours-L_type":      {"alpha_type": 0.0},
}
EPOCHS, DIM = 150, 100
SEEDS = [42, 43, 44, 45, 46]
ROB_SEEDS = [42, 43, 44]
ROB_FRACS = [0.25, 0.5, 0.75]


def log(out_dir, msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (out_dir / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_jobs():
    jobs = []
    for m in MAIN_MODELS:
        for s in SEEDS:
            jobs.append({"id": f"main_{m}_s{s}", "stage": "main", "model": m,
                         "seed": s, "frac": 1.0, "overrides": {}})
    for name, ov in ABLATIONS.items():
        for s in SEEDS:
            jobs.append({"id": f"abl_{name}_s{s}", "stage": "ablation", "model": "Ours",
                         "variant": name, "seed": s, "frac": 1.0, "overrides": ov})
    for m in ["TransE", "Ours"]:
        for f in ROB_FRACS:
            for s in ROB_SEEDS:
                jobs.append({"id": f"rob_{m}_f{int(f*100)}_s{s}", "stage": "robustness",
                             "model": m, "seed": s, "frac": f, "overrides": {}})
    for j in jobs:
        j["status"] = "pending"      # pending | training | done
        j["next_epoch"] = 0
    return jobs


def save_ckpt(path, P, next_epoch):
    np.savez_compressed(path, __next_epoch=np.array([next_epoch]),
                        **{k: v for k, v in P.items()})


def load_ckpt(path):
    z = np.load(path)
    ne = int(z["__next_epoch"][0])
    P = {k: z[k] for k in z.files if k != "__next_epoch"}
    return P, ne


def strip_private(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def job_cfg(job):
    return config_for(job["model"], seed=job["seed"], dim=DIM, epochs=EPOCHS,
                      grad_clip=5.0, schedule_epochs=EPOCHS // 2,
                      hier_warmup=max(20, EPOCHS // 4), **job["overrides"])


def process_jobs(kg, jobs, out_dir, raw_dir, state_dir, budget):
    """시간 예산 내에서 최대한 많은 작업 진행. 변경 여부 반환."""
    t0 = time.time()
    changed = False
    for job in jobs:
        if job["status"] == "done":
            continue
        remain = budget - (time.time() - t0)
        if remain < 6:
            break
        ck = state_dir / f"{job['id']}.npz"
        P, start_ep = (load_ckpt(ck) if ck.exists() else (None, 0))
        train_arr = (kg.train if job["frac"] >= 1.0
                     else subsample_train(kg.train, job["frac"], seed=job["seed"]))
        cfg = job_cfg(job)

        if start_ep < cfg.epochs:
            model, P, hist, next_ep = train_chunk(
                kg, train_arr, cfg, P=P, start_epoch=start_ep, max_seconds=remain - 3)
            save_ckpt(ck, P, next_ep)
            job["next_epoch"] = next_ep
            job["status"] = "training"
            changed = True
            if next_ep < cfg.epochs:
                log(out_dir, f"  {job['id']}: epoch {next_ep}/{cfg.epochs} (재개 대기)")
                continue   # 예산 소진 → 다음 호출에서 재개

        # 학습 완료 → 평가 (RotatE 평가 ~6s)
        remain = budget - (time.time() - t0)
        if remain < 10:
            break
        from src.np_engine.models import build_model
        model = build_model(cfg.model)
        res = evaluate(model, P, kg, kg.test)
        np.savez_compressed(raw_dir / f"{job['id']}.npz",
                            rr=res["_rr"], rel_of=res["_rel_of"],
                            dir_mask=res["_dir_mask"], hier_mask=res["_hier_mask"])
        (raw_dir / f"{job['id']}.json").write_text(
            json.dumps(strip_private(res), ensure_ascii=False), encoding="utf-8")
        job["status"] = "done"
        changed = True
        o = res["overall"]
        log(out_dir, f"  {job['id']} 완료: MRR={o['mrr']:.4f} H@1={o['hits@1']:.4f} "
                     f"H@10={o['hits@10']:.4f}")
    return changed


# ───────────────────────── 집계 (Stage D) ─────────────────────────

def load_results(jobs, raw_dir):
    res = {"main": {}, "ablation": {}, "robustness": {}}
    for job in jobs:
        metrics = json.loads((raw_dir / f"{job['id']}.json").read_text(encoding="utf-8"))
        z = np.load(raw_dir / f"{job['id']}.npz")
        metrics["_rr"] = z["rr"]
        metrics["_dir_mask"] = z["dir_mask"]
        if job["stage"] == "main":
            res["main"].setdefault(job["model"], []).append(metrics)
        elif job["stage"] == "ablation":
            res["ablation"].setdefault(job["variant"], []).append(metrics)
        else:
            res["robustness"].setdefault(job["model"], {}).setdefault(
                str(job["frac"]), []).append(metrics)
    res["ablation"]["Ours(full)"] = res["main"]["Ours"]
    # 표시 순서 : full 을 맨 앞으로
    order = ["Ours(full)"] + list(ABLATIONS.keys())
    res["ablation"] = {k: res["ablation"][k] for k in order if k in res["ablation"]}
    return res


def mean_metric(runs, *path):
    vals = []
    for r in runs:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


def aggregate(kg, jobs, out_dir, raw_dir, fig_dir):
    results = load_results(jobs, raw_dir)

    # results.json
    clean = {
        "main": {m: [strip_private(r) for r in v] for m, v in results["main"].items()},
        "ablation": {m: [strip_private(r) for r in v] for m, v in results["ablation"].items()},
        "robustness": {m: {f: [strip_private(r) for r in v] for f, v in d.items()}
                       for m, d in results["robustness"].items()},
    }
    (out_dir / "results.json").write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 가설 검정 ──
    hyp = {}
    rr_o = np.concatenate([r["_rr"][r["_dir_mask"]] for r in results["main"]["Ours"]])
    rr_t = np.concatenate([r["_rr"][r["_dir_mask"]] for r in results["main"]["TransO"]])
    bs = paired_bootstrap(rr_o, rr_t, n_boot=1000)
    gm = lambda m, k: float(np.mean([r["groups"]["directional"][k]
                                     for r in results["main"][m]]))
    d_h1 = (gm("Ours", "hits@1") - gm("TransO", "hits@1")) * 100
    hyp["H1"] = {
        "hypothesis": "H1_directionality",
        "criterion": "방향성관계 Hits@1 >= TransO +5.0%p & paired-bootstrap p<0.05",
        "ours_hits@1": gm("Ours", "hits@1"), "transo_hits@1": gm("TransO", "hits@1"),
        "delta_hits@1_pp": d_h1,
        "delta_mrr_pp": (gm("Ours", "mrr") - gm("TransO", "mrr")) * 100,
        "bootstrap_mrr": bs,
        "passed": bool(d_h1 >= 5.0 and bs["p_value"] < 0.05),
    }
    hm = lambda v, k=("hierarchy", "mrr"): float(
        np.mean([r["groups"][k[0]][k[1]] for r in results["ablation"][v]]))
    mrr_full, mrr_abl = hm("Ours(full)"), hm("Ours-L_hier_attn")
    drop = (mrr_full - mrr_abl) * 100
    hyp["H2"] = {
        "hypothesis": "H2_hierarchy_adaptation",
        "criterion": "L_hier_attn 제거 시 계층관계 MRR >= 3.0%p 저하 (5 seed 평균)",
        "mrr_full": mrr_full, "mrr_ablation": mrr_abl,
        "drop_pp": drop, "passed": bool(drop >= 3.0),
    }
    base_mrr = {m: float(np.mean([r["overall"]["mrr"] for r in results["main"][m]]))
                for m in ["TransE", "Ours"]}
    deltas = {}
    for m in ["TransE", "Ours"]:
        deltas[m] = {}
        for frac in ROB_FRACS:
            mu = float(np.mean([r["overall"]["mrr"]
                                for r in results["robustness"][m][str(frac)]]))
            deltas[m][frac] = base_mrr[m] - mu
    hyp["H3"] = test_H3(deltas["Ours"], deltas["TransE"], factor=0.7)
    hyp["H3"]["deltas"] = {m: {str(k): v for k, v in d.items()} for m, d in deltas.items()}
    hyp["H3"]["base_mrr"] = base_mrr
    (out_dir / "hypotheses.json").write_text(
        json.dumps(hyp, ensure_ascii=False, indent=2), encoding="utf-8")

    _make_csvs(results, out_dir, kg)
    _make_figures(results, fig_dir)
    _make_report(results, hyp, out_dir)
    for k, v in hyp.items():
        log(out_dir, f"  {k}: passed={v['passed']}")


def _make_csvs(results, out_dir, kg):
    import csv
    with (out_dir / "main_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "MRR", "MRR_std", "Hits@1", "Hits@3", "Hits@10",
                    "dir_MRR", "dir_Hits@1", "hier_MRR", "n_seeds"])
        for m, runs in results["main"].items():
            mu, sd = mean_metric(runs, "overall", "mrr")
            row = [m, f"{mu:.4f}", f"{sd:.4f}"]
            for k in ["hits@1", "hits@3", "hits@10"]:
                row.append(f"{mean_metric(runs, 'overall', k)[0]:.4f}")
            for grp, k in [("directional", "mrr"), ("directional", "hits@1"),
                           ("hierarchy", "mrr")]:
                row.append(f"{mean_metric(runs, 'groups', grp, k)[0]:.4f}")
            row.append(len(runs))
            w.writerow(row)
    rels = [kg.id2rel[i] for i in sorted(kg.id2rel)]
    with (out_dir / "per_relation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "relation", "relation_en", "MRR", "Hits@1", "n"])
        for m, runs in results["main"].items():
            for rel in rels:
                if rel not in runs[0]["per_relation"]:
                    continue
                mrr = float(np.mean([r["per_relation"][rel]["mrr"] for r in runs]))
                h1 = float(np.mean([r["per_relation"][rel]["hits@1"] for r in runs]))
                w.writerow([m, rel, REL_EN.get(rel, rel), f"{mrr:.4f}", f"{h1:.4f}",
                            runs[0]["per_relation"][rel]["n"]])
    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant", "MRR", "MRR_std", "Hits@1", "dir_MRR", "hier_MRR"])
        for m, runs in results["ablation"].items():
            mu, sd = mean_metric(runs, "overall", "mrr")
            w.writerow([m, f"{mu:.4f}", f"{sd:.4f}",
                        f"{mean_metric(runs, 'overall', 'hits@1')[0]:.4f}",
                        f"{mean_metric(runs, 'groups', 'directional', 'mrr')[0]:.4f}",
                        f"{mean_metric(runs, 'groups', 'hierarchy', 'mrr')[0]:.4f}"])
    with (out_dir / "robustness.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "train_fraction", "MRR", "MRR_std"])
        for m, fr in results["robustness"].items():
            for frac, runs in sorted(fr.items()):
                vals = [r["overall"]["mrr"] for r in runs]
                w.writerow([m, frac, f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}"])
        for m in results["robustness"]:
            vals = [r["overall"]["mrr"] for r in results["main"][m]]
            w.writerow([m, "1.0", f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}"])


def _make_figures(results, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results["main"].keys())
    colors = ["#888888"] * (len(models) - 1) + ["#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "mrr", "MRR (Filtered)"),
                           (axes[1], "hits@10", "Hits@10 (Filtered)")]:
        mus = [mean_metric(results["main"][m], "overall", key)[0] for m in models]
        sds = [mean_metric(results["main"][m], "overall", key)[1] for m in models]
        ax.bar(models, mus, yerr=sds, capsize=3, color=colors)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Main comparison - defense-law KG (5 seeds)")
    fig.tight_layout(); fig.savefig(fig_dir / "main_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    mus = [mean_metric(results["main"][m], "groups", "directional", "hits@1")[0]
           for m in models]
    ax.bar(models, mus, color=colors)
    ax.set_title("Directional-relation Hits@1 (H1: Ours vs TransO)")
    ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(fig_dir / "directional_hits1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(results["ablation"].keys())
    mus = [mean_metric(results["ablation"][m], "overall", "mrr")[0] for m in names]
    sds = [mean_metric(results["ablation"][m], "overall", "mrr")[1] for m in names]
    ax.bar(names, mus, yerr=sds, capsize=3,
           color=["#d62728"] + ["#1f77b4"] * (len(names) - 1))
    ax.set_title("Ablation: overall MRR (remove one loss)")
    ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(fig_dir / "ablation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    fracs = ROB_FRACS + [1.0]
    for m, style in [("TransE", "o--"), ("Ours", "s-")]:
        ys = [np.mean([r["overall"]["mrr"] for r in results["robustness"][m][str(fr)]])
              for fr in ROB_FRACS]
        ys.append(np.mean([r["overall"]["mrr"] for r in results["main"][m]]))
        ax.plot([int(f * 100) for f in fracs], ys, style, label=m)
    ax.set_xlabel("train fraction (%)"); ax.set_ylabel("MRR")
    ax.set_title("Robustness to training-data reduction (H3)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "robustness.png", dpi=150)
    plt.close(fig)


def _make_report(results, hyp, out_dir):
    L = []
    L.append("# 실험 결과 리포트 — 온톨로지 기반 KGE (국방 법령 KG)\n")
    L.append(f"- 생성: {datetime.now().isoformat(timespec='seconds')}  |  "
             f"엔진: NumPy np_engine (PyTorch 미설치 샌드박스)")
    L.append(f"- 설정: dim={DIM}, epochs={EPOCHS}, seeds={SEEDS}, "
             f"negatives=8, margin=1.0, λ curriculum 0→1\n")
    L.append("## 1. 메인 비교 (5 seed 평균)\n")
    L.append("| 모델 | MRR | Hits@1 | Hits@3 | Hits@10 | 방향성 MRR | 방향성 H@1 | 계층 MRR |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m, runs in results["main"].items():
        row = [f"**{m}**" if m == "Ours" else m]
        for path in [("overall", "mrr"), ("overall", "hits@1"), ("overall", "hits@3"),
                     ("overall", "hits@10"), ("groups", "directional", "mrr"),
                     ("groups", "directional", "hits@1"), ("groups", "hierarchy", "mrr")]:
            row.append(f"{mean_metric(runs, *path)[0]:.4f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("\n## 2. 가설 검정\n")
    for k in ["H1", "H2", "H3"]:
        v = hyp[k]
        L.append(f"### {k} — {'✅ 충족' if v['passed'] else '❌ 미충족'}")
        L.append(f"- 기준: {v['criterion']}")
        if k == "H1":
            L.append(f"- Ours 방향성 Hits@1={v['ours_hits@1']:.4f} vs "
                     f"TransO={v['transo_hits@1']:.4f} (Δ {v['delta_hits@1_pp']:+.2f}%p)")
            b = v["bootstrap_mrr"]
            L.append(f"- paired bootstrap(MRR): Δ={b['delta']:+.4f}, p={b['p_value']:.4f}, "
                     f"95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}], n={b['n']}")
        elif k == "H2":
            L.append(f"- 계층 MRR: full={v['mrr_full']:.4f} → "
                     f"−L_hier_attn={v['mrr_ablation']:.4f} (저하 {v['drop_pp']:+.2f}%p)")
        else:
            if v.get("threshold") is not None:
                L.append(f"- ΔMRR(50%): Ours={v['delta_mrr_ours_50pct']:.4f} vs "
                         f"기준(TransE×0.7)={v['threshold']:.4f}")
        L.append("")
    L.append("## 3. Ablation (Ours 구성요소 제거, 5 seed 평균)\n")
    L.append("| 변형 | MRR | Hits@1 | 방향성 MRR | 계층 MRR |")
    L.append("|---|---|---|---|---|")
    for m, runs in results["ablation"].items():
        L.append(f"| {m} | {mean_metric(runs, 'overall', 'mrr')[0]:.4f} "
                 f"| {mean_metric(runs, 'overall', 'hits@1')[0]:.4f} "
                 f"| {mean_metric(runs, 'groups', 'directional', 'mrr')[0]:.4f} "
                 f"| {mean_metric(runs, 'groups', 'hierarchy', 'mrr')[0]:.4f} |")
    L.append("\n## 4. 산출물\n")
    L.append("`results.json`(원시지표) · `*.csv`(표) · `figures/*.png`(그래프) · "
             "`hypotheses.json`(가설판정) · `manifest.json`(재현정보) · `raw/`(런별 원자료)")
    L.append("\n> 주의: 폐쇄 샌드박스 제약으로 PyTorch 대신 동일 수식의 NumPy 엔진"
             "(src/np_engine)으로 수행. GPU 본 실험 시 src/ PyTorch 코드 사용.")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="full_run")
    ap.add_argument("--budget", type=float, default=34.0)
    args = ap.parse_args()

    out_dir = ROOT / "experiments" / args.out
    fig_dir, raw_dir, state_dir = out_dir / "figures", out_dir / "raw", out_dir / "state"
    for d in (out_dir, fig_dir, raw_dir, state_dir):
        d.mkdir(parents=True, exist_ok=True)

    jobs_file = state_dir / "jobs.json"
    if jobs_file.exists():
        jobs = json.loads(jobs_file.read_text(encoding="utf-8"))
    else:
        jobs = build_jobs()
        manifest = {
            "run_name": args.out, "date": datetime.now().isoformat(),
            "engine": "np_engine (NumPy)",
            "hyperparameters": {"dim": DIM, "epochs": EPOCHS, "lr": 0.05,
                                "margin": 1.0, "margin_dir": 1.0, "num_neg": 8,
                                "batch_size": 1024, "weight_decay": 1e-5,
                                "grad_clip": 5.0, "curriculum": "lambda 0->1"},
            "seeds_main": SEEDS, "seeds_robust": ROB_SEEDS, "n_jobs": len(jobs),
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log(out_dir, f"작업 큐 생성 : {len(jobs)} jobs")

    kg = load_kg(ROOT / "data/processed")
    process_jobs(kg, jobs, out_dir, raw_dir, state_dir, args.budget)
    jobs_file.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")

    n_done = sum(1 for j in jobs if j["status"] == "done")
    log(out_dir, f"진행 : {n_done}/{len(jobs)} jobs 완료")
    if n_done == len(jobs) and not (out_dir / "DONE").exists():
        log(out_dir, "=== Stage D : 집계 + 가설 검정 ===")
        aggregate(kg, jobs, out_dir, raw_dir, fig_dir)
        (out_dir / "DONE").write_text("done", encoding="utf-8")
        log(out_dir, "=== 완료 ===")
    print(f"STATUS {n_done}/{len(jobs)}")


if __name__ == "__main__":
    main()
