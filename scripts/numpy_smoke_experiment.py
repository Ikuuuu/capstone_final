"""샌드박스용 NumPy 미니 KGE 실험기.

PyTorch 설치가 차단된 환경에서 본 연구의 가설을 빠르게 검증할 수 있도록 만든
경량 구현. 실제 본 실험은 src/ 의 PyTorch 코드를 사용한다.

학습 모델 2종:
  1. TransE Baseline:     L = L_struct
  2. Ours (L_dir 포함):   L = L_struct + δ · L_dir

평가 지표: MRR, Hits@1/3/10  (Filtered tail prediction)
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ───── 데이터 로드 ─────
def load_kg(processed_dir):
    processed_dir = Path(processed_dir)
    def read_triples(p):
        triples = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                a = line.strip().split("\t")
                if len(a) >= 3:
                    triples.append((int(a[0]), int(a[1]), int(a[2])))
        return triples
    def read_map(p):
        m = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                a = line.strip().split("\t")
                if len(a) >= 2:
                    m[a[0]] = int(a[1])
        return m
    return {
        "train": read_triples(processed_dir / "train.tsv"),
        "valid": read_triples(processed_dir / "valid.tsv"),
        "test":  read_triples(processed_dir / "test.tsv"),
        "ent2id": read_map(processed_dir / "entity2id.tsv"),
        "rel2id": read_map(processed_dir / "relation2id.tsv"),
    }


# ───── TransE 점수 함수 ─────
def score(E, R, h, r, t):
    """f(h,r,t) = -||h+r-t||_1  (작을수록 좋다는 의미를 -로 뒤집어 큰 값이 좋게)"""
    return -np.sum(np.abs(E[h] + R[r] - E[t]), axis=-1)


# ───── 학습기 ─────
def train_kge(
    triples,
    num_ent,
    num_rel,
    dim=32,
    epochs=100,
    lr=0.01,
    margin=1.0,
    num_neg=5,
    use_l_dir=False,
    delta_dir=0.5,
    margin_dir=1.0,
    asym_relations=None,
    seed=42,
    log_every=20,
):
    """SGD margin-ranking loss (수동 그래디언트)."""
    rng = np.random.RandomState(seed)
    # 임베딩 초기화 (Xavier-like)
    bound = 6.0 / np.sqrt(dim)
    E = rng.uniform(-bound, bound, (num_ent, dim))
    R = rng.uniform(-bound, bound, (num_rel, dim))
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    asym_relations = asym_relations or set()
    triple_set = set(triples)

    history = []
    for ep in range(epochs):
        # 셔플
        random.Random(seed + ep).shuffle(triples)
        total_struct = 0.0
        total_dir = 0.0
        n_dir = 0
        # 정규화
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1.0)
        # mini-batch SGD (트리플 단위)
        for (h, r, t) in triples:
            # ─── L_struct (margin-ranking) ───
            for _ in range(num_neg):
                if rng.random() < 0.5:
                    h_neg, t_neg = rng.randint(num_ent), t
                else:
                    h_neg, t_neg = h, rng.randint(num_ent)
                if (h_neg, r, t_neg) in triple_set:
                    continue
                pos_dist = E[h] + R[r] - E[t]
                neg_dist = E[h_neg] + R[r] - E[t_neg]
                pos_norm = np.sum(np.abs(pos_dist))
                neg_norm = np.sum(np.abs(neg_dist))
                loss = margin + pos_norm - neg_norm
                if loss > 0:
                    total_struct += loss
                    # 그래디언트 ∂loss/∂E
                    g_pos = np.sign(pos_dist)
                    g_neg = np.sign(neg_dist)
                    E[h] -= lr * g_pos
                    E[t] += lr * g_pos
                    R[r] -= lr * g_pos
                    E[h_neg] += lr * g_neg
                    E[t_neg] -= lr * g_neg
                    R[r] += lr * g_neg
            # ─── L_dir (관계 방향성 제약, 본 연구 기여 ①) ───
            if use_l_dir and r not in asym_relations:
                # 대칭 관계는 제외했지만 보통은 모두 비대칭이므로 적용
                pass  # placeholder
            if use_l_dir:
                # f_forward = -||h+r-t||  →  forward 점수 (큰 값이 좋음)
                fwd = -np.sum(np.abs(E[h] + R[r] - E[t]))
                bwd = -np.sum(np.abs(E[t] + R[r] - E[h]))
                # 정방향이 역방향보다 margin_dir 이상 커야 함
                # 즉 −fwd + bwd + margin_dir < 0 이 되도록.
                # loss_dir = max(0, margin_dir + bwd - fwd)
                # (점수 함수 부호 일관성을 위해 −거리 사용 → 큰 값이 좋음)
                # 거리로 표현 : ||h+r-t|| 가 ||t+r-h|| 보다 margin 이상 작아야 함
                d_fw = np.sum(np.abs(E[h] + R[r] - E[t]))
                d_bw = np.sum(np.abs(E[t] + R[r] - E[h]))
                dir_loss = margin_dir + d_fw - d_bw
                if dir_loss > 0:
                    total_dir += dir_loss
                    n_dir += 1
                    g_fw = np.sign(E[h] + R[r] - E[t])
                    g_bw = np.sign(E[t] + R[r] - E[h])
                    # ∂d_fw/∂E[h] = g_fw,  ∂d_bw/∂E[h] = −g_bw
                    E[h] -= lr * delta_dir * (g_fw + g_bw)
                    E[t] -= lr * delta_dir * (-g_fw - g_bw)
                    R[r] -= lr * delta_dir * (g_fw - g_bw)
        if (ep + 1) % log_every == 0 or ep == 0:
            n_train = len(triples)
            history.append({
                "epoch": ep + 1,
                "L_struct": total_struct / max(n_train, 1),
                "L_dir": total_dir / max(n_dir, 1) if n_dir > 0 else 0.0,
                "n_dir_active": n_dir,
            })
            print(f"  ep {ep+1:3d}:  L_struct={total_struct/n_train:.4f}  "
                  f"L_dir={total_dir/max(n_dir,1):.4f}  (active {n_dir})")
    return E, R, history


# ───── 평가 ─────
def evaluate(E, R, eval_triples, all_triples, num_ent, by_relation=None):
    """Filtered tail prediction MRR/Hits@K."""
    triple_set = set(all_triples)
    rr = []
    h1 = []
    h3 = []
    h10 = []
    per_rel = defaultdict(list)
    for (h, r, t) in eval_triples:
        # 후보 = 모든 엔티티
        scores = score(E, R, h, r, np.arange(num_ent))   # (E,)
        # filter: 다른 양성 트리플의 점수를 -inf 로
        for tt in range(num_ent):
            if tt != t and (h, r, tt) in triple_set:
                scores[tt] = -np.inf
        # 큰 값이 좋음 (score 함수에서 -거리 반환)
        target_score = scores[t]
        rank = int((scores > target_score).sum()) + 1
        rr.append(1.0 / rank)
        h1.append(1.0 if rank <= 1 else 0.0)
        h3.append(1.0 if rank <= 3 else 0.0)
        h10.append(1.0 if rank <= 10 else 0.0)
        if by_relation is not None:
            per_rel[r].append((1.0 / rank, rank))
    out = {
        "mrr": float(np.mean(rr)),
        "hits@1": float(np.mean(h1)),
        "hits@3": float(np.mean(h3)),
        "hits@10": float(np.mean(h10)),
        "n": len(eval_triples),
    }
    if by_relation is not None:
        out["by_relation"] = {
            int(rid): {
                "mrr": float(np.mean([x[0] for x in xs])),
                "n": len(xs),
            } for rid, xs in per_rel.items()
        }
    return out


# ───── 메인 ─────
def main():
    print("=" * 70)
    print("샌드박스 NumPy 미니 실험  ─  TransE vs. Ours (L_dir 추가)")
    print("=" * 70)

    data = load_kg(ROOT / "data/processed")
    num_ent = len(data["ent2id"])
    num_rel = len(data["rel2id"])
    all_triples = data["train"] + data["valid"] + data["test"]

    id2rel = {v: k for k, v in data["rel2id"].items()}
    print(f"\n|E|={num_ent}, |R|={num_rel}")
    print(f"train={len(data['train'])}, valid={len(data['valid'])}, test={len(data['test'])}")
    print(f"관계 종류: {list(data['rel2id'].keys())}\n")

    # 비대칭 관계 (L_dir 적용 대상) — 본 연구 데이터에서 사실상 모두 비대칭
    # 단, type/소속/포함 같은 메타 관계는 학습 자체가 어려우므로 모두 적용
    asym_relation_ids = set(range(num_rel))

    HP = {"dim": 32, "epochs": 60, "lr": 0.05, "margin": 1.0, "num_neg": 5, "seed": 42}

    # ─── 1. Baseline TransE ───
    print("─" * 70)
    print("[1/2]  Baseline TransE 학습")
    print("─" * 70)
    E1, R1, hist1 = train_kge(
        list(data["train"]),
        num_ent, num_rel,
        use_l_dir=False,
        **HP,
    )
    print("\n  ▶ 평가 (Test)")
    res_baseline = evaluate(E1, R1, data["test"], all_triples, num_ent, by_relation=True)
    print(f"  MRR={res_baseline['mrr']:.4f}  "
          f"Hits@1={res_baseline['hits@1']:.4f}  "
          f"Hits@3={res_baseline['hits@3']:.4f}  "
          f"Hits@10={res_baseline['hits@10']:.4f}")

    # ─── 2. Ours (L_dir) ───
    print("\n" + "─" * 70)
    print("[2/2]  Ours  =  TransE + L_dir (관계 방향성 제약, δ=0.5)")
    print("─" * 70)
    E2, R2, hist2 = train_kge(
        list(data["train"]),
        num_ent, num_rel,
        use_l_dir=True, delta_dir=0.5, margin_dir=1.0,
        asym_relations=asym_relation_ids,
        **HP,
    )
    print("\n  ▶ 평가 (Test)")
    res_ours = evaluate(E2, R2, data["test"], all_triples, num_ent, by_relation=True)
    print(f"  MRR={res_ours['mrr']:.4f}  "
          f"Hits@1={res_ours['hits@1']:.4f}  "
          f"Hits@3={res_ours['hits@3']:.4f}  "
          f"Hits@10={res_ours['hits@10']:.4f}")

    # ─── 3. 비교 ───
    print("\n" + "=" * 70)
    print("결과 비교")
    print("=" * 70)
    print(f"{'Metric':<10s} {'TransE':>12s} {'Ours (+L_dir)':>16s} {'Δ (%p)':>10s}")
    print("─" * 50)
    for k in ["mrr", "hits@1", "hits@3", "hits@10"]:
        b, o = res_baseline[k], res_ours[k]
        diff = (o - b) * 100
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
        print(f"{k:<10s} {b:>12.4f} {o:>16.4f}   {sign}{abs(diff):>6.2f}p")

    # ─── 4. 관계별 비교 (L_dir 효과 핵심) ───
    print("\n" + "─" * 70)
    print("관계별 MRR (L_dir 효과는 인용·위임·상위법령 등 방향성 관계에서 두드러져야 함)")
    print("─" * 70)
    print(f"{'관계':<15s} {'#test':>6s} {'TransE MRR':>12s} {'Ours MRR':>12s} {'Δ':>8s}")
    print("─" * 60)
    for rid in sorted(res_baseline["by_relation"].keys()):
        rname = id2rel.get(rid, f"rel_{rid}")
        b = res_baseline["by_relation"][rid]
        o = res_ours["by_relation"][rid]
        diff = (o["mrr"] - b["mrr"]) * 100
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "─")
        print(f"{rname:<15s} {b['n']:>6d} {b['mrr']:>12.4f} {o['mrr']:>12.4f}   {sign}{abs(diff):>5.2f}p")

    # 결과 저장
    out_dir = ROOT / "experiments/sandbox_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({
            "baseline_transe": res_baseline,
            "ours_l_dir": res_ours,
            "hyperparameters": HP,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 : {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
