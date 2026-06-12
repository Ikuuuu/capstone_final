"""벡터화 SGD 학습 루프 (L_struct + 보조 손실 결합) — 중단·재개 지원.

손실 정의 (가이드라인 §2 Step6, §4 와 일치):
  L_total = L_struct + λ · (α·L_type + β·L_hier_attn + γ·L_attr + δ·L_dir)
  - L_struct      : margin-ranking (모든 모델 공통)
  - L_dir   (기여①): max(0, γ_dir + f(h,r,t) − f(t,r,h)), 대칭관계 제외
  - L_attr        : inverse pair 일관성 (f(h,r,t) ≈ f(t,r_inv,h))
  - L_type        : 엔티티를 자신의 온톨로지 클래스 중심으로 응집
  - L_hier_attn(기여②): 계층 관계 학습 가중치를 클래스 난이도에 따라 적응 재할당
λ 는 curriculum 으로 0→1 점진 증가.

재현성 : 에폭별 RNG = RandomState(seed·9973 + epoch) → 중단·재개와 무관하게
         동일 seed 면 동일 결과.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import KG
from .models import build_model, zeros_like_params


@dataclass
class TrainConfig:
    model: str = "TransE"
    dim: int = 100
    epochs: int = 150
    lr: float = 0.05
    margin: float = 1.0
    margin_dir: float = 1.0
    num_neg: int = 8
    batch_size: int = 1024
    weight_decay: float = 1e-5
    grad_clip: float = 5.0          # 파라미터 행(row)별 그래디언트 노름 상한
    # 보조 손실 가중치 (활성 여부 = 가중치 > 0)
    alpha_type: float = 0.0
    beta_hier: float = 0.0
    gamma_attr: float = 0.0
    delta_dir: float = 0.0
    dir_directional_only: bool = True   # L_dir 을 방향성 관계(인용·위임·소속 등)에만 적용
    # curriculum
    curriculum: bool = True
    lambda_end: float = 1.0
    schedule_epochs: int = 75       # λ 가 0→lambda_end 도달까지의 에폭
    hier_warmup: int = 40           # L_hier_attn 균등가중 유지 에폭
    seed: int = 42
    log_every: int = 50


# ── 모델별 표준 구성 (가이드라인 §3.3 / §5) ──
def config_for(model: str, **overrides) -> TrainConfig:
    base = dict(model=model)
    if model == "TransO":
        base.update(alpha_type=0.5)
    elif model == "Ours":
        # valid 셋 탐색으로 결정 (2026-06-12): L_type 은 소규모 KG 에서 과한 응집을
        # 유발하므로 0.1, L_dir 은 방향성 관계 한정 + delta 3.0 / margin_dir 2.0
        base.update(alpha_type=0.1, beta_hier=0.5, gamma_attr=0.5,
                    delta_dir=3.0, margin_dir=2.0)
    base.update(overrides)
    return TrainConfig(**base)


def _lambda(epoch, cfg: TrainConfig):
    if not cfg.curriculum:
        return cfg.lambda_end
    return cfg.lambda_end * min(1.0, (epoch + 1) / max(cfg.schedule_epochs, 1))


def _epoch_rng(cfg: TrainConfig, ep: int) -> np.random.RandomState:
    return np.random.RandomState((cfg.seed * 9973 + ep * 7919 + 1) % (2 ** 31 - 1))


def train_chunk(
    kg: KG,
    train: np.ndarray,
    cfg: TrainConfig,
    P: Optional[Dict[str, np.ndarray]] = None,
    start_epoch: int = 0,
    max_seconds: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[object, Dict[str, np.ndarray], List[Dict], int]:
    """start_epoch 부터 (시간 예산 내) 학습. 반환 next_epoch == cfg.epochs 면 완료."""
    model = build_model(cfg.model)
    if P is None:
        P = model.init_params(kg.num_ent, kg.num_rel, cfg.dim, np.random.RandomState(cfg.seed))
    ne = kg.num_ent

    use_dir = cfg.delta_dir > 0
    use_attr = cfg.gamma_attr > 0
    use_type = cfg.alpha_type > 0 and model.distance_based
    use_hier = cfg.beta_hier > 0 and len(kg.hierarchy_rel_ids) > 0

    h_all, r_all, t_all = train[:, 0], train[:, 1], train[:, 2]
    N = len(train)
    hier_ids = np.array(sorted(kg.hierarchy_rel_ids)) if use_hier else np.array([], dtype=int)
    inv_keys = np.array(sorted(kg.inverse_pairs.keys())) if use_attr else np.array([], dtype=int)

    t_start = time.time()
    history: List[Dict] = []
    ep = start_epoch
    while ep < cfg.epochs:
        rng = _epoch_rng(cfg, ep)
        lam = _lambda(ep, cfg)
        perm = rng.permutation(N)
        model.normalize(P)

        # ── L_hier_attn : 계층 클래스별 적응 가중치 (에폭 단위 갱신) ──
        hier_w: Dict[int, float] = {}
        if use_hier:
            if ep < cfg.hier_warmup:
                for rid in hier_ids:
                    hier_w[int(rid)] = 1.0 / len(hier_ids)
            else:
                losses = []
                for rid in hier_ids:
                    idx = np.where(r_all == rid)[0]
                    if len(idx) == 0:
                        losses.append(0.0)
                        continue
                    e_pos = model.energy(P, h_all[idx], r_all[idx], t_all[idx])
                    losses.append(float(np.mean(e_pos)))
                arr = np.array(losses)
                w = np.exp((arr - arr.max()) / (arr.std() + 1e-6))
                w = w / w.sum()
                for j, rid in enumerate(hier_ids):
                    hier_w[int(rid)] = float(w[j])

        tot = {"L_struct": 0.0, "L_dir": 0.0, "L_attr": 0.0, "L_type": 0.0, "L_hier": 0.0}
        n_batches = 0

        for start in range(0, N, cfg.batch_size):
            bidx = perm[start:start + cfg.batch_size]
            h, r, t = h_all[bidx], r_all[bidx], t_all[bidx]
            B = len(h)
            G = zeros_like_params(P)

            # ── L_struct (margin-ranking) ──
            K = cfg.num_neg
            neg_ent = rng.randint(0, ne, size=(B, K))
            corrupt_head = rng.rand(B, K) < 0.5
            neg_h = np.where(corrupt_head, neg_ent, h[:, None])
            neg_t = np.where(corrupt_head, t[:, None], neg_ent)
            neg_r = np.repeat(r[:, None], K, axis=1)

            e_pos = model.energy(P, h, r, t)
            e_neg = model.energy(P, neg_h.ravel(), neg_r.ravel(), neg_t.ravel()).reshape(B, K)
            active = (cfg.margin + e_pos[:, None] - e_neg) > 0

            w_s = np.ones(B)
            if use_hier:
                for rid, wv in hier_w.items():
                    m = r == rid
                    if m.any():
                        w_s[m] += lam * cfg.beta_hier * wv * len(hier_ids)
                tot["L_hier"] += float(np.mean([hier_w[int(x)] for x in hier_ids]))

            pos_coeff = w_s * active.sum(axis=1)
            model.grad_accum(P, G, h, r, t, pos_coeff)
            neg_coeff = -(np.repeat(w_s[:, None], K, axis=1) * active).ravel()
            model.grad_accum(P, G, neg_h.ravel(), neg_r.ravel(), neg_t.ravel(), neg_coeff)
            tot["L_struct"] += float(np.clip(cfg.margin + e_pos[:, None] - e_neg, 0, None).mean())

            # ── L_dir (기여①) ──
            if use_dir:
                sym = kg.symmetric_rel_ids
                if cfg.dir_directional_only and kg.directional_rel_ids:
                    keep = np.isin(r, list(kg.directional_rel_ids))
                else:
                    keep = np.array([rr not in sym for rr in r]) if sym else np.ones(B, bool)
                if keep.any():
                    hh, rr, tt = h[keep], r[keep], t[keep]
                    e_fw = model.energy(P, hh, rr, tt)
                    e_bw = model.energy(P, tt, rr, hh)
                    act = (cfg.margin_dir + e_fw - e_bw) > 0
                    coeff = lam * cfg.delta_dir * act.astype(float)
                    model.grad_accum(P, G, hh, rr, tt, coeff)
                    model.grad_accum(P, G, tt, rr, hh, -coeff)
                    tot["L_dir"] += float(np.clip(cfg.margin_dir + e_fw - e_bw, 0, None).mean())

            # ── L_attr (inverse pair 일관성) ──
            if use_attr and len(inv_keys) > 0:
                m = np.isin(r, inv_keys)
                if m.any():
                    hh, rr, tt = h[m], r[m], t[m]
                    rinv = np.array([kg.inverse_pairs[int(x)] for x in rr])
                    e_fw = model.energy(P, hh, rr, tt)
                    e_bw = model.energy(P, tt, rinv, hh)
                    diff = e_fw - e_bw
                    coeff = lam * cfg.gamma_attr * np.tanh(diff)   # 유계 그래디언트
                    model.grad_accum(P, G, hh, rr, tt, coeff)
                    model.grad_accum(P, G, tt, rinv, hh, -coeff)
                    tot["L_attr"] += float(np.mean(np.abs(diff)))

            # ── L_type (온톨로지 클래스 응집) ──
            if use_type:
                ents = np.concatenate([h, t])
                types = kg.entity_type[ents]
                valid = types >= 0
                if valid.any():
                    ev, tv = ents[valid], types[valid]
                    uniq = np.unique(tv)
                    cent = {int(c): P["E"][kg.entity_type == c].mean(axis=0)
                            for c in uniq if (kg.entity_type == c).any()}
                    grad = np.zeros((len(ev), P["E"].shape[1]))
                    for i, (e, c) in enumerate(zip(ev, tv)):
                        grad[i] = P["E"][e] - cent[int(c)]
                    np.add.at(G["E"], ev, lam * cfg.alpha_type * 2.0 * grad)
                    tot["L_type"] += float(np.mean(np.sum(grad ** 2, axis=1)))

            # ── 그래디언트 클리핑 + SGD ──
            if cfg.grad_clip > 0:
                for k in G:
                    if k.startswith("_"):
                        continue
                    rn = np.linalg.norm(G[k], axis=1, keepdims=True)
                    G[k] *= np.minimum(1.0, cfg.grad_clip / np.maximum(rn, 1e-12))
            for k in P:
                if k.startswith("_"):
                    continue
                P[k] -= cfg.lr * (G[k] + cfg.weight_decay * P[k])
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in tot.items()}
        if verbose and ((ep + 1) % cfg.log_every == 0 or ep == 0):
            print(f"    ep{ep + 1:3d} lam={lam:.2f}  " +
                  "  ".join(f"{k}={avg[k]:.3f}" for k in ["L_struct", "L_dir", "L_attr", "L_type"]))
        history.append({"epoch": ep + 1, **avg})
        ep += 1

        if max_seconds is not None and (time.time() - t_start) > max_seconds:
            break

    return model, P, history, ep


def train_model(kg: KG, train: np.ndarray, cfg: TrainConfig, verbose=False):
    """전체 에폭을 한 번에 학습 (단순 API, 하위 호환)."""
    model, P, history, _ = train_chunk(kg, train, cfg, verbose=verbose)
    return model, P, history
