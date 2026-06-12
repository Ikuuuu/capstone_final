"""KGE 모델 6종 (벡터화 NumPy, 해석적 그래디언트).

통일 인터페이스 ─ "에너지(energy)" 는 낮을수록 좋은 트리플.
 거리 기반 모델(TransE/RotatE/TransO/Ours) : energy = 거리
 의미매칭 모델(DistMult/ComplEx)           : energy = −score
margin-ranking 학습과 평가가 동일 규약(낮을수록 정답)을 공유한다.

각 모델은 다음을 제공한다:
  init_params(ne, nr, d, rng) -> dict[str, np.ndarray]
  energy(P, h, r, t) -> (N,)
  grad_accum(P, G, h, r, t, coeff) : G 에 coeff·∂energy/∂param 누적
  normalize(P) : 에폭마다 임베딩 정규화(안정화)
"""
from __future__ import annotations

from typing import Dict

import numpy as np

Params = Dict[str, np.ndarray]
EPS = 1e-12


def xavier(rng, shape):
    bound = 6.0 / np.sqrt(shape[1])
    return rng.uniform(-bound, bound, shape)


def zeros_like_params(P: Params) -> Params:
    return {k: np.zeros_like(v) for k, v in P.items()}


def _clip_norm(M, max_norm=1.0):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(n, EPS))
    M *= scale


# ───────────────────────── TransE 계열 ─────────────────────────
class TransE:
    name = "TransE"
    distance_based = True

    def init_params(self, ne, nr, d, rng) -> Params:
        E = xavier(rng, (ne, d))
        R = xavier(rng, (nr, d))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), EPS)
        return {"E": E, "R": R}

    def energy(self, P, h, r, t):
        diff = P["E"][h] + P["R"][r] - P["E"][t]
        return np.sum(np.abs(diff), axis=-1)

    def grad_accum(self, P, G, h, r, t, coeff):
        diff = P["E"][h] + P["R"][r] - P["E"][t]
        s = np.sign(diff) * coeff[:, None]
        np.add.at(G["E"], h, s)
        np.add.at(G["E"], t, -s)
        np.add.at(G["R"], r, s)

    def normalize(self, P):
        _clip_norm(P["E"], 1.0)


# TransO / Ours 는 점수 함수가 TransE 와 동일(거리). 차이는 학습 손실(auxiliary)에 있음.
class TransO(TransE):
    name = "TransO"


class Ours(TransE):
    name = "Ours"


# ───────────────────────── DistMult ─────────────────────────
class DistMult:
    name = "DistMult"
    distance_based = False

    def init_params(self, ne, nr, d, rng) -> Params:
        return {"E": xavier(rng, (ne, d)) * 0.1, "R": xavier(rng, (nr, d)) * 0.1}

    def energy(self, P, h, r, t):
        score = np.sum(P["E"][h] * P["R"][r] * P["E"][t], axis=-1)
        return -score

    def grad_accum(self, P, G, h, r, t, coeff):
        eh, rr, et = P["E"][h], P["R"][r], P["E"][t]
        c = coeff[:, None]
        np.add.at(G["E"], h, -(rr * et) * c)
        np.add.at(G["E"], t, -(eh * rr) * c)
        np.add.at(G["R"], r, -(eh * et) * c)

    def normalize(self, P):
        _clip_norm(P["E"], 2.0)


# ───────────────────────── ComplEx ─────────────────────────
class ComplEx:
    name = "ComplEx"
    distance_based = False

    def init_params(self, ne, nr, d, rng) -> Params:
        # E, R 모두 (·, 2d) = [real | imag]
        return {"E": xavier(rng, (ne, 2 * d)) * 0.1,
                "R": xavier(rng, (nr, 2 * d)) * 0.1,
                "_d": np.array([d])}

    def _split(self, M):
        d = M.shape[1] // 2
        return M[:, :d], M[:, d:]

    def energy(self, P, h, r, t):
        ah, bh = self._split(P["E"][h])
        cr, dr = self._split(P["R"][r])
        at, bt = self._split(P["E"][t])
        score = np.sum(ah * cr * at + ah * dr * bt + bh * cr * bt - bh * dr * at, axis=-1)
        return -score

    def grad_accum(self, P, G, h, r, t, coeff):
        ah, bh = self._split(P["E"][h])
        cr, dr = self._split(P["R"][r])
        at, bt = self._split(P["E"][t])
        c = coeff[:, None]
        # ∂score/∂· (energy = −score 이므로 부호 반전 후 누적)
        d_ah = -(cr * at + dr * bt) * c
        d_bh = -(cr * bt - dr * at) * c
        d_cr = -(ah * at + bh * bt) * c
        d_dr = -(ah * bt - bh * at) * c
        d_at = -(ah * cr - bh * dr) * c
        d_bt = -(ah * dr + bh * cr) * c
        np.add.at(G["E"], h, np.concatenate([d_ah, d_bh], axis=1))
        np.add.at(G["E"], t, np.concatenate([d_at, d_bt], axis=1))
        np.add.at(G["R"], r, np.concatenate([d_cr, d_dr], axis=1))

    def normalize(self, P):
        _clip_norm(P["E"], 2.0)


# ───────────────────────── RotatE ─────────────────────────
class RotatE:
    name = "RotatE"
    distance_based = True

    def init_params(self, ne, nr, d, rng) -> Params:
        E = xavier(rng, (ne, 2 * d)) * 0.1            # [real | imag]
        phase = rng.uniform(-np.pi, np.pi, (nr, d))   # 관계 = 단위 복소수의 위상
        return {"E": E, "Rphase": phase, "_d": np.array([d])}

    def _split(self, M):
        d = M.shape[1] // 2
        return M[:, :d], M[:, d:]

    def _terms(self, P, h, r, t):
        ah, bh = self._split(P["E"][h])
        at, bt = self._split(P["E"][t])
        th = P["Rphase"][r]
        cos, sin = np.cos(th), np.sin(th)
        dr_re = ah * cos - bh * sin - at
        dr_im = ah * sin + bh * cos - bt
        return ah, bh, at, bt, cos, sin, dr_re, dr_im

    def energy(self, P, h, r, t):
        *_, dr_re, dr_im = self._terms(P, h, r, t)
        return np.sum(np.sqrt(dr_re ** 2 + dr_im ** 2 + EPS), axis=-1)

    def grad_accum(self, P, G, h, r, t, coeff):
        ah, bh, at, bt, cos, sin, dr_re, dr_im = self._terms(P, h, r, t)
        m = np.sqrt(dr_re ** 2 + dr_im ** 2 + EPS)
        gre, gim = dr_re / m, dr_im / m
        c = coeff[:, None]
        d_ah = (gre * cos + gim * sin) * c
        d_bh = (-gre * sin + gim * cos) * c
        d_at = (-gre) * c
        d_bt = (-gim) * c
        d_th = (gre * (-ah * sin - bh * cos) + gim * (ah * cos - bh * sin)) * c
        np.add.at(G["E"], h, np.concatenate([d_ah, d_bh], axis=1))
        np.add.at(G["E"], t, np.concatenate([d_at, d_bt], axis=1))
        np.add.at(G["Rphase"], r, d_th)

    def normalize(self, P):
        _clip_norm(P["E"], 2.0)


REGISTRY = {
    "TransE": TransE,
    "RotatE": RotatE,
    "DistMult": DistMult,
    "ComplEx": ComplEx,
    "TransO": TransO,
    "Ours": Ours,
}


def build_model(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name}; available={list(REGISTRY)}")
    return REGISTRY[name]()


# ───────────────────────── TKRL (Xie et al., 2016 간소화) ─────────────────────────
class TKRL:
    """타입별 투영 행렬 M_c 로 엔티티를 투영 후 TransE 거리.

    f(h,r,t) = ‖ M_{type(h)}·h + r − M_{type(t)}·t ‖₁
    타입이 없는 엔티티는 항등 투영 슬롯(C번)을 사용.
    """
    name = "TKRL"
    distance_based = True

    def set_context(self, kg):
        et = kg.entity_type
        classes = sorted({int(c) for c in et if c >= 0})
        self._cls_idx = {c: i for i, c in enumerate(classes)}
        self._n_cls = len(classes)
        idx = np.full(len(et), self._n_cls, dtype=np.int64)   # 기본 = 항등 슬롯
        for e, c in enumerate(et):
            if c >= 0:
                idx[e] = self._cls_idx[int(c)]
        self._etype = idx

    def init_params(self, ne, nr, d, rng) -> Params:
        E = xavier(rng, (ne, d))
        R = xavier(rng, (nr, d))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), EPS)
        C = getattr(self, "_n_cls", 8)
        M = np.tile(np.eye(d), (C + 1, 1, 1)) + rng.normal(0, 0.01, (C + 1, d, d))
        return {"E": E, "R": R, "M": M}

    def _proj(self, P, ents):
        """타입별 그룹 행렬곱 (row-wise einsum 대비 5~10배 빠름)."""
        t = self._etype[ents]
        out = np.empty((len(ents), P["E"].shape[1]))
        for c in np.unique(t):
            m = t == c
            out[m] = P["E"][ents[m]] @ P["M"][c].T
        return out

    def prepare_eval(self, P):
        """평가 전 전체 엔티티 투영을 1회 계산 (에너지 호출 가속)."""
        self._PE = self._proj(P, np.arange(P["E"].shape[0]))

    def energy(self, P, h, r, t):
        PE = getattr(self, "_PE", None)
        if PE is not None:
            diff = PE[h] + P["R"][r] - PE[t]
        else:
            diff = self._proj(P, h) + P["R"][r] - self._proj(P, t)
        return np.sum(np.abs(diff), axis=-1)

    def grad_accum(self, P, G, h, r, t, coeff):
        th, tt = self._etype[h], self._etype[t]
        ph, pt = self._proj(P, h), self._proj(P, t)
        self._PE = None    # 파라미터 변경 → 캐시 무효화
        s = np.sign(ph + P["R"][r] - pt) * coeff[:, None]      # (B, d)
        np.add.at(G["R"], r, s)
        # dE[h] += M^T s ;  dM[type] += s ⊗ E  (타입별 그룹 누적)
        for sign_, ents, types, svec in ((1.0, h, th, s), (-1.0, t, tt, s)):
            mt_s = np.empty_like(svec)
            for c in np.unique(types):
                m = types == c
                mt_s[m] = svec[m] @ P["M"][c]
                G["M"][c] += sign_ * svec[m].T @ P["E"][ents[m]]
            np.add.at(G["E"], ents, sign_ * mt_s)

    def normalize(self, P):
        _clip_norm(P["E"], 1.0)


# ───────────────────────── TransC (Lv et al., 2018 간소화) ─────────────────────────
class TransC:
    """클래스 = 구(sphere : 중심 + 반지름), 인스턴스 = 점.

    type 트리플  : f = max(0, ‖E[h] − E[t]‖₂ − rad[t])   (구 포함 제약)
    일반 트리플  : TransE 거리 ‖h + r − t‖₁
    """
    name = "TransC"
    distance_based = True

    def set_context(self, kg):
        self._type_rel = kg.rel2id.get("type", -1)

    def init_params(self, ne, nr, d, rng) -> Params:
        E = xavier(rng, (ne, d))
        R = xavier(rng, (nr, d))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), EPS)
        rad = np.full(ne, 0.5)
        return {"E": E, "R": R, "rad": rad}

    def energy(self, P, h, r, t):
        h = np.asarray(h); r = np.asarray(r); t = np.asarray(t)
        out = np.empty(len(h))
        m = r == self._type_rel
        if m.any():
            d2 = np.linalg.norm(P["E"][h[m]] - P["E"][t[m]], axis=1)
            out[m] = np.maximum(0.0, d2 - P["rad"][t[m]])
        if (~m).any():
            diff = P["E"][h[~m]] + P["R"][r[~m]] - P["E"][t[~m]]
            out[~m] = np.sum(np.abs(diff), axis=-1)
        return out

    def grad_accum(self, P, G, h, r, t, coeff):
        h = np.asarray(h); r = np.asarray(r); t = np.asarray(t)
        coeff = np.asarray(coeff, dtype=float)
        m = r == self._type_rel
        if m.any():
            hh, tt, cc = h[m], t[m], coeff[m]
            diff = P["E"][hh] - P["E"][tt]
            d2 = np.linalg.norm(diff, axis=1)
            act = (d2 - P["rad"][tt]) > 0
            g = diff / np.maximum(d2, EPS)[:, None] * (cc * act)[:, None]
            np.add.at(G["E"], hh, g)
            np.add.at(G["E"], tt, -g)
            np.add.at(G["rad"], tt, -(cc * act))
        if (~m).any():
            hh, rr, tt, cc = h[~m], r[~m], t[~m], coeff[~m]
            s = np.sign(P["E"][hh] + P["R"][rr] - P["E"][tt]) * cc[:, None]
            np.add.at(G["E"], hh, s)
            np.add.at(G["E"], tt, -s)
            np.add.at(G["R"], rr, s)

    def normalize(self, P):
        _clip_norm(P["E"], 1.0)
        np.clip(P["rad"], 0.05, 5.0, out=P["rad"])


REGISTRY["TKRL"] = TKRL
REGISTRY["TransC"] = TransC
