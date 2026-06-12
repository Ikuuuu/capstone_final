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
