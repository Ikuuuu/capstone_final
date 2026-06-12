"""L_type, L_attr : 온톨로지 제약 손실 (기존 TransO 계열).

L_type : 관계의 domain/range 위반 패널티
L_attr : 대칭성·역관계 등 관계 속성 반영
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TypeLoss(nn.Module):
    """관계별 domain/range 위반 패널티 (L_type)."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        h_emb: torch.Tensor,        # (B, d)
        t_emb: torch.Tensor,        # (B, d)
        expected_h_type_emb: torch.Tensor,  # (B, d) 관계의 domain 타입 임베딩
        expected_t_type_emb: torch.Tensor,  # (B, d) 관계의 range 타입 임베딩
    ) -> torch.Tensor:
        # head 가 domain 타입 클래스 임베딩에 가깝도록
        head_dev = torch.norm(h_emb - expected_h_type_emb, p=2, dim=1)
        tail_dev = torch.norm(t_emb - expected_t_type_emb, p=2, dim=1)
        loss = torch.clamp(head_dev - self.margin, min=0) + torch.clamp(
            tail_dev - self.margin, min=0
        )
        return loss.mean()


class AttrLoss(nn.Module):
    """관계 속성 (대칭성·역관계) 반영 손실 (L_attr)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        scoring_fn,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        symmetric_relation_ids: set | None = None,
        inverse_pairs: Dict[int, int] | None = None,
    ) -> torch.Tensor:
        """대칭 관계는 f(h,r,t) ≈ f(t,r,h),
        역관계 pair (r, r_inv) 는 f(h,r,t) ≈ f(t,r_inv,h) 가 되도록 강제."""
        loss = torch.zeros(1, device=h.device)
        count = 0

        if symmetric_relation_ids:
            sym = torch.tensor(list(symmetric_relation_ids), device=r.device, dtype=r.dtype)
            sym_mask = torch.isin(r, sym)
            if sym_mask.any():
                f_fw = scoring_fn(h[sym_mask], r[sym_mask], t[sym_mask])
                f_bw = scoring_fn(t[sym_mask], r[sym_mask], h[sym_mask])
                loss = loss + (f_fw - f_bw).pow(2).mean()
                count += 1

        if inverse_pairs:
            inv_h, inv_r, inv_t, inv_r_inv = [], [], [], []
            r_list = r.tolist()
            for i, r_i in enumerate(r_list):
                if r_i in inverse_pairs:
                    inv_h.append(h[i].item())
                    inv_r.append(r_i)
                    inv_t.append(t[i].item())
                    inv_r_inv.append(inverse_pairs[r_i])
            if inv_h:
                ih = torch.tensor(inv_h, device=h.device, dtype=h.dtype)
                ir = torch.tensor(inv_r, device=r.device, dtype=r.dtype)
                it = torch.tensor(inv_t, device=t.device, dtype=t.dtype)
                ir_inv = torch.tensor(inv_r_inv, device=r.device, dtype=r.dtype)
                f_fw = scoring_fn(ih, ir, it)
                f_inv = scoring_fn(it, ir_inv, ih)
                loss = loss + (f_fw - f_inv).pow(2).mean()
                count += 1

        return loss / max(count, 1)
