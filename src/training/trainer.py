"""KGE 학습 루프 (본 연구 모델 + 베이스라인 공용, PyTorch 참조 구현).

L_total = L_struct + λ · (α·L_type + β·L_hier_attn + γ·L_attr + δ·L_dir)

개선 사항 (v2):
  - L_type   : schema 의 relation domain/range → 클래스 임베딩 거리 패널티 (실구현)
  - L_hier_attn : 계층 관계(상위/하위법령) 트리플을 head 클래스별로 묶어
                  클래스별 손실 → attention 가중합 (실구현)
  - 클래스 인덱스는 KG 의 type 트리플에서 자동 구축
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from ..data.dataset import KGData, TripleDataset
from ..data.ontology import OntologySchema
from ..losses import (
    AttrLoss,
    DirectionLoss,
    HierarchyAttentionLoss,
    StructLoss,
    TypeLoss,
)
from ..utils.checkpoint import save_checkpoint
from .curriculum import lambda_curriculum

logger = logging.getLogger("kge.training.trainer")


def extract_class_info(kg: KGData) -> Tuple[List[int], Dict[int, int]]:
    """KG 의 type 트리플에서 (클래스 엔티티 id 목록, 엔티티→클래스idx) 추출."""
    type_rel = kg.relation2id.get("type")
    class_ents: List[int] = []
    ent2cls: Dict[int, int] = {}
    if type_rel is None:
        return class_ents, ent2cls
    class_ents = sorted({t for h, r, t in kg.train_triples if r == type_rel})
    cls_idx = {c: i for i, c in enumerate(class_ents)}
    for h, r, t in kg.train_triples:
        if r == type_rel and t in cls_idx:
            ent2cls[h] = cls_idx[t]
    return class_ents, ent2cls


class Trainer:
    """4종 손실의 가중합을 최적화하는 학습 루프."""

    def __init__(
        self,
        model: torch.nn.Module,
        kg: KGData,
        schema: OntologySchema | None,
        cfg: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.kg = kg
        self.schema = schema
        self.cfg = cfg
        self.device = device

        # ── 클래스 정보 (type 트리플 기반) ──
        class_ents, ent2cls = extract_class_info(kg)
        self.class_ents = class_ents
        self.num_classes = max(1, len(class_ents))
        ent_cls = torch.full((kg.num_entities,), -1, dtype=torch.long)
        for e, c in ent2cls.items():
            ent_cls[e] = c
        self.entity_class = ent_cls.to(device)            # (E,) 클래스 idx or -1
        # 클래스명 → 클래스 idx (schema domain/range 매핑용)
        id2ent = {v: k for k, v in kg.entity2id.items()}
        self.classname2idx = {id2ent[c]: i for i, c in enumerate(class_ents)}

        # ── 관계 → (domain cls idx, range cls idx) ──
        self.rel_domain = torch.full((kg.num_relations,), -1, dtype=torch.long)
        self.rel_range = torch.full((kg.num_relations,), -1, dtype=torch.long)
        if schema is not None:
            for rname, rid in kg.relation2id.items():
                d = schema.relation_domain.get(rname)
                g = schema.relation_range.get(rname)
                if d in self.classname2idx:
                    self.rel_domain[rid] = self.classname2idx[d]
                if g in self.classname2idx:
                    self.rel_range[rid] = self.classname2idx[g]
        self.rel_domain = self.rel_domain.to(device)
        self.rel_range = self.rel_range.to(device)

        # ── 계층 관계 id (L_hier_attn 대상 : 상위/하위법령 등 transitive) ──
        hier_names = (schema.transitive_relations if schema else set()) or set()
        self.hier_rel_ids = torch.tensor(
            sorted(kg.relation2id[n] for n in hier_names if n in kg.relation2id),
            dtype=torch.long, device=device,
        )

        # ── 손실 함수 ──
        self.l_struct = StructLoss(margin=cfg["model"]["margin"])
        self.l_type = TypeLoss()
        self.l_attr = AttrLoss()

        is_ours = cfg["model"]["name"] == "ours"
        self.is_ours = is_ours
        if is_ours:
            self.l_dir = DirectionLoss(
                margin_dir=cfg["losses"]["l_dir"]["margin_dir"],
                skip_symmetric=cfg["losses"]["l_dir"]["skip_symmetric"],
            )
            self.l_hier_attn = HierarchyAttentionLoss(
                num_classes=self.num_classes,
                embedding_dim=cfg["model"]["embedding_dim"],
                attention_hidden_dim=cfg["losses"]["l_hier_attn"]["attention_hidden_dim"],
                initial_uniform_epochs=cfg["losses"]["l_hier_attn"]["initial_uniform_epochs"],
            ).to(device)

        params = list(self.model.parameters())
        if is_ours:
            params += list(self.l_hier_attn.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )

        self.train_dataset = TripleDataset(
            triples=kg.train_triples,
            num_entities=kg.num_entities,
            num_negatives=cfg["training"]["negative_sampling"]["num_negatives"],
            filter_triples=set(kg.all_triples),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
        )

    # ────────────────────── 손실 계산 ──────────────────────
    def _compute_loss(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        epoch: int,
    ) -> Dict[str, torch.Tensor]:
        pos_score = self.model.score(h, r, t)  # (B,)

        # 부정 샘플링
        B = h.size(0)
        K = self.cfg["training"]["negative_sampling"]["num_negatives"]
        neg_h = h.repeat_interleave(K)
        neg_r = r.repeat_interleave(K)
        neg_t = t.repeat_interleave(K)
        mask = torch.rand(B * K, device=h.device) < 0.5
        random_ents = torch.randint(0, self.kg.num_entities, (B * K,), device=h.device)
        neg_h_ = torch.where(mask, random_ents, neg_h)
        neg_t_ = torch.where(mask, neg_t, random_ents)
        neg_score = self.model.score(neg_h_, neg_r, neg_t_).view(B, K)

        losses: Dict[str, torch.Tensor] = {}
        losses["L_struct"] = self.l_struct(pos_score, neg_score)

        if not self.is_ours:
            return losses

        score_fn = lambda hh, rr, tt: self.model.score(hh, rr, tt)
        sym_ids = self._symmetric_relation_ids()

        # ── L_dir (기여 ①) ──
        losses["L_dir"] = self.l_dir(score_fn, h, r, t, sym_ids)

        # ── L_attr (대칭 + inverseOf) ──
        losses["L_attr"] = self.l_attr(
            score_fn, h, r, t,
            symmetric_relation_ids=sym_ids,
            inverse_pairs=self._inverse_pair_ids(),
        )

        # ── L_type (domain/range 위반 패널티, 실구현) ──
        cls_emb = self.model.get_class_embeddings()           # (C, d)
        dom = self.rel_domain[r]                              # (B,)
        rng = self.rel_range[r]
        valid = (dom >= 0) & (rng >= 0)
        if valid.any():
            hv, tv = h[valid], t[valid]
            losses["L_type"] = self.l_type(
                self.model.entity_emb(hv),
                self.model.entity_emb(tv),
                cls_emb[dom[valid]],
                cls_emb[rng[valid]],
            )
        else:
            losses["L_type"] = torch.tensor(0.0, device=h.device)

        # ── L_hier_attn (기여 ②, 실구현) ──
        # 계층 관계 트리플을 head 클래스별로 묶어 클래스별 평균 score → attention 가중합
        per_class = torch.zeros(self.num_classes, device=h.device)
        if self.hier_rel_ids.numel() > 0:
            hm = torch.isin(r, self.hier_rel_ids)
            if hm.any():
                hh, rr_, tt = h[hm], r[hm], t[hm]
                sc = self.model.score(hh, rr_, tt)            # (M,) 거리(작을수록 좋음)
                cls = self.entity_class[hh]                   # (M,) head 클래스
                for c in cls.unique():
                    if int(c) < 0:
                        continue
                    per_class[int(c)] = sc[cls == c].mean()
        losses["L_hier_attn"] = self.l_hier_attn(per_class, cls_emb, current_epoch=epoch)

        return losses

    def _symmetric_relation_ids(self) -> set:
        if self.schema is None:
            return set()
        return {
            self.kg.relation2id[r]
            for r in self.schema.symmetric_relations
            if r in self.kg.relation2id
        }

    def _inverse_pair_ids(self) -> Dict[int, int]:
        if self.schema is None:
            return {}
        out: Dict[int, int] = {}
        for r, r_inv in self.schema.inverse_pairs.items():
            if r in self.kg.relation2id and r_inv in self.kg.relation2id:
                out[self.kg.relation2id[r]] = self.kg.relation2id[r_inv]
        return out

    # ────────────────────── 학습 루프 ──────────────────────
    def train(self) -> Dict[str, Any]:
        epochs = self.cfg["training"]["epochs"]
        ck_dir = Path(self.cfg["paths"]["checkpoints"]) / self.cfg["experiment"]["name"]
        ck_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.model.train()
            total = {"L_struct": 0.0, "L_dir": 0.0, "L_attr": 0.0,
                     "L_type": 0.0, "L_hier_attn": 0.0}
            n_batches = 0

            for batch in self.train_loader:
                h = batch["h"].to(self.device)
                r = batch["r"].to(self.device)
                t = batch["t"].to(self.device)
                losses = self._compute_loss(h, r, t, epoch)

                weights = self.cfg.get("losses", {}).get("weights", {})
                if self.is_ours:
                    L_onto = (
                        weights.get("alpha_type", 0.5) * losses["L_type"]
                        + weights.get("beta_hier", 0.5) * losses["L_hier_attn"]
                        + weights.get("gamma_attr", 0.5) * losses["L_attr"]
                        + weights.get("delta_dir", 1.0) * losses["L_dir"]
                    )
                    if self.cfg["training"]["curriculum"]["enable"]:
                        lam = lambda_curriculum(
                            epoch,
                            self.cfg["training"]["curriculum"]["lambda_start"],
                            self.cfg["training"]["curriculum"]["lambda_end"],
                            self.cfg["training"]["curriculum"]["schedule_epochs"],
                        )
                    else:
                        lam = 1.0
                    total_loss = losses["L_struct"] + lam * L_onto
                else:
                    total_loss = losses["L_struct"]

                self.optimizer.zero_grad()
                total_loss.backward()
                # 안정화 : 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    max_norm=self.cfg["training"].get("grad_clip", 5.0),
                )
                self.optimizer.step()

                for k in total:
                    if k in losses:
                        total[k] += float(losses[k])
                n_batches += 1

            avg = {k: v / max(n_batches, 1) for k, v in total.items()}
            logger.info(f"Epoch {epoch + 1}/{epochs}  " +
                        "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

            if (epoch + 1) % self.cfg["training"]["checkpoint"]["every_n_epochs"] == 0:
                save_checkpoint(
                    ck_dir / f"epoch_{epoch + 1}.pt",
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    metrics=avg,
                )

        save_checkpoint(ck_dir / "best.pt", self.model, self.optimizer, epoch=epochs - 1)
        logger.info(f"학습 완료 - 체크포인트: {ck_dir}")
        return {"final_checkpoint": str(ck_dir / "best.pt")}
