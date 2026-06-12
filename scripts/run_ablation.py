"""Ablation Study : 본 연구 모델의 구성요소 단계적 제거 후 성능 측정."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import json
import subprocess
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import setup_logger


ABLATIONS = [
    # 이름, 변경할 키 경로, 값
    ("full",          [],                                                   None),
    ("no_L_dir",      [("losses", "weights", "delta_dir")],                 0.0),
    ("no_L_hier_attn", [("losses", "weights", "beta_hier")],                0.0),
    ("no_LM_init",    [("model", "lm_init")],                               False),
    ("no_OWL_Lite",   [("losses", "weights", "alpha_type"),
                        ("losses", "weights", "gamma_attr")],               0.0),
]


def set_nested(cfg, path, value):
    cur = cfg
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ours.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("kge.ablation", cfg["paths"]["logs"])

    summary = {}
    for name, paths, value in ABLATIONS:
        logger.info(f"===> Ablation : {name}")
        ab_cfg = copy.deepcopy(cfg)
        ab_cfg["experiment"]["name"] = f"ablation_{name}"
        for path in paths:
            set_nested(ab_cfg, path, value)

        # 임시 설정 저장
        tmp_cfg_path = Path(cfg["paths"]["experiments"]) / f"ablation_{name}.yaml"
        tmp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        with tmp_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(ab_cfg, f, allow_unicode=True)

        # 학습
        subprocess.run(
            ["python", "scripts/03_train.py", "--config", str(tmp_cfg_path)],
            check=False,
        )
        # 평가
        subprocess.run(
            ["python", "scripts/04_evaluate.py", "--config", str(tmp_cfg_path)],
            check=False,
        )

        # 결과 수집
        metrics_path = (
            Path(cfg["paths"]["experiments"])
            / f"ablation_{name}"
            / "metrics.json"
        )
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                summary[name] = json.load(f)

    # 종합 결과
    out = Path(cfg["paths"]["experiments"]) / "ablation_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Ablation 결과 종합 : {out}")


if __name__ == "__main__":
    main()
