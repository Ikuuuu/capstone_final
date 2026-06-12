"""λ Curriculum : 학습 진행에 따라 L_onto 의 가중치를 점진적으로 증가."""
from __future__ import annotations


def lambda_curriculum(
    epoch: int,
    lambda_start: float = 0.0,
    lambda_end: float = 1.0,
    schedule_epochs: int = 100,
) -> float:
    """선형 보간으로 λ 값을 반환.

    Args:
        epoch: 현재 epoch (0-based)
        lambda_start: 시작 λ 값
        lambda_end: 종료 λ 값
        schedule_epochs: lambda_end 에 도달하는 epoch
    """
    if epoch >= schedule_epochs:
        return lambda_end
    frac = epoch / max(schedule_epochs, 1)
    return lambda_start + (lambda_end - lambda_start) * frac
