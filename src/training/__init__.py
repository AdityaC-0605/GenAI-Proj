"""Training module for Cross-Lingual QA System."""

from src.training.zero_shot_trainer import ZeroShotTrainer
from src.training.few_shot_trainer import FewShotTrainer

__all__ = ['ZeroShotTrainer', 'FewShotTrainer']
