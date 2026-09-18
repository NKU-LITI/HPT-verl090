"""Hybrid Post-Training (HPT) support for the verl PPO trainer.

The implementation is intentionally isolated from the rollout backend.  HPT
first evaluates normal on-policy rollouts and then replaces unsolved prompt
groups with their supervised target before log-probabilities are computed.
"""

from .algorithm import apply_hpt
from .dataset import HPTDataset

__all__ = ["HPTDataset", "apply_hpt"]
