from __future__ import annotations

from typing import Any
import numpy as np


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (int, float, np.number)):
        return np.isfinite(value)
    return False


def select_history_indices(message_history: list[dict], timestep: int, history_len: int,
                           k_best: int, include_recency: bool = True) -> dict:
    """
    Returns dict with keys:
      - 'recency': list[int]  # indices of recent timesteps (subset)
      - 'best': list[int]     # indices of best timesteps by metric (subset)
    """
    recency = []
    max_index = min(timestep, len(message_history) - 1)
    if include_recency and max_index >= 0:
        start = max(0, max_index - history_len)
        recency = list(range(start, max_index + 1))

    best = []
    if k_best and k_best > 0:
        last_index = min(timestep - 1, len(message_history) - 1)
        candidates = []
        for idx in range(0, last_index + 1):
            metric = message_history[idx].get('metric')
            if _is_finite_number(metric):
                candidates.append((idx, float(metric)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = [idx for idx, _ in candidates[:k_best]]

    return {'recency': recency, 'best': best}
