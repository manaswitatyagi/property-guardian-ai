from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Shared contract: one responsibility, typed inputs, a structured output."""

    name: str = "base-agent"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's intended analysis."""


def bounded_score(value: float) -> float:
    """Clamp a heuristic index to the 0-100 scale used across agents."""
    return round(max(0.0, min(100.0, value)), 2)
