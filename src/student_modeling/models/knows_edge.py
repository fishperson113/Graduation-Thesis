from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class KnowsEdge:
    """Represents the KNOWS relationship between User and Concept.

    Carries GAM-RAG memory properties: task memory vector, perplexities,
    derived mastery score, and interaction metadata.
    """

    user_id: str
    concept_id: str
    m_task: list[float]
    m_time_last: datetime
    pi_task: float
    pi_time: float
    mastery: float
    attempts: int

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> KnowsEdge:
        m_time_last = record["m_time_last"]
        if hasattr(m_time_last, "to_native"):
            m_time_last = m_time_last.to_native()
        return cls(
            user_id=record["user_id"],
            concept_id=record["concept_id"],
            m_task=record.get("m_task", []),
            m_time_last=m_time_last,
            pi_task=record.get("pi_task", 1.0),
            pi_time=record.get("pi_time", 1.0),
            mastery=record.get("mastery", 0.0),
            attempts=record.get("attempts", 0),
        )

    def to_properties(self) -> dict[str, Any]:
        return {
            "m_task": self.m_task,
            "m_time_last": self.m_time_last,
            "pi_task": self.pi_task,
            "pi_time": self.pi_time,
            "mastery": self.mastery,
            "attempts": self.attempts,
        }
