from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j.time import DateTime


@dataclass
class KnowledgeState:
    """Represents a Student's mastery of a Skill.

    Maps to the [:HAS_KNOWLEDGE] relationship between (:Student) and (:Skill).
    """

    student_id: str
    skill_id: str
    mastery_level: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    last_assessed_at: DateTime | None = None
    attempts: int = 0

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> KnowledgeState:
        return cls(
            student_id=record["student_id"],
            skill_id=record["skill_id"],
            mastery_level=record.get("mastery_level", 0.0),
            confidence=record.get("confidence", 0.0),
            last_assessed_at=record.get("last_assessed_at"),
            attempts=record.get("attempts", 0),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "mastery_level": self.mastery_level,
            "confidence": self.confidence,
            "attempts": self.attempts,
        }
        if self.last_assessed_at is not None:
            props["last_assessed_at"] = self.last_assessed_at
        return props
