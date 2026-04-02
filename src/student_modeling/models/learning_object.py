from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningObject:
    lo_id: str
    question: str
    answer: str
    fact: str = ""
    entities: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list, repr=False)
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> LearningObject:
        return cls(
            lo_id=node["lo_id"],
            question=node["question"],
            answer=node["answer"],
            fact=node.get("fact", ""),
            entities=node.get("entities", []),
            embedding=node.get("embedding", []),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "lo_id": self.lo_id,
            "question": self.question,
            "answer": self.answer,
            "fact": self.fact,
            "entities": self.entities,
        }
        if self.embedding:
            props["embedding"] = self.embedding
        return props
