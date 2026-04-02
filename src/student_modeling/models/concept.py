from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Concept:
    concept_id: str
    name: str
    domain_id: str
    description: str = ""
    embedding: list[float] = field(default_factory=list, repr=False)
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Concept:
        return cls(
            concept_id=node["concept_id"],
            name=node["name"],
            domain_id=node.get("domain_id", ""),
            description=node.get("description", ""),
            embedding=node.get("embedding", []),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "concept_id": self.concept_id,
            "name": self.name,
            "domain_id": self.domain_id,
            "description": self.description,
        }
        if self.embedding:
            props["embedding"] = self.embedding
        return props
