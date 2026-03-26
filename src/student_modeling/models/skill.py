from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Skill:
    skill_id: str
    name: str
    category: str = ""
    description: str = ""
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Skill:
        return cls(
            skill_id=node["skill_id"],
            name=node["name"],
            category=node.get("category", ""),
            description=node.get("description", ""),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
        }
