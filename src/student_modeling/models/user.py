from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neo4j.time import DateTime


@dataclass
class User:
    user_id: str
    name: str
    email: str
    created_at: DateTime | None = None
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> User:
        return cls(
            user_id=node["user_id"],
            name=node["name"],
            email=node.get("email", ""),
            created_at=node.get("created_at"),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
        }
