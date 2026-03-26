from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neo4j.time import DateTime


@dataclass
class Student:
    student_id: str
    name: str
    email: str
    enrolled_at: DateTime | None = None
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Student:
        return cls(
            student_id=node["student_id"],
            name=node["name"],
            email=node["email"],
            enrolled_at=node.get("enrolled_at"),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
        }
        if self.enrolled_at is not None:
            props["enrolled_at"] = self.enrolled_at
        return props
