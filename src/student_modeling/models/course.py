from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Course:
    course_id: str
    title: str
    description: str = ""
    difficulty_level: int = 1
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Course:
        return cls(
            course_id=node["course_id"],
            title=node["title"],
            description=node.get("description", ""),
            difficulty_level=node.get("difficulty_level", 1),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        return {
            "course_id": self.course_id,
            "title": self.title,
            "description": self.description,
            "difficulty_level": self.difficulty_level,
        }
