from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Domain:
    domain_id: str
    name: str
    parent_id: str | None = None
    element_id: str | None = field(default=None, repr=False)

    @classmethod
    def from_node(cls, node: dict[str, Any]) -> Domain:
        return cls(
            domain_id=node["domain_id"],
            name=node["name"],
            parent_id=node.get("parent_id"),
            element_id=node.get("element_id"),
        )

    def to_properties(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "domain_id": self.domain_id,
            "name": self.name,
        }
        if self.parent_id is not None:
            props["parent_id"] = self.parent_id
        return props
