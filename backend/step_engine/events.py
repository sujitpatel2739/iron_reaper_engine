"""
events.py
---------
StepEvent and EventKind — pure data, zero logic, zero engine imports.

Every other module in execution/ imports from here.
Nothing here imports from anywhere in the project.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class EventKind(Enum):
    STEP_DONE         = auto()   # one layer completed forward or backward
    BRANCH_POINT      = auto()   # fork encountered — N branches available
    BRANCH_DONE       = auto()   # one branch completed
    BRANCHES_COMPLETE = auto()   # all branches done, main stream resumes
    FORWARD_COMPLETE  = auto()   # entire forward pass finished
    BACKWARD_COMPLETE = auto()   # entire backward pass finished
    ERROR             = auto()   # something went wrong


@dataclass
class StepEvent:
    kind:          EventKind
    layer_id:      Optional[str]       = None
    metrics:       Dict[str, Any]      = field(default_factory=dict)
    branches:      List[str]           = field(default_factory=list)
    branch:        Optional[str]       = None
    next_layer_id: Optional[str]       = None
    error:         Optional[str]       = None

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON transmission over WebSocket."""
        d: dict = {"event": self.kind.name.lower()}
        if self.layer_id      is not None: d["layer_id"]      = self.layer_id
        if self.metrics:                   d["metrics"]        = self.metrics
        if self.branches:                  d["branches"]       = self.branches
        if self.branch        is not None: d["branch"]         = self.branch
        if self.next_layer_id is not None: d["next_layer_id"]  = self.next_layer_id
        if self.error         is not None: d["message"]        = self.error
        return d