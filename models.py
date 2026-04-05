from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import json

# Removed openenv imports to avoid errors; using standalone dataclasses

VALID_DIRECTIONS = ["up", "down", "left", "right", "focus", "answer", "zoom_in", "zoom_out", "scan"]

@dataclass
class GazeAction:
    direction: str = "focus"
    answer: Optional[str] = None
    confidence: float = 1.0  # Confidence in the action

    def __post_init__(self):
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {self.direction}. Must be one of {VALID_DIRECTIONS}")
        if self.direction == "answer" and self.answer is None:
            raise ValueError("Answer must be provided when direction is 'answer'")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

@dataclass
class GridObservation:
    visible_cells: List[str] = field(default_factory=list)  # type: ignore
    gaze_position: Tuple[int, int] = (3, 3)
    target_found: bool = False
    reward: float = 0.0
    task: str = ""
    steps_taken: int = 0
    done: bool = False
    message: str = ""
    attention_map: List[List[float]] = field(default_factory=lambda: [[0.0] * 3 for _ in range(3)])  # 3x3 attention weights
    confidence: float = 0.0  # Confidence in observation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visible_cells": self.visible_cells,
            "gaze_position": list(self.gaze_position),
            "target_found": self.target_found,
            "reward": self.reward,
            "task": self.task,
            "steps_taken": self.steps_taken,
            "done": self.done,
            "message": self.message,
            "attention_map": self.attention_map,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

@dataclass
class BlinkState:
    grid: List[List[str]] = field(default_factory=list)  # type: ignore
    gaze_x: int = 3
    gaze_y: int = 3
    steps_taken: int = 0
    target_object: str = "none"
    target_found: bool = False
    episode_id: str = "init"
    done: bool = False
    total_reward: float = 0.0
    history: List[GridObservation] = field(default_factory=list)  # type: ignore # History of observations
    max_history: int = 10  # Max history length

    def add_observation(self, obs: GridObservation) -> None:
        self.history.append(obs)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_observations(self, n: int = 5) -> List[GridObservation]:
        return self.history[-n:]

    def reset_history(self) -> None:
        self.history.clear()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid": self.grid,
            "gaze_x": self.gaze_x,
            "gaze_y": self.gaze_y,
            "steps_taken": self.steps_taken,
            "target_object": self.target_object,
            "target_found": self.target_found,
            "episode_id": self.episode_id,
            "done": self.done,
            "total_reward": self.total_reward,
            "history": [obs.to_dict() for obs in self.history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

# Utility functions
def create_default_grid(size: int = 7) -> List[List[str]]:
    return [["empty"] * size for _ in range(size)]

def validate_grid(grid: List[List[str]]) -> bool:
    if not grid:
        return False
    size = len(grid)
    for row in grid:
        if len(row) != size:
            return False
    return True
