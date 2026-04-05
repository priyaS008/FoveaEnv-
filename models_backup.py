# models.py
# BlinkEnv - Privacy-Aware Vision Attention Environment

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from openenv.core.env_server import Action, Observation, State


@dataclass
class GazeAction(Action):
    direction: str = "focus"
    answer: Optional[str] = None


@dataclass
class GridObservation(Observation):
    visible_cells: List[str] = field(default_factory=list)
    gaze_position: Tuple[int, int] = (3, 3)
    target_found: bool = False
    reward: float = 0.0
    task: str = ""
    steps_taken: int = 0
    done: bool = False
    message: str = ""


@dataclass
class BlinkState(State):
    grid: List[List[str]] = field(default_factory=list)
    gaze_x: int = 3
    gaze_y: int = 3
    steps_taken: int = 0
    target_object: str = ""
    target_found: bool = False
    episode_id: str = "init"
    done: bool = False
    total_reward: float = 0.0
