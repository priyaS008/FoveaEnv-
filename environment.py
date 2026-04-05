import importlib
import uuid
import random
from typing import List

class Environment:
    """Fallback stub Environment when openenv is not installed."""
    def __init__(self):
        pass

try:
    _openenv = importlib.import_module("openenv.core.env_server")
    Environment = _openenv.Environment
except ImportError:
    pass

from models import GazeAction, GridObservation, BlinkState

# --- Constants ---
GRID_SIZE = 7          # 7x7 grid world
VISION_RADIUS = 1      # agent sees 3x3 area (radius=1 around gaze)
MAX_STEPS = 30         # max steps per episode

# Objects placed in the grid
ALL_OBJECTS = [
    "cup", "phone", "book", "keys", "bottle",
    "glasses", "wallet", "pen", "watch", "bag"
]

# Reward values
REWARD_FOUND = 10.0       # found target!
REWARD_STEP = -0.1        # each step costs a little
REWARD_WRONG_ANSWER = -2.0  # wrong answer penalty
REWARD_TIMEOUT = -5.0     # ran out of steps


class BlinkEnvironment(Environment):
    """
    BlinkEnv: Privacy-Aware Vision Attention Environment

    Simulates an AI agent on a smart wearable device (like Meta Ray-Ban glasses)
    that cannot process the full scene due to battery/compute constraints.

    The agent must learn WHERE to look (gaze control) to find a target object
    using the minimum number of observations possible.

    This trains 'attention as a skill' - the core challenge for edge AI.
    """

    def __init__(self):
        super().__init__()
        self._state = BlinkState(grid=[], gaze_x=3, gaze_y=3, steps_taken=0, target_object="none", target_found=False, episode_id="init", done=False, total_reward=0.0)

    # ------------------------------------------------------------------ #
    # RESET — start a new episode
    # ------------------------------------------------------------------ #
    def reset(self) -> GridObservation:
        # Build empty 7x7 grid
        grid = [["empty"] * GRID_SIZE for _ in range(GRID_SIZE)]

        # Pick 5 random unique positions for objects
        all_positions = [
            (r, c)
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
        ]
        chosen_positions = random.sample(all_positions, 5)
        chosen_objects = random.sample(ALL_OBJECTS, 5)

        for obj, (r, c) in zip(chosen_objects, chosen_positions):
            grid[r][c] = obj

        # Pick one object as the target
        target = random.choice(chosen_objects)

        # Start gaze at center
        start_x, start_y = GRID_SIZE // 2, GRID_SIZE // 2

        self._state = BlinkState(
            grid=grid,
            gaze_x=start_x,
            gaze_y=start_y,
            steps_taken=0,
            target_object=target,
            target_found=False,
            episode_id=str(uuid.uuid4()),
            done=False,
            total_reward=0.0
        )

        return self._build_observation(
            reward=0.0,
            message=f"Find the '{target}'. You can only see a 3x3 area at a time."
        )

    # ------------------------------------------------------------------ #
    # STEP — agent takes one action
    # ------------------------------------------------------------------ #
    def step(self, action: GazeAction) -> GridObservation:
        if self._state.done:
            return self._build_observation(
                reward=0.0,
                message="Episode already done. Call reset()."
            )

        self._state.steps_taken += 1
        reward = REWARD_STEP  # base step cost

        # --- Handle ANSWER action ---
        if action.direction == "answer":
            if action.answer == self._state.target_object:
                reward = REWARD_FOUND
                self._state.target_found = True
                self._state.done = True
                self._state.total_reward += reward
                return self._build_observation(
                    reward=reward,
                    message=f"CORRECT! Found '{action.answer}' in {self._state.steps_taken} steps!"
                )
            else:
                reward = REWARD_WRONG_ANSWER
                self._state.total_reward += reward
                return self._build_observation(
                    reward=reward,
                    message=f"WRONG! '{action.answer}' is not the target. Keep searching."
                )

        # --- Handle GAZE MOVEMENT ---
        move_map = {
            "up":    (-1, 0),
            "down":  (1,  0),
            "left":  (0, -1),
            "right": (0,  1),
            "focus": (0,  0),   # stay and observe
        }

        if action.direction in move_map:
            dx, dy = move_map[action.direction]
            new_x = max(0, min(GRID_SIZE - 1, self._state.gaze_x + dx))
            new_y = max(0, min(GRID_SIZE - 1, self._state.gaze_y + dy))
            self._state.gaze_x = new_x
            self._state.gaze_y = new_y

        # Check if target is visible in current window
        visible = self._get_visible_cells()
        target_visible = self._state.target_object in visible

        if target_visible:
            message = f"You can see the '{self._state.target_object}'! Use action='answer' to confirm."
        else:
            message = f"Target not visible here. Keep searching. Steps left: {MAX_STEPS - self._state.steps_taken}"

        # Timeout check
        if self._state.steps_taken >= MAX_STEPS:
            reward += REWARD_TIMEOUT
            self._state.done = True
            message = f"Out of steps! Target was '{self._state.target_object}'."

        self._state.total_reward += reward

        return self._build_observation(reward=reward, message=message)

    # ------------------------------------------------------------------ #
    # STATE — return current episode metadata
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> BlinkState:
        return self._state

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #
    def _get_visible_cells(self) -> List[str]:
        """Return cells visible in the 3x3 focus window."""
        cells: List[str] = []
        gx, gy = self._state.gaze_x, self._state.gaze_y
        for r in range(gx - VISION_RADIUS, gx + VISION_RADIUS + 1):
            for c in range(gy - VISION_RADIUS, gy + VISION_RADIUS + 1):
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    cells.append(self._state.grid[r][c])
                else:
                    cells.append("wall")  # out of bounds
        return cells

    def _build_observation(self, reward: float, message: str) -> GridObservation:
        return GridObservation(
            visible_cells=self._get_visible_cells(),
            gaze_position=(self._state.gaze_x, self._state.gaze_y),
            target_found=self._state.target_found,
            reward=reward,
            task=f"Find the '{self._state.target_object}'",
            steps_taken=self._state.steps_taken,
            done=self._state.done,
            message=message
        )

    def get_metadata(self) -> dict[str, object]:
        return {
            "name": "BlinkEnv",
            "version": "1.0.0",
            "description": (
                "Privacy-aware vision attention environment. "
                "Agent sees only 3x3 of a 7x7 grid and must find a target "
                "object using minimum gaze movements. "
                "Inspired by compute constraints on Meta Ray-Ban smart glasses."
            ),
            "actions": ["up", "down", "left", "right", "focus", "answer"],
            "grid_size": GRID_SIZE,
            "vision_radius": VISION_RADIUS,
            "max_steps": MAX_STEPS,
            "reward_found": REWARD_FOUND,
            "reward_step": REWARD_STEP,
        }
