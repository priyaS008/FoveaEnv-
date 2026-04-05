# app.py
# BlinkEnv - FastAPI Server
# Run with: uvicorn app:app --reload

import importlib
from typing import Any, Dict, Optional

from environment import BlinkEnvironment
from models import GazeAction, GridObservation

# Create environment instance
env = BlinkEnvironment()

def _as_dict(observation: GridObservation) -> Dict[str, Any]:
    return {
        "visible_cells": observation.visible_cells,
        "gaze_position": observation.gaze_position,
        "target_found": observation.target_found,
        "reward": observation.reward,
        "task": observation.task,
        "steps_taken": observation.steps_taken,
        "done": observation.done,
        "message": observation.message,
    }

# Try to create an OpenEnv FastAPI app if available.
create_fastapi_app = None
try:
    openenv = importlib.import_module("openenv.core.env_server")
    create_fastapi_app = getattr(openenv, "create_fastapi_app", None)
except ImportError:
    create_fastapi_app = None

if create_fastapi_app is not None:
    app = create_fastapi_app(env, GazeAction, GridObservation)
else:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    FastAPI = None
    BaseModel = None
    try:
        fastapi = importlib.import_module("fastapi")
        pydantic = importlib.import_module("pydantic")
        FastAPI = getattr(fastapi, "FastAPI", None)
        HTTPException = getattr(fastapi, "HTTPException", HTTPException)
        BaseModel = getattr(pydantic, "BaseModel", None)
    except ImportError:
        pass

    if FastAPI is not None and BaseModel is not None:
        app = FastAPI(title="BlinkEnv")

        class GazeActionPayload(BaseModel):
            direction: str
            answer: Optional[str] = None

        @app.get("/info")
        def info() -> Dict[str, Any]:
            return env.get_metadata()

        @app.post("/reset")
        def reset_environment() -> Dict[str, Any]:
            observation = env.reset()
            return _as_dict(observation)

        @app.post("/step")
        def step(action: GazeActionPayload) -> Dict[str, Any]:
            if env.state.done:
                raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

            gaze_action = GazeAction(direction=action.direction, answer=action.answer)
            observation = env.step(gaze_action)
            return _as_dict(observation)
    else:
        app = None
