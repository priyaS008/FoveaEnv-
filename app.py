# app.py
# BlinkEnv - FastAPI Server
# Run with: uvicorn app:app --reload

from openenv.core.env_server import create_fastapi_app
from models import GazeAction, GridObservation
from environment import BlinkEnvironment

# Create environment instance
env = BlinkEnvironment()

# Create FastAPI app (OpenEnv standard)
app = create_fastapi_app(env, GazeAction, GridObservation)

# Health check info
@app.get("/info")
def info():
    return env.get_metadata()
