"""
FastAPI Backend for Smart City RL Navigation System
Provides endpoints for model inference and environment interaction
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import numpy as np
import os
from datetime import datetime

from rl_navigation_env import NavigationEnvironment
from dqn_agent import DQNAgent

# Global variables
env: Optional[NavigationEnvironment] = None
agent: Optional[DQNAgent] = None
current_state = None
current_episode_path: List[List[int]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global env, agent

    print("Starting Smart City RL Navigation API.")

    # Initializing environment
    env = NavigationEnvironment(grid_size=15, dynamic_traffic=True)

    # Loading trained agent 
    model_path = None
    if os.path.exists("models/dqn_best.zip"):
        model_path = "models/dqn_best.zip"
    elif os.path.exists("models/dqn_best.pth"):
        model_path = "models/dqn_best.pth"

    if model_path:
        try:
            agent = DQNAgent(
                state_shape=env.state_shape,
                action_size=env.action_space_n,
            )
            agent.load_model(model_path)
            agent.epsilon = 0.0  # No exploration during inference
            print(f"Loaded trained model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Agent will use random actions")
            agent = None
    else:
        print("Model not found at models/dqn_best.zip or models/dqn_best.pth")
        print("Please train a model first using train_dqn.py")
        agent = None

    print("API ready!")

    yield

    # Shutdown (cleanup if needed)
    print("Shutting down API.")


app = FastAPI(
    title="Smart City RL Navigation API",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class EnvironmentConfig(BaseModel):
    grid_size: int = 15
    dynamic_traffic: bool = True


class NavigationRequest(BaseModel):
    start_pos: Optional[List[int]] = None
    goal_pos: Optional[List[int]] = None
    grid_size: int = 15
    dynamic_traffic: bool = True


class ActionRequest(BaseModel):
    action: int


class StepResponse(BaseModel):
    state: Dict
    reward: float
    done: bool
    info: Dict
    agent_pos: List[int]
    path: List[List[int]]


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Smart City RL Navigation API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": agent is not None,
    }


@app.post("/reset")
async def reset_environment(config: EnvironmentConfig):
    """Reset the environment with optional configuration."""
    global env, current_state, current_episode_path

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        # Reinitialize environment if grid size changed
        if config.grid_size != env.grid_size:
            env = NavigationEnvironment(
                grid_size=config.grid_size,
                dynamic_traffic=config.dynamic_traffic,
            )

        # Resetting environment (new layout, traffic, start, goal)
        state, info = env.reset()
        current_state = state
        current_episode_path = [env.agent_pos.tolist()]

        grid_state = env.get_grid_state()

        return {
            "message": "Environment reset successfully",
            "grid_state": grid_state,
            "info": info,
            "state_shape": list(state.shape),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/step")
async def step_environment(action_req: ActionRequest):
    """Take a single step in the environment."""
    global current_state, current_episode_path

    if current_state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first",
        )

    try:
        next_state, reward, terminated, truncated, info = env.step(action_req.action)
        done = terminated or truncated

        current_state = next_state
        current_episode_path.append(env.agent_pos.tolist())

        grid_state = env.get_grid_state()

        return {
            "grid_state": grid_state,
            "reward": float(reward),
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "path": current_episode_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/navigate")
async def navigate_with_rl(nav_req: NavigationRequest):
    """
    Complete navigation from start to goal using trained RL agent.
    Returns the full path and statistics.
    """
    global env, agent, current_state, current_episode_path

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Please train a model first.",
        )

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        # Reinitializing environment if needed (grid size changed)
        if nav_req.grid_size != env.grid_size:
            env = NavigationEnvironment(
                grid_size=nav_req.grid_size,
                dynamic_traffic=nav_req.dynamic_traffic,
            )

        # Always resetting environment, but we will override start/goal
        state, info = env.reset()

        # If start_pos / goal_pos are present, use them instead of random.
        if nav_req.start_pos is not None:
            env.agent_pos = np.array(nav_req.start_pos, dtype=int)
        if nav_req.goal_pos is not None:
            env.goal_pos = np.array(nav_req.goal_pos, dtype=int)

        # Saving the actual starting positions we used for this episode
        start_pos = env.agent_pos.copy()
        goal_pos = env.goal_pos.copy()

        # Refreshing state after overriding positions
        state = env._get_state()
        current_state = state
        current_episode_path = [env.agent_pos.tolist()]

        # Running episode
        episode_reward = 0.0
        steps = 0
        max_steps = 200
        done = False

        trajectory = []

        while not done and steps < max_steps:
            # Selecting action using trained agent
            action = agent.select_action(state, training=False)

            # Taking step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trajectory.append(
                {
                    "step": steps,
                    "position": env.agent_pos.tolist(),
                    "action": int(action),
                    "reward": float(reward),
                }
            )

            state = next_state
            episode_reward += reward
            steps += 1
            current_episode_path.append(env.agent_pos.tolist())

        grid_state = env.get_grid_state()

        return {
            "success": bool(terminated),
            "grid_state": grid_state,
            "path": current_episode_path,
            "trajectory": trajectory,
            "total_reward": float(episode_reward),
            "steps": steps,
            # Return to the actual start/goal used for this episode
            "start_pos": start_pos.tolist(),
            "goal_pos": goal_pos.tolist(),
            "info": info,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/stats")
async def get_model_stats():
    """Get training statistics of the loaded model."""
    if agent is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        stats = agent.get_stats()
        return {
            "model_stats": stats,
            "model_parameters": sum(p.numel() for p in agent.policy_net.parameters()),
            "device": str(agent.device),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/environment/info")
async def get_environment_info():
    """Get current environment information."""
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")

    try:
        grid_state = env.get_grid_state()
        return {
            "grid_size": env.grid_size,
            "action_space": env.action_space_n,
            "state_shape": list(env.state_shape),
            "max_steps": env.max_steps,
            "current_step": env.steps_taken,
            "grid_state": grid_state,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/live_navigation")
async def websocket_live_navigation(websocket: WebSocket):
    """
    WebSocket endpoint for live navigation visualization.
    Streams agent's actions step by step.
    """
    await websocket.accept()

    global env, agent

    if agent is None:
        await websocket.send_json({"error": "No trained model available"})
        await websocket.close()
        return

    try:
        # Receiving configuration
        config = await websocket.receive_json()
        grid_size = config.get("grid_size", 15)

        # Resetting environment
        if grid_size != env.grid_size:
            env = NavigationEnvironment(grid_size=grid_size, dynamic_traffic=True)

        state, info = env.reset()

        # Sending initial state
        await websocket.send_json(
            {
                "type": "init",
                "grid_state": env.get_grid_state(),
                "info": info,
            }
        )

        # Running episode with live updates
        done = False
        steps = 0
        max_steps = 200

        import asyncio

        while not done and steps < max_steps:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            await websocket.send_json(
                {
                    "type": "step",
                    "step": steps,
                    "action": int(action),
                    "grid_state": env.get_grid_state(),
                    "reward": float(reward),
                    "done": done,
                    "info": info,
                }
            )

            state = next_state
            steps += 1

            await asyncio.sleep(0.1)

        await websocket.send_json({"type": "complete", "steps": steps})

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment_initialized": env is not None,
        "model_loaded": agent is not None
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI server...")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )