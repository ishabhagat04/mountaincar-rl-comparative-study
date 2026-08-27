"""
🚀 MountainCar RL - FastAPI Deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="MountainCar RL API",
    description="Reinforcement Learning agent for MountainCar environment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=3, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=2, action_dim=3, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

def load_dqn_model(model_path='models/dqn_model.pth'):
    model = DQNNetwork()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ DQN model loaded")
    return model

def load_reinforce_model(model_path='models/reinforce_model.pth'):
    model = PolicyNetwork()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ REINFORCE model loaded")
    return model

def load_q_table(q_table_path='models/q_table.npy'):
    if os.path.exists(q_table_path):
        q_table = np.load(q_table_path)
        print(f"✅ Q-table loaded")
        return q_table
    return None

dqn_model = load_dqn_model()
reinforce_model = load_reinforce_model()
q_table = load_q_table()

ENV_NAME = 'MountainCar-v0'
STATE_BOUNDS = [(-1.2, 0.6), (-0.07, 0.07)]
BINS = [30, 30]

def discretize_state(state):
    indices = []
    for i, (value, (low, high)) in enumerate(zip(state, STATE_BOUNDS)):
        value = np.clip(value, low, high)
        bin_width = (high - low) / BINS[i]
        index = int((value - low) / bin_width)
        index = min(index, BINS[i] - 1)
        indices.append(index)
    return tuple(indices)

ACTION_NAMES = {0: "Push Left", 1: "Neutral", 2: "Push Right"}

class StateRequest(BaseModel):
    position: float = 0.0
    velocity: float = 0.0

class PredictionResponse(BaseModel):
    action: int
    action_name: str
    algorithm: str

class SimulateRequest(BaseModel):
    algorithm: str = "DQN"
    max_steps: int = 1000

class SimulateResponse(BaseModel):
    total_reward: float
    steps: int
    goal_reached: bool

@app.get("/")
async def root():
    return {
        "name": "MountainCar RL API",
        "status": "running",
        "algorithms": ["Q-Learning", "DQN", "REINFORCE"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "q_table": q_table is not None,
            "dqn": dqn_model is not None,
            "reinforce": reinforce_model is not None
        }
    }

@app.get("/info")
async def get_info():
    return {
        "environment": ENV_NAME,
        "actions": ACTION_NAMES,
        "goal": "Reach position 0.5",
        "reward": "-1 per step, 0 at goal"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(state: StateRequest, algorithm: str = "DQN"):
    state_array = np.array([state.position, state.velocity])

    if algorithm == "Q-Learning" and q_table is not None:
        state_disc = discretize_state(state_array)
        action = int(np.argmax(q_table[state_disc]))
        return PredictionResponse(action=action, action_name=ACTION_NAMES[action], algorithm="Q-Learning")

    elif algorithm == "DQN" and dqn_model is not None:
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn_model(state_tensor)
        action = int(q_values.argmax().item())
        return PredictionResponse(action=action, action_name=ACTION_NAMES[action], algorithm="DQN")

    elif algorithm == "REINFORCE" and reinforce_model is not None:
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
        with torch.no_grad():
            probs = reinforce_model(state_tensor)
        action = int(probs.argmax().item())
        return PredictionResponse(action=action, action_name=ACTION_NAMES[action], algorithm="REINFORCE")

    else:
        raise HTTPException(status_code=400, detail=f"Algorithm '{algorithm}' not available")

@app.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest):
    if request.algorithm == "Q-Learning" and q_table is not None:
        def get_action(state):
            state_disc = discretize_state(state)
            return int(np.argmax(q_table[state_disc]))
    elif request.algorithm == "DQN" and dqn_model is not None:
        def get_action(state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = dqn_model(state_tensor)
            return int(q_values.argmax().item())
    elif request.algorithm == "REINFORCE" and reinforce_model is not None:
        def get_action(state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs = reinforce_model(state_tensor)
            return int(probs.argmax().item())
    else:
        raise HTTPException(status_code=400, detail=f"Algorithm '{request.algorithm}' not available")

    sim_env = gym.make(ENV_NAME)
    state, _ = sim_env.reset()
    total_reward = 0
    steps = 0

    for step in range(request.max_steps):
        action = get_action(state)
        next_state, reward, terminated, truncated, _ = sim_env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break

    sim_env.close()
    return SimulateResponse(total_reward=float(total_reward), steps=steps, goal_reached=reward == 0 if done else False)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
