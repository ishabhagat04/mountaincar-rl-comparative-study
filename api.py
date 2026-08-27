
from fastapi import FastAPI
from pydantic import BaseModel
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

app = FastAPI(title="MountainCar RL API", version="1.0.0")

# DQN Network
class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Load model
model = DQNNetwork()
if os.path.exists('models/dqn_model.pth'):
    model.load_state_dict(torch.load('models/dqn_model.pth', map_location='cpu'))
    model.eval()
    print("✅ Model loaded")

class StateRequest(BaseModel):
    position: float
    velocity: float

class ActionResponse(BaseModel):
    action: int
    action_name: str
    q_values: list

@app.get("/")
def root():
    return {"message": "MountainCar RL API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": os.path.exists('models/dqn_model.pth')}

@app.post("/predict", response_model=ActionResponse)
def predict(state: StateRequest):
    state_array = np.array([state.position, state.velocity])
    state_tensor = torch.FloatTensor(state_array).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).numpy()[0]

    action = int(np.argmax(q_values))
    action_names = {0: "Push Left", 1: "Neutral", 2: "Push Right"}

    return ActionResponse(
        action=action,
        action_name=action_names[action],
        q_values=q_values.tolist()
    )
