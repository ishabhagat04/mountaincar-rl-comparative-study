
import streamlit as st
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

st.set_page_config(
    page_title="MountainCar RL Dashboard",
    page_icon="🏔️",
    layout="wide"
)

st.title("🏔️ MountainCar RL Dashboard")
st.markdown("*Watch the trained agent in action!*")

# -------------------- DQN Network --------------------
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

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = DQNNetwork()
    if os.path.exists('models/dqn_model.pth'):
        try:
            model.load_state_dict(torch.load('models/dqn_model.pth', map_location='cpu'))
            model.eval()
            return model, True
        except:
            return model, False
    return model, False

model, model_loaded = load_model()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if model_loaded:
        st.success("✅ DQN Model Loaded")
    else:
        st.error("❌ Model Not Found")
        st.info("Run training cells first to generate the model")

    st.markdown("---")

    max_steps = st.slider("Max Steps", 100, 500, 200, 50)

    st.markdown("---")
    st.markdown("### 🏔️ MountainCar")
    st.markdown("""
    - **State**: Position, Velocity
    - **Actions**: Left, Neutral, Right
    - **Goal**: Reach the flag at position 0.5
    """)

# -------------------- Main Content --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "DQN")
with col2:
    st.metric("Status", "✅ Loaded" if model_loaded else "❌ Not Found")
with col3:
    st.metric("Environment", "MountainCar-v0")

st.markdown("---")

# -------------------- Run Episode --------------------
if st.button("▶️ Run Episode", type="primary", use_container_width=True):
    if not model_loaded:
        st.error("❌ Model not loaded! Please train first.")
    else:
        with st.spinner("Running episode..."):
            # Create environment
            env = gym.make('MountainCar-v0', render_mode='rgb_array')
            state, _ = env.reset()
            frames = []
            total_reward = 0
            steps = 0
            goal_reached = False

            progress_bar = st.progress(0)
            status_text = st.empty()

            for step in range(max_steps):
                # Capture frame
                frame = env.render()
                frames.append(frame)

                # Get action from model
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

                # Take step
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

                # Update progress
                progress_bar.progress((step + 1) / max_steps)
                status_text.text(f"Step {step+1}/{max_steps} | Reward: {total_reward:.1f}")

                if terminated or truncated:
                    if reward == 0:
                        goal_reached = True
                    break

            env.close()

            # Display results
            st.success(f"✅ Episode Complete! Steps: {steps}, Reward: {total_reward:.1f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reward", f"{total_reward:.1f}")
            with col2:
                st.metric("Steps Taken", steps)
            with col3:
                st.metric("Goal Reached", "✅ Yes!" if goal_reached else "❌ No")

            # Show sample frames
            if frames:
                st.subheader("📸 Sample Frames")

                # Show first, middle, and last frames
                sample_indices = [0, len(frames)//2, len(frames)-1]
                sample_indices = [i for i in sample_indices if i < len(frames)]

                cols = st.columns(len(sample_indices))
                for idx, col in enumerate(cols):
                    if idx < len(sample_indices):
                        frame_idx = sample_indices[idx]
                        col.image(frames[frame_idx], caption=f"Step {frame_idx}", use_column_width=True)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Built for SMARTED INNOVATIONS Internship<br>
    🏔️ MountainCar Reinforcement Learning
</div>
""", unsafe_allow_html=True)
