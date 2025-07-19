import streamlit as st
import numpy as np
from grid_env import GridWorldEnv
from dqn_agent import DQNAgent

# --------- Train Agent ----------
def train_agent(episodes=500, grid_size=10):
    env = GridWorldEnv(grid_size=grid_size)
    agent = DQNAgent(state_size=2, action_size=4)
    max_steps = grid_size * grid_size
    episode_rewards = []
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for time in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        
        # Update replay memory
        agent.replay()
        episode_rewards.append(total_reward)
        
        # Update progress
        progress = (e + 1) / episodes
        progress_bar.progress(progress)
        status_text.text(f"Training Episode {e + 1}/{episodes} - Reward: {total_reward:.2f}")

    progress_bar.empty()
    status_text.empty()
    return agent, episode_rewards

# --------- Simulate Agent ----------
def simulate_agent(agent, grid_size=10):
    env = GridWorldEnv(grid_size=grid_size)
    state = env.reset()
    path = [tuple(state)]
    max_steps = grid_size * grid_size
    total_reward = 0
    
    for step in range(max_steps):
        # Use trained agent (no exploration during simulation)
        old_epsilon = agent.epsilon
        agent.epsilon = 0  # No random actions during simulation
        
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        path.append(tuple(next_state))
        total_reward += reward
        state = next_state
        
        # Restore epsilon
        agent.epsilon = old_epsilon
        
        if done:
            break

    return path, done, state, total_reward

# --------- Streamlit UI ----------
st.set_page_config(page_title="AI Robot Path Planner (DQN)", layout="wide")
st.markdown("# 🤖 AI Robot Path Planner (DQN)")
st.markdown("Train a Deep Q-Learning agent to navigate from start (🚩) to goal (🏁) in a grid world.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    grid_size = st.slider("Grid Size", 4, 15, 6)
    episodes = st.slider("Training Episodes", 100, 1000, 500, step=100)

# --- Session State ---
if "agent" not in st.session_state:
    st.session_state.agent = None
if "path" not in st.session_state:
    st.session_state.path = []
if "success" not in st.session_state:
    st.session_state.success = False
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "sim_reward" not in st.session_state:
    st.session_state.sim_reward = 0

# Create two columns for buttons
col1, col2 = st.columns(2)

# --- Train Button ---
with col1:
    if st.button("⚙️ Train Agent", type="primary", use_container_width=True):
        with st.spinner("Training agent..."):
            agent, rewards = train_agent(episodes=episodes, grid_size=grid_size)
            st.session_state.agent = agent
            st.session_state.rewards = rewards
            st.success("✅ Training completed!")

# --- Simulate Button ---
with col2:
    if st.button("▶️ Simulate Agent", type="secondary", use_container_width=True):
        if st.session_state.agent:
            with st.spinner("Running simulation..."):
                path, success, final_state, total_reward = simulate_agent(
                    st.session_state.agent, grid_size=grid_size
                )
                st.session_state.path = path
                st.session_state.success = success
                st.session_state.final_state = final_state
                st.session_state.sim_reward = total_reward
                st.success("✅ Simulation completed!")
        else:
            st.warning("⚠️ Please train the agent first!")

# --- Reward Chart ---
if st.session_state.rewards:
    st.markdown("### 📈 Training Progress")
    st.line_chart(st.session_state.rewards)

# --- Grid Display ---
st.markdown("### 🧭 Grid World Simulation")

if st.session_state.path:
    # Create grid visualization
    grid = np.full((grid_size, grid_size), "⬛", dtype=object)
    
    # Mark start and goal
    grid[0][0] = "🚩"  # Start
    grid[grid_size - 1][grid_size - 1] = "🏁"  # Goal
    
    # Mark path
    for i, (x, y) in enumerate(st.session_state.path):
        if (x, y) != (0, 0) and (x, y) != (grid_size - 1, grid_size - 1):
            grid[x][y] = "🔸"
    
    # Mark final robot position
    if st.session_state.final_state is not None:
        fx, fy = st.session_state.final_state
        if (fx, fy) == (grid_size - 1, grid_size - 1):
            grid[fx][fy] = "🤖🏁"  # Robot reached goal
        else:
            grid[fx][fy] = "🤖"  # Robot's final position
    
    # Display the grid
    for row in grid:
        st.write(" ".join(row))
    
    # Results
    if st.session_state.success:
        st.success("🎯 Agent reached the goal!")
    else:
        st.error("❌ Agent failed to reach the goal.")
    
    st.info(f"💰 Total Reward: {st.session_state.sim_reward}")

else:
    # Show empty grid
    grid = np.full((grid_size, grid_size), "⬛", dtype=object)
    grid[0][0] = "🚩"  # Start
    grid[grid_size - 1][grid_size - 1] = "🏁"  # Goal
    
    for row in grid:
        st.write(" ".join(row))
    st.info("🎮 Train and simulate the agent to see the path!")
