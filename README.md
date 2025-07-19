# 🤖 AI Robot Path Planner using Deep Q-Network (DQN)

An intelligent grid-based robot navigation system powered by Deep Reinforcement Learning.

---

## 🎯 Project Overview

This project implements a **Deep Q-Network (DQN)** agent that learns to navigate a 2D grid world from a **start** point to a **goal**, optimizing its route using Reinforcement Learning techniques.

---

## ✨ Key Features

- 🧠 **Deep Q-Learning Algorithm** – Neural network-based action decision system  
- 🎮 **Interactive Streamlit Dashboard** – Real-time training and simulation  
- 📊 **Performance Tracking** – Visualize episode rewards and Q-value learning  
- 🗺️ **Visual Path Display** – Grid map with step-by-step emoji paths  
- ⚙️ **Configurable Parameters** – Control grid size, episodes, and learning rate  
- 🚀 **Real-Time Simulation** – See the agent learn and move in real time  

---

## 🛠️ Technical Implementation

| Component       | Details                                                                      |
|------------------|-------------------------------------------------------------------------------|
| **Environment**  | Custom GridWorld with start (🚩) and goal (🏁) positions                     |
| **Agent**        | DQN Agent with ε-greedy strategy and experience replay                      |
| **Neural Net**   | Fully connected NN with 2 hidden layers (24 → 24 → 4 neurons)               |
| **Reward System**| +100 for reaching goal, -1 per step to promote efficiency                   |
| **Training**     | Batch training with replay memory (buffer size: 2000)                       |

---

## 🧠 System Architecture

<img width="399" height="105" alt="image" src="https://github.com/user-attachments/assets/fad7b207-b225-4ca9-8f8e-f250648438f3" />

🎯 Results
🧠 Learning Efficiency: Learns optimal paths in 200–500 episodes

🏆 Success Rate: Over 95% goal-reaching after training

🧭 Path Optimization: Shortest route discovered via exploration-exploitation

🔧 Technologies Used
Python 3.8+ – Core programming

TensorFlow / Keras – Deep learning framework

Streamlit – Interactive GUI

NumPy – Numerical computations

Matplotlib – Plotting episode rewards
