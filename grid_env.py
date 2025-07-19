import numpy as np
import random

class GridWorldEnv:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.state_size = 2
        self.action_size = 4  # Up, Down, Left, Right
        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.agent_pos = [0, 0]  # Start at top-left
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]  # Goal at bottom-right
        self.state = self.agent_pos.copy()
        return np.array(self.state)

    def step(self, action):
        """Take an action and return next state, reward, and done flag"""
        x, y = self.agent_pos

        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # Right
            y += 1

        self.agent_pos = [x, y]
        
        # Check if goal is reached
        done = (self.agent_pos == self.goal_pos)
        
        # Reward structure
        if done:
            reward = 100  # Large positive reward for reaching goal
        else:
            reward = -1  # Small negative reward for each step
        
        return np.array(self.agent_pos), reward, done

    def render(self):
        """Print the grid to console for debugging"""
        grid = [[" " for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        x, y = self.agent_pos
        gx, gy = self.goal_pos
        
        grid[x][y] = "A"  # Agent
        if [x, y] != [gx, gy]:  # Don't overwrite agent if at goal
            grid[gx][gy] = "G"  # Goal
            
        for row in grid:
            print(" ".join(row))
        print()
