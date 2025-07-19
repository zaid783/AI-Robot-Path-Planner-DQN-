import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network model"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Predict Q-values for current state
        state_array = np.array([state])
        act_values = self.model.predict(state_array, verbose=0)
        return np.argmax(act_values[0])  # Return action with highest Q-value

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare training data
        states = np.array([transition[0] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        
        # Get current Q-values and next Q-values in batch
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        # Update Q-values based on Bellman equation
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(next_q_values[i])
            
            current_q_values[i][action] = target
        
        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
