## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_dqn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

## CartPole DQN with Rendering - Compatible with Gym 0.26.2
## Shows the game being played during training

import random
import gym
import os
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time as time_module  # Rename to avoid conflict

EPISODES = 100
RENDER_EVERY = 10  # Render every N episodes to see progress

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # Try to create environment with render mode
    try:
        # For newer Gym versions
        env = gym.make('CartPole-v1', render_mode='human')
        print("Created environment with render_mode='human'")
        use_old_render = False
    except:
        # For older Gym versions
        env = gym.make('CartPole-v1')
        print("Created environment without render_mode (will use env.render())")
        use_old_render = True
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Optional: Load pre-trained weights
    # agent.load("./save/cartpole-dqn.h5")
    
    batch_size = 32
    scores = []  # Store scores for plotting

    print("Starting DQN Training on CartPole")
    print("=" * 50)

    for e in range(EPISODES):
        # Reset environment
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            state, _ = reset_output
        else:
            state = reset_output
            
        state = np.reshape(state, [1, state_size])
        
        # Determine if we should render this episode
        render_this_episode = (e % RENDER_EVERY == 0) or (e >= EPISODES - 5)
        
        if render_this_episode:
            print(f"\n🎮 Rendering Episode {e+1}/{EPISODES}")
        
        for step in range(500):  # Changed from 'time' to 'step' to avoid conflict
            # Select action
            action = agent.act(state)
            
            # Take action
            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_output
            
            # Render if it's a display episode
            if render_this_episode and use_old_render:
                try:
                    env.render()
                    time_module.sleep(0.02)  # Slow down for visibility
                except:
                    pass
            elif render_this_episode and not use_old_render:
                # For new API with render_mode='human', rendering is automatic
                time_module.sleep(0.02)  # Just slow down
            
            # Modify reward
            reward = reward if not done else -10
            
            # Remember experience
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            if done:
                scores.append(step + 1)
                print(f"Episode: {e+1}/{EPISODES}, Score: {step+1}, ε: {agent.epsilon:.3f}")
                
                # Print progress bar
                if (e + 1) % 10 == 0:
                    avg_score = np.mean(scores[-10:])
                    print(f"📊 Last 10 episodes average: {avg_score:.1f}")
                    
                break
            
            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Save model periodically
        if (e + 1) % 50 == 0:
            print(f"💾 Checkpoint at episode {e+1}")
            # Create save directory if it doesn't exist
            # os.makedirs("./save", exist_ok=True)
            # agent.save(f"./save/cartpole-dqn-ep{e+1}.h5")
    
    env.close()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Final average score (last 10 episodes): {np.mean(scores[-10:]):.1f}")
    print(f"Best score achieved: {max(scores)}")
    print("=" * 50)
    
    # Optional: Plot learning curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(scores, alpha=0.6, label='Episode scores')
        
        # Calculate rolling average
        window = 10
        rolling_avg = [np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))]
        plt.plot(rolling_avg, linewidth=2, label=f'{window}-episode average')
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('DQN Learning Progress on CartPole-v1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")