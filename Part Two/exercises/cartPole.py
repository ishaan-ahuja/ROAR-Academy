import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import warnings
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time as time_module  # Rename to avoid conflict
warnings.filterwarnings('ignore')

EPISODES = 100
RENDER_EVERY = 10  # Render every N episodes to see progress

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.vstack([m[0] for m in minibatch])
        actions = [m[1] for m in minibatch]
        rewards = [m[2] for m in minibatch]
        next_states = np.vstack([m[3] for m in minibatch])
        dones = [m[4] for m in minibatch]

        # Batch prediction for efficiency
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Try different ways to create the environment
env = None
render_method = None

# Method 1: Try with old API
try:
    env = gym.make('CartPole-v1')
    env.reset()
    env.render(mode='rgb_array')
    render_method = 'old_rgb'
    print("Using old render API with rgb_array mode")
except:
    pass

# Method 2: Try default render
if env is None or render_method is None:
    try:
        env = gym.make('CartPole-v1')
        env.reset()
        env.render()
        render_method = 'old_default'
        print("Using old render API with default mode")
    except:
        pass

# Method 3: Just create environment without render
if env is None:
    env = gym.make('CartPole-v1')
    render_method = 'manual'
    print("Created environment, will attempt manual visualization")

print(render_method)
# Test reset to check API version
state_or_tuple = env.reset()
if isinstance(state_or_tuple, tuple):
    state, _ = state_or_tuple
    uses_new_api = True
else:
    state = state_or_tuple
    uses_new_api = False

print(f"API version: {'new' if uses_new_api else 'old'}")

# Function to get frame based on render method
def get_frame(env, render_method):
    try:
        if render_method == 'old_rgb':
            return env.render(mode='rgb_array')
        elif render_method == 'old_default':
            # Try to get rgb_array even if not default
            try:
                return env.render(mode='rgb_array')
            except:
                return None
        else:
            return None
    except:
        return None

# Manual visualization fallback
def draw_cartpole_state(episode, state, average):
    """Manually draw the cartpole state"""
    cart_pos = state[0]
    pole_angle = state[2]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set up the plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 2)
    ax.set_aspect('equal')
    
    # Draw track
    ax.plot([-2.4, 2.4], [0, 0], 'k-', linewidth=2)
    ax.plot([-2.4, -2.4], [-0.05, 0.05], 'r-', linewidth=4)
    ax.plot([2.4, 2.4], [-0.05, 0.05], 'r-', linewidth=4)
    
    # Draw cart
    cart_width = 0.3
    cart_height = 0.2
    cart = plt.Rectangle((cart_pos - cart_width/2, -cart_height/2), 
                        cart_width, cart_height, 
                        fill=True, color='blue')
    ax.add_patch(cart)
    
    # Draw pole
    pole_length = 1.0
    pole_end_x = cart_pos + pole_length * np.sin(pole_angle)
    pole_end_y = pole_length * np.cos(pole_angle)
    ax.plot([cart_pos, pole_end_x], [0, pole_end_y], 'brown', linewidth=8)
    
    # Add pole joint
    circle = plt.Circle((cart_pos, 0), 0.05, color='black')
    ax.add_patch(circle)
    
    # Add title
    ax.set_title(f'Episode: {episode}, Angle: {np.degrees(pole_angle):.1f}°, Average Reward: {average}')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Position')
    ax.set_ylabel('Height')
    
    return fig

# Run one episode
angle = state[2]
angle_velocity = state[3]
integral = 0.0
prev_error = angle
batch_size = 32
scores = []  # Store scores for plotting


print("\nRunning CartPole with PID Controller...")
print("Watch the game below:\n")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

for e in range(EPISODES):

    # Test reset to check API version
    state_or_tuple = env.reset()
    if uses_new_api:
        state, _ = state_or_tuple
    else:
        state = state_or_tuple

    state = np.reshape(state, [1, state_size])
    # Determine if we should render this episode
    render_this_episode = (e % RENDER_EVERY == 0) or (e >= EPISODES - 5)
    
    if render_this_episode:
        print(f"\n🎮 Rendering Episode {e+1}/{EPISODES}")

    for step in range(500): 

        # DQN control
        action = agent.act(state)
    
        # Step environment
        if uses_new_api:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            next_state, reward, done, info = env.step(action)

        if step % 10 == 0 or done:
            clear_output(wait=True)
            
            # Try to get frame
            frame = get_frame(env, render_method)
            
            if frame is not None:
                # Display captured frame
                plt.figure(figsize=(8, 6))
                plt.imshow(frame)
                plt.axis('off')
                plt.title(f'Step: {step+1}')
                plt.pause(0.001)
            else:
                # Use manual visualization
                fig = draw_cartpole_state(e, next_state, np.mean(scores[-10:]))
                plt.pause(0.001)
                plt.close()            

        reward = reward if not done else -10
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
    plt.title('DQN Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.pause(0.001)

except ImportError:
    print("Matplotlib not available for plotting. Skipping learning curve visualization.")