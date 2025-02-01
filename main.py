import gymnasium as gym
from stable_baselines3 import DQN



class FuelEfficientLander(gym.Wrapper):
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # Additional fuel penalty for main engine usage (action 2)
        if action == 2:
            reward -= 2  # Stronger punishment for using fuel

        return state, reward, terminated, truncated, info


env = FuelEfficientLander(gym.make("LunarLander-v3"))


# Initialize the DQN agent
agent = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.01,
    buffer_size=10_000,
    batch_size=64,
    exploration_fraction=0.2,
    exploration_final_eps=0.01,
    target_update_interval=10_000,
    gamma=0.99,
    train_freq=4,
)

agent.load('DQN')

# Train the agent
agent.learn(total_timesteps=10000)

# Save the trained agent
agent.save('DQN')


# Reload environment with rendering enabled
env = gym.make('LunarLander-v3', render_mode="human")

# Evaluate the agent
state, info = env.reset()
for _ in range(30):
    done = False
    while not done:
        action, _ = agent.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        done = terminated or truncated

env.close()