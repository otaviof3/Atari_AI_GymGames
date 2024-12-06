import gymnasium as gym
from stable_baselines3 import DQN
import ale_py

env = gym.make('ALE/Pacman-v5', render_mode='human')

model = DQN.load("Pacman.zip", env=env)

obs, info = env.reset()

for _ in range(1000):  
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render() 
    if done or truncated:
        obs, info = env.reset()

env.close()