from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import ale_py

eval_env = Monitor(gym.make("ALE/Pacman-v5", render_mode="rgb_array", frameskip=4))
train_env = gym.make("ALE/Pacman-v5", render_mode="rgb_array", frameskip=4)

model = DQN(
    policy="CnnPolicy", 
    env=train_env,
    learning_rate=0.00005,
    buffer_size=70_000, 
    batch_size=64,
    gamma=0.999,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.005,
    learning_starts=100_000,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

try:
    model.learn(total_timesteps=1_000_000,  
                tb_log_name="./tb/", 
                reset_num_timesteps=False)

except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving the model and exiting...")
    model.save("Pacman")
    model.save_replay_buffer("Pacman_ReplayBuffer")
    model.policy.save("Pacman_DqnPolicy")
    print("Model saved successfully.")

else:
    print("\nTraining finalized. Saving the model and exiting...")
    model.save("Pacman")
    model.save_replay_buffer("Pacman_ReplayBuffer")
    model.policy.save("Pacman_DqnPolicy")
    print("Model saved successfully.")