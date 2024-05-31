import os
import gymnasium as gym
import gymnasium_robotics  # This registers Gymnasium-Robotics environments
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure

from dotenv import load_dotenv
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--visualize-episodes', type=int, help='Number of episodes to visualize for a limited debug run')
parser.add_argument('--disable-logging', action='store_true', help='Disable WandB logging')
args = parser.parse_args()

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 100000,
    "env_name": "FetchPickAndPlace-v2",
}

# Define render_mode based on visualization argument
render_mode = 'human' if args.visualize_episodes else None

# Define your custom environment
env_id = config['env_name']
env = gym.make(env_id, render_mode=render_mode)

if not args.disable_logging:
    run = wandb.init(
    project="fetch-pick-and-place",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

# Create the PPO model using MultiInputPolicy
model = PPO('MultiInputPolicy', env, verbose=1, device='cuda')

# Create a folder for logs and checkpoints
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Configure TensorBoard logger
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Set the new logger
model.set_logger(new_logger)


# Define evaluation and checkpoint callbacks
eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=10000,
                             deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir,
                                         name_prefix='ppo_pick_place_checkpoint')

# Add WandB logging callback if logging is enabled
callbacks = [eval_callback, checkpoint_callback]
if not args.disable_logging:
    callbacks.append(WandbCallback(
                            gradient_save_freq=100,  # frequency to log gradients
                            model_save_path=log_dir,  # path to save the model
                            verbose=2,
                            log="all"  # log all metrics
                        ))

# Check for visualization argument
if args.visualize_episodes:
    print(f"Running a limited debug visualization run for {args.visualize_episodes} episodes...")
    obs, info = env.reset()
    for _ in range(args.visualize_episodes):
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
else:
    # Train the model with callbacks
    total_timesteps = config['total_timesteps']
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

# Save the final model
model.save("ppo_pick_place")
env.close()
