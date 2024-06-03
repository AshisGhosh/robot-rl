import os
import logging
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
from stable_baselines3 import PPO, HerReplayBuffer  # noqa: F401
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from omegaconf import DictConfig


def eval_sb3(cfg: DictConfig, logger: logging.Logger):
    # Use the MODEL_PATH environment variable or config
    model_path = os.getenv("MODEL_PATH", cfg.model_path)
    visualize = cfg.visualize

    # Ensure the model path is valid
    if not os.path.isfile(model_path):
        raise ValueError(f"The specified model path {model_path} is not a valid file.")

    # Derive the VecNormalize path from the model path
    base_path = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    # Split the model_name to get the step part correctly
    parts = model_name.split("_")
    step_part = parts[-2] + "_" + parts[-1]  # e.g., "1000000_steps.zip"
    vecnormalize_name = model_name.replace(
        step_part, f"vecnormalize_{step_part.replace('.zip', '.pkl')}"
    )
    vecnormalize_path = os.path.join(base_path, vecnormalize_name)

    # Example:
    # model_path: /path/to/model_checkpoint_1000000_steps.zip
    # vecnormalize_path: /path/to/model_checkpoint_vecnormalize_1000000_steps.pkl

    def make_env():
        render_mode = "human" if visualize else "rgb_array"
        env = gym.make(cfg.env.id, render_mode=render_mode)
        env = Monitor(env)  # record stats such as returns
        env = TimeFeatureWrapper(env)
        return env

    env = DummyVecEnv([make_env])

    # Load the normalization stats if they exist
    if os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False

    # Load the model
    model = TQC.load(model_path, env=env)

    success_count = 0
    total_steps = 0

    for episode in range(cfg.num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        episode_rewards = 0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            step_count += 1

            if visualize:
                env.render()

        total_steps += step_count
        success = info[0].get("is_success", False)
        if success:
            success_count += 1

        logger.info(
            f"Episode {episode + 1}: Success: {success}, Steps: {step_count}, Total Reward: {episode_rewards}"
        )

    env.close()

    logger.info(
        f"Evaluation Summary: {success_count}/{cfg.num_episodes} episodes successful."
    )
    logger.info(f"Average steps per episode: {total_steps / cfg.num_episodes:.2f}")
