import os
import yaml
import copy
import gymnasium as gym
from stable_baselines3 import PPO, HerReplayBuffer  # noqa: F401
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from hydra.utils import get_original_cwd, to_absolute_path

load_dotenv()

def set_hydra_run_dir(base_output_dir, env_id, seed):
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, f"{run_name}_{env_id}_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    os.environ['HYDRA_RUN_DIR'] = output_dir
    return output_dir

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    hyperparams = OmegaConf.to_container(cfg.hyperparams[cfg.env.id], resolve=True)

    # Set the run directory dynamically
    output_dir = set_hydra_run_dir(to_absolute_path(cfg.paths.base_output), cfg.env.id, cfg.env.seed)

    callback = None
    if not cfg.disable_logging:
        run = wandb.init(
            project="fetch-pick-and-place",
            config=hyperparams,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            name=f"{os.path.basename(output_dir)}",
        )
        callback = WandbCallback(
            gradient_save_freq=200000,
            model_save_freq=200000,
            model_save_path=os.path.join(output_dir, "models"),
            verbose=2,
        )

    def make_env():
        render_mode = "human" if cfg.visualize_episodes else "rgb_array"
        env = gym.make(cfg.env.id, render_mode=render_mode)
        env = Monitor(env)  # record stats such as returns
        env = TimeFeatureWrapper(env)
        return env

    if not cfg.visualize_episodes:
        env = DummyVecEnv([make_env])
        normalize_kwargs = {"gamma": hyperparams["tqc_policy"]["gamma"]}
        env = VecNormalize(env, **normalize_kwargs)
        env = VecVideoRecorder(
            env,
            os.path.join(output_dir, "videos"),
            record_video_trigger=lambda x: x % 200000 == 0,
            video_length=200,
        )
    else:
        env = make_env()
    
    n_timesteps = hyperparams["n_timesteps"]

    hyperparams = copy.deepcopy(hyperparams['tqc_policy'])
    hyperparams['policy_kwargs'] = eval(hyperparams["policy_kwargs"])
    hyperparams['replay_buffer_kwargs'] = eval(hyperparams["replay_buffer_kwargs"])

    model = TQC(
        env=env,
        replay_buffer_class=HerReplayBuffer,
        verbose=1,
        seed=cfg.env.seed,
        device="cuda",
        tensorboard_log=os.path.join(output_dir, "runs"),
        **hyperparams,
    )

    if cfg.visualize_episodes:
        print(f"Visualizing {cfg.visualize_episodes} episodes")
        n_timesteps = cfg.visualize_episodes

    model.learn(
        total_timesteps=n_timesteps,
        callback=callback,
    )

    if not cfg.disable_logging:
        run.finish()

if __name__ == "__main__":
    main()