import os
import copy
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mani_skill.envs  # noqa: F401
from stable_baselines3 import PPO, HerReplayBuffer  # noqa: F401
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def train_sb3(cfg: DictConfig, output_dir: str) -> None:
    hyperparams = OmegaConf.to_container(cfg.hyperparams[cfg.env.id], resolve=True)

    callback = None
    if not cfg.disable_logging:
        if cfg.dry_run:
            os.environ["WANDB_MODE"] = "dryrun"
        run = wandb.init(
            project="fetch-pick-and-place",
            config=hyperparams,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            name=f"{output_dir.split('/')[-2]}_{os.path.basename(output_dir)}",
            dir=output_dir,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=cfg.save_freqs.gradient_save_freq,
            model_save_freq=cfg.save_freqs.model_save_freq,
            model_save_path=os.path.join(output_dir, "model"),
            verbose=2,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=cfg.save_freqs.checkpoint_save_freq,
            save_path=os.path.join(output_dir, "checkpoints"),
            name_prefix="model_checkpoint",
            save_vecnormalize=True,
        )
        callback = CallbackList([wandb_callback, checkpoint_callback])

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
            record_video_trigger=lambda x: x % cfg.save_freqs.video_save_freq == 0,
            video_length=200,
        )
    else:
        env = make_env()

    n_timesteps = hyperparams["n_timesteps"]
    seed = hyperparams["seed"]

    hyperparams = copy.deepcopy(hyperparams["tqc_policy"])
    hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
    hyperparams["replay_buffer_kwargs"] = eval(hyperparams["replay_buffer_kwargs"])

    model = TQC(
        env=env,
        replay_buffer_class=HerReplayBuffer,
        verbose=1,
        seed=seed,
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

    if isinstance(env, VecNormalize):
        env.save(os.path.join(output_dir, "model", "vecnormalize.pkl"))

    if not cfg.disable_logging:
        run.finish()
