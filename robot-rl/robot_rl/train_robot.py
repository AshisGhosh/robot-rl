import yaml
import copy
import gymnasium as gym
import argparse
from stable_baselines3 import PPO, HerReplayBuffer  # noqa: F401
from sb3_contrib import TQC

# from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecNormalize
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import wandb
from wandb.integration.sb3 import WandbCallback

from dotenv import load_dotenv

load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-id",
    type=str,
    default="FetchPickAndPlace-v2",
    help="The environment ID to train the agent on",
)
parser.add_argument(
    "--visualize-episodes",
    type=int,
    help="Number of episodes to visualize for a limited debug run",
)
parser.add_argument(
    "--disable-logging", action="store_true", help="Disable WandB logging"
)
parser.add_argument(
    "--seed", type=int, default=0, help="Random seed for reproducibility"
)
args = parser.parse_args()

with open("hyperparam/rl-zoo.yml") as f:
    hyperparams_dict = yaml.safe_load(f)
    if args.env_id in list(hyperparams_dict.keys()):
        hyperparams = hyperparams_dict[args.env_id].copy()
    else:
        raise ValueError(f"Hyperparameters not found for {args.env_id}")

env_config = hyperparams_dict[args.env_id]
env_config.update({"env_id": args.env_id})

callback = None
if not args.disable_logging:
    run = wandb.init(
        project="fetch-pick-and-place",
        config=env_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=args.env_id + "_" + str(args.seed) + "_TRPO",
    )
    callback = (
        WandbCallback(
            gradient_save_freq=200000,
            model_save_freq=200000,
            model_save_path=f"models/{args.env_id}",
            verbose=2,
        ),
    )


def make_env():
    render_mode = "human" if args.visualize_episodes else None
    env = gym.make(args.env_id, render_mode=render_mode)

    env = Monitor(env)  # record stats such as returns
    env = TimeFeatureWrapper(env)

    return env


if not args.visualize_episodes:
    env = DummyVecEnv([make_env])

    normalize_kwargs = {"gamma": hyperparams["gamma"]}

    env = VecNormalize(env, **normalize_kwargs)
    # Get the env_wrapper hyperparam

    env = VecVideoRecorder(
        env,
        f"videos/{args.env_id}_{args.seed}",
        record_video_trigger=lambda x: x % 200000 == 0,
        video_length=200,
    )
else:
    env = make_env()


hyperparams["policy_kwargs"] = eval(hyperparams["policy_kwargs"])
hyperparams["replay_buffer_kwargs"] = eval(hyperparams["replay_buffer_kwargs"])

n_timesteps = copy.deepcopy(hyperparams["n_timesteps"])
del hyperparams["n_timesteps"]

model = TQC(
    env=env,
    replay_buffer_class=HerReplayBuffer,
    verbose=1,
    seed=args.seed,
    device="cuda",
    tensorboard_log=f"runs/{args.env_id}_{args.seed}_1",
    **hyperparams,
)
# model = TRPO(env=env, verbose=1,  seed=args.seed, device='cuda', tensorboard_log=f"runs/{args.env_id}_{args.seed}_1", **hyperparams)

# tensorboard_log=f"runs/{run.id}",

if args.visualize_episodes:
    print(f"Visualizing {args.visualize_episodes} episodes")
    n_timesteps = args.visualize_episodes

model.learn(
    total_timesteps=n_timesteps,
    callback=callback,
)

if not args.disable_logging:
    run.finish()
