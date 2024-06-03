# Robot-RL

This repository contains code to train Reinforcement Learning agents to control robots using the following:
* Gymnasium
* SB3
* ManiSkill

## Benchmark Runs
* Fetch Pick and Place with TQC with HER (SB3) | [WandB](https://wandb.ai/peanut-robotics/fetch-pick-and-place/workspace?nw=nwuserashisghosh)
* Panda Pick Cube with PPO (ClearMl/Maniskill) | [WandB](https://wandb.ai/peanut-robotics/maniskill-ppo/workspace?nw=nwuserashisghosh)

## Directory Structure

The project is structured as follows:

- `robot_rl/`: Contains the code for training Reinforcement Learning agents using the Gym environment.
  - `maniskill_ppo.py`: Contains the code for the PPO agent used to train the agents.
  - `train_robot.py`: Contains the code for training the agents.
  - `eval_robot.py`: Contains the code for evaluating the agents.
  - `visualize_env.py`: Contains the code for visualizing the environment.
- `docker-compose.yml`: Contains the Docker Compose file for running the Docker containers.
- `.env`: Contains environment variables needed for the project.

## Running the Project

To run the project, you can use Docker and Docker Compose, and `just`. Follow these steps:

1. Run `just` to get a list of commands
2. To train, edit the command in `docker-compose.yml` and run `just train`
3. To eval, edit the command in `docker-compose.yml` and run `just eval`

## Files

- `maniskill_ppo.py`: Contains the code for the PPO agent used to train the agents.
- `train_robot.py`: Contains the code for training the agents.
- `eval_robot.py`: Contains the code for evaluating the agents.
- `visualize_env.py`: Contains the code for visualizing the environment.
- `Dockerfile`: Contains the Dockerfile for building the Docker image.
- `docker-compose.yml`: Contains the Docker Compose file for running the Docker containers.
- `requirements.txt`: Contains the Python dependencies needed for the project.
- `.env`: Contains environment variables needed for the project.

