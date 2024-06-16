import tyro
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mani_skill.envs  # noqa: F401
import envs  # noqa: F401


def visualize_env(env_id: str = "PickCube-v1") -> None:
    # Define your custom environment
    env = gym.make(env_id, render_mode="human", sim_backend="gpu")

    # Reset the environment
    obs = env.reset()

    # Render and interact with the environment
    try:
        for _ in range(1000):  # Adjust the range as needed
            env.render()

            # Take a random action
            # action = env.action_space.sample()
            print(type(env.action_space.sample()))
            import numpy as np

            action = np.zeros(len(env.action_space.sample()))

            # Uncomment this if you want to use a pre-trained model
            # action, _ = model.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            print(info["tcp_pose"])

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


tyro.cli(visualize_env)
