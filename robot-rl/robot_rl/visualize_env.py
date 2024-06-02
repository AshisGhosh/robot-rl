import gymnasium as gym

# Define your custom environment
env_id = "FetchPickAndPlace-v2"
env = gym.make(env_id, render_mode="human")

# Reset the environment
obs = env.reset()

# Render and interact with the environment
try:
    for _ in range(1000):  # Adjust the range as needed
        env.render()

        # Take a random action
        action = env.action_space.sample()

        # Uncomment this if you want to use a pre-trained model
        # action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
