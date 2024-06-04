from typing import Any, Dict, Union
import numpy as np
import torch
import mani_skill.envs.utils.randomization as randomization  # noqa: F401
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("WipeEnv-v0", max_episode_steps=50)
class WipeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.spot_grid = []
        grid_size = 5

        # Create tensors for the grid positions
        i_indices = (
            torch.arange(grid_size, device=self.device).repeat(grid_size, 1).T.flatten()
        )
        j_indices = torch.arange(grid_size, device=self.device).repeat(grid_size)

        # Calculate the x, y positions for the spots
        x_positions = i_indices * 0.04 - 0.08
        y_positions = j_indices * 0.04 - 0.08
        z_positions = torch.full((grid_size * grid_size,), 0.01, device=self.device)

        # Combine x, y, z positions into a single tensor
        self.positions = torch.stack((x_positions, y_positions, z_positions), dim=1)
        print(f"positions: {self.positions.shape}")

        spot_half_size = 0.01
        for idx in range(grid_size * grid_size):
            spot = actors.build_box(
                self.scene,
                half_sizes=[spot_half_size, spot_half_size, 0.01],
                color=[0.5, 0.5, 0.5, 1],
                name=f"spot_{idx // grid_size}_{idx % grid_size}",
            )
            # print(type(spot))
            self.spot_grid.append(spot)
        self.grid_size = grid_size

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Setting poses need to be in the _intialize_episode method if using GPU
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            for idx, spot in enumerate(self.spot_grid):
                pos = self.positions[idx].unsqueeze(0)
                quat = [1, 0, 0, 0]
                spot.set_pose(Pose.create_from_pq(p=pos, q=quat))
            self.cleaned_spots = torch.tensor(
                [False] * len(self.spot_grid), device=self.device
            ).repeat(b, 1)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            spot_positions=torch.cat([spot.pose.p for spot in self.spot_grid], dim=1),
            # cleaned_spots=torch.tensor(self.cleaned_spots, device=self.device).unsqueeze(0),
        )
        return obs

    def evaluate(self):
        tcp_pose = self.agent.tcp.pose.p
        new_cleaned_spots = torch.zeros(
            self.cleaned_spots.shape, device=self.device, dtype=torch.bool
        )
        for i, spot in enumerate(self.spot_grid):
            distance = torch.linalg.norm(tcp_pose - spot.pose.p, axis=1)
            within_distance = distance <= 0.02
            new_cleaned_spots[:, i] = within_distance & ~self.cleaned_spots[:, i]
            self.cleaned_spots[:, i] = within_distance | self.cleaned_spots[:, i]

        # Convert success to a tensor
        success = torch.tensor(
            [all(self.cleaned_spots[i]) for i in range(len(self.cleaned_spots))],
            device=self.device,
        )
        return {
            "success": success,
            "new_cleaned_spots": new_cleaned_spots,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p  # Assuming this is a tensor with shape (3,)
        reward_per_spot_array = []

        for i, spot in enumerate(self.spot_grid):
            reward_per_spot = torch.zeros(tcp_pose.shape[0], device=self.device)
            reward_per_spot[info["new_cleaned_spots"][:, i]] += (
                1  # Reward for cleaning a spot
            )
            reward_per_spot_array.append(reward_per_spot)

        reward = torch.sum(torch.stack(reward_per_spot_array), dim=0)

        # reward for TCP being in bounds of the grid
        height_buffer = 0.03
        lower_bounds = torch.tensor(
            [-0.08, -0.08, 0.01 + height_buffer], device=self.device
        )
        upper_bounds = torch.tensor([0.08, 0.08, 0.01], device=self.device)
        center_bounds = (lower_bounds + upper_bounds) / 2
        distances_sq = torch.sum((tcp_pose - center_bounds) ** 2, dim=1)

        within_bounds = ((tcp_pose >= lower_bounds) & (tcp_pose <= upper_bounds)).all(
            dim=1
        )
        reward_within_bounds = (
            0.1  # Define the fixed reward for positions within the bounds
        )

        sigma = 0.04  # Adjust this value based on your requirements
        gaussian_reward = torch.exp(-distances_sq / (2 * sigma**2))
        outside_bounds_reward_factor = 0.5
        gaussian_reward *= outside_bounds_reward_factor

        reward += torch.where(within_bounds, reward_within_bounds, gaussian_reward)

        reward[info["success"]] += 5

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
