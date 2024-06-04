from typing import Any, Dict, Union
import numpy as np
import torch
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("WipeEnv-v0", max_episode_steps=100)
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
        self.cleaned_spots = []
        grid_size = 5
        spot_half_size = 0.01

        # Create tensors for the grid positions
        i_indices = torch.arange(grid_size, device=self.device).repeat(grid_size, 1).T.flatten()
        j_indices = torch.arange(grid_size, device=self.device).repeat(grid_size)

        # Calculate the x, y positions for the spots
        x_positions = i_indices * 0.04 - 0.08
        y_positions = j_indices * 0.04 - 0.08
        z_positions = torch.full((grid_size * grid_size,), 0.01, device=self.device)

        # Combine x, y, z positions into a single tensor
        positions = torch.stack((x_positions, y_positions, z_positions), dim=1)

        # Create quaternion tensor (no rotation)
        quats = torch.tensor([[1, 0, 0, 0]], device=self.device).repeat(grid_size * grid_size, 1)

        for idx in range(grid_size * grid_size):
            spot = actors.build_box(
                self.scene,
                half_sizes=[spot_half_size, spot_half_size, 0.01],
                color=[0.5, 0.5, 0.5, 1],
                name=f"spot_{idx // grid_size}_{idx % grid_size}"
            )
            pos = positions[idx].unsqueeze(0)
            quat = quats[idx].unsqueeze(0)
            spot.set_pose(Pose.create_from_pq(p=pos, q=quat))
            self.spot_grid.append(spot)
            self.cleaned_spots.append(False)
        self.grid_size = grid_size

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.cleaned_spots = [False] * len(self.spot_grid)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            spot_positions=torch.cat([spot.pose.p for spot in self.spot_grid], dim=1),
            cleaned_spots=torch.tensor(self.cleaned_spots, device=self.device).unsqueeze(0),
        )
        return obs

    def evaluate(self):
        tcp_pose = self.agent.tcp.pose.p
        for i, spot in enumerate(self.spot_grid):
            distance = torch.linalg.norm(tcp_pose - spot.pose.p, axis=1)
            if distance <= 0.01:
                self.cleaned_spots[i] = True
        
        # Convert success to a tensor
        success = torch.tensor(all(self.cleaned_spots), device=self.device)
        return {
            "success": success,
            "cleaned_spots": self.cleaned_spots,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p  # Assuming this is a tensor with shape (3,)
        rewards = []

        for i, spot in enumerate(self.spot_grid):
            spot_pose = spot.pose.p  # Assuming this is a tensor with shape (3,)
            distance = torch.linalg.norm(tcp_pose - spot_pose, dim=-1)  # Ensure correct dimension
            reward = 1 - torch.tanh(5 * distance)
            
            if distance <= 0.01 and not self.cleaned_spots[i]:
                self.cleaned_spots[i] = True
                reward += 1  # Reward for cleaning a new spot
            
            rewards.append(reward)
        
        total_reward = torch.tensor(rewards).sum()  # Ensure rewards are summed correctly
        
        if all(self.cleaned_spots):
            total_reward += 5  # Bonus for cleaning all spots
        
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


