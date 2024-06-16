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

        spot_half_size = 0.01
        for idx in range(grid_size * grid_size):
            spot = actors.build_box(
                self.scene,
                half_sizes=[spot_half_size, spot_half_size, 0.01],
                color=[1, 0, 0, 1],
                name=f"spot_{idx // grid_size}_{idx % grid_size}",
                body_type="kinematic",
                add_collision=False,
            )
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
            self.previous_velocities = torch.zeros(
                (b, 3), device=self.device
            )  # Initialize previous velocities
            self.internal_rewards = None

    def _get_obs_extra(self, info: Dict):
        tcp_pose = self.agent.tcp.pose.raw_pose  # (N, 7)
        tcp_velocity = self.agent.tcp.get_linear_velocity()  # (N, 3)

        # Collect the positions of all spots in a batch: (G, N, 3)
        spot_positions = torch.stack(
            [spot.pose.p for spot in self.spot_grid], dim=0
        )  # (G, N, 3)

        # Transpose to align with batch dimension: (N, G, 3)
        spot_positions = spot_positions.transpose(0, 1)  # (N, G, 3)

        # Calculate the distances to each spot: (N, G)
        distances_to_spots = torch.norm(
            tcp_pose[:, :3].unsqueeze(1) - spot_positions, dim=2
        )

        # Calculate the distance to the nearest uncleaned spot for each environment in the batch: (N,)
        distance_to_nearest_uncleaned_spot = torch.min(
            distances_to_spots * (~self.cleaned_spots).float(), dim=1
        )[0].unsqueeze(1)

        # Cleaned spots: (N, G)
        cleaned_spots = self.cleaned_spots.float()

        # Concatenate all observations into a single 2D tensor per environment: (N, ?)
        obs = torch.cat(
            [
                tcp_pose,  # (N, 7)
                tcp_velocity,  # (N, 3)
                distances_to_spots,  # (N, G)
                distance_to_nearest_uncleaned_spot,  # (N, 1)
                cleaned_spots,  # (N, G)
            ],
            dim=1,
        )

        return obs

    def clean_spot(self, env_idx, spot_idx):
        """
        Sets a new post for the spot to be far away.
        """
        spot = self.spot_grid[spot_idx]
        print(f"spot:{spot.shape}")
        print(f"spot pose: {spot.pose.p.shape}")
        new_pose = Pose.create_from_pq(
            p=torch.tensor([10, 10, 10], device=self.device), q=[1, 0, 0, 0]
        )
        self.px.cuda_rigid_body_data.torch()[
            self._body_data_index[self.scene._reset_mask[self._scene_idxs]], :7
        ] = new_pose
        # spot.set_pose(Pose.create_from_pq(p=torch.tensor([10, 10, 10], device=self.device), q=[1, 0, 0, 0]))

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

            spot_positions = spot.pose.p
            spot_positions[new_cleaned_spots[:, i], 2] = 99999
            spot.set_pose(Pose.create_from_pq(spot_positions))

        self.scene.px.gpu_apply_rigid_dynamic_data()
        self.scene.px.gpu_fetch_rigid_dynamic_data()

        # Convert success to a tensor
        success = torch.tensor(
            [all(self.cleaned_spots[i]) for i in range(len(self.cleaned_spots))],
            device=self.device,
        )
        return {
            "success": success,
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "new_cleaned_spots": new_cleaned_spots,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p
        tcp_orientation = self.agent.tcp.pose.q

        # 1. Reward for cleaning spots
        new_cleaned_spots = info["new_cleaned_spots"]
        reward_for_cleaning = (
            new_cleaned_spots.sum(dim=1).float() * 2.0
        )  # Reward for each new spot cleaned

        # 2. Reward for maintaining TCP position within the grid bounds
        height_buffer = 0.05  # Apply height buffer to the upper bound
        lower_bounds = torch.tensor([-0.08, -0.08, 0.01], device=self.device)
        upper_bounds = torch.tensor(
            [0.08, 0.08, 0.01 + height_buffer], device=self.device
        )

        within_bounds = ((tcp_pose >= lower_bounds) & (tcp_pose <= upper_bounds)).all(
            dim=1
        )
        reward_for_position = torch.where(
            within_bounds, 1.0, -0.5
        )  # Reward for staying in bounds, penalty for drifting out

        # 3. Reward for maintaining the end effector in the desired orientation
        desired_orientation = torch.tensor([0, 1, 0, 0], device=self.device)
        orientation_deviation = torch.sum(
            (tcp_orientation - desired_orientation) ** 2, dim=1
        )
        reward_for_orientation = torch.exp(-orientation_deviation / (2 * 0.48**2)) * 0.2

        # 4. Penalty for high acceleration (encourages smooth movements)
        tcp_velocity = self.agent.tcp.get_linear_velocity()
        acceleration = tcp_velocity - self.previous_velocities
        self.previous_velocities = tcp_velocity.clone()
        penalty_for_acceleration = -torch.sum(acceleration**2, dim=1) * 0.005

        # 5. Efficiency reward (encourage minimizing steps)
        efficiency_reward = (
            -0.01 * self.elapsed_steps
        )  # Small penalty per step to encourage task completion

        # Total reward
        reward = (
            reward_for_cleaning
            + reward_for_position
            + reward_for_orientation
            + penalty_for_acceleration
            + efficiency_reward
        )

        # Log internal rewards for debugging
        if self.internal_rewards is None:
            self.internal_rewards = torch.zeros(
                (reward.shape[0], 5), device=self.device
            )
        self.internal_rewards += torch.stack(
            [
                reward_for_cleaning,
                reward_for_position,
                reward_for_orientation,
                penalty_for_acceleration,
                efficiency_reward,
            ],
            dim=1,
        )

        # Bonus for success
        reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
