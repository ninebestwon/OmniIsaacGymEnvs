# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.table import Table
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.valve import Valve
from omniisaacgymenvs.robots.articulations.views.table_view import TableView
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from omniisaacgymenvs.robots.articulations.views.valve_view import ValveView

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.cloner import Cloner

import numpy as np
import torch
import math

from pxr import Usd, UsdGeom


class FrankaValveTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1/60

        # these must be defined in the task class
        self._num_observations = 23
        self._num_actions = 9

        # call the parent class constructor to initialize key RL variables
        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        """
        utility method for parsing the sim_config object
        """
        # extract task config from main config dictionary
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # parse task config parameters
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        """
        Franka_Valve Config
        """
        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        # reward scale
        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

    def set_up_scene(self, scene) -> None:
        """
        setting up our simulation world
        """
        # first create a single table
        self.get_table()
        # first create a single franka
        self.get_franka()
        # first create a single valve
        self.get_valve()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an FrankaView to hold our collection of environments
        self._tables = TableView(
            prim_paths_expr="/World/envs/.*/table",
            name="table_view"
        )

        # construct an FrankaView to hold our collection of environments
        self._frankas = FrankaView(
            prim_paths_expr="/World/envs/.*/franka",
            name="franka_view"
        )
        # register the FrankaView object to the world, so that it can be initialized
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)

        # construct an ValveView to hold our collection of environments
        self._valves = ValveView(
            prim_paths_expr="/World/envs/.*/valve",
            name="valve_view"
        )
        # register the ValveView object to the world, so that it can be initialized
        scene.add(self._valves)
        scene.add(self._valves._handles)
        
        self.init_data()
        return

    def get_table(self):
        # add a single table to the stage
        table = Table(
            prim_path=self.default_zero_env_path + "/table",
            name="table"
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "table",
            get_prim_at_path(table.prim_path),
            self._sim_config.parse_actor_config("table")
        )

    def get_franka(self):
        # add a single franka robot to the stage
        franka = Franka(
            prim_path=self.default_zero_env_path + "/franka",
            name="franka"
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "franka", 
            get_prim_at_path(franka.prim_path), 
            self._sim_config.parse_actor_config("franka")
        )

    def get_valve(self):
        # add a single valve to the stage
        valve = Valve(
            prim_path=self.default_zero_env_path + "/valve",
            name="valve",
            usd_path=f"/home/vision/Downloads/valves/round_valve/round_valve_1.usd",
            translation=(0.8, 0.0, 0.0),
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "valve",
            get_prim_at_path(valve.prim_path),
            self._sim_config.parse_actor_config("valve")
        )

    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            
            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")),
            self._device
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")),
            self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")),
            self._device
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        valve_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self._device)
        self.valve_local_grasp_pos = valve_local_grasp_pose[0:3].repeat((self._num_envs, 1))
        self.valve_local_grasp_rot = valve_local_grasp_pose[3:7].repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))
        self.valve_up_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat((self._num_envs, 1))

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )

        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

    def post_reset(self):
        """
        automatically triggered by the Isaac Sim framework at the very beginning of simulation
        """
        # retrieve franka joint indices
        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros(
            (self.num_envs, self.num_franka_dofs), device=self._device
        )
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def reset_idx(self, env_ids):
        """
        reset method for the Cartpole task
        """
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        # reset cabinet
        self._valves.set_joint_positions(torch.zeros_like(self._valves.get_joint_positions(clone=False)[env_ids]), indices=indices)
        self._valves.set_joint_velocities(torch.zeros_like(self._valves.get_joint_velocities(clone=False)[env_ids]), indices=indices)

        # reset props
        if self.num_props > 0:
            self._props.set_world_poses(
                self.default_prop_pos[self.prop_indices[env_ids].flatten()], 
                self.default_prop_rot[self.prop_indices[env_ids].flatten()], 
                self.prop_indices[env_ids].flatten().to(torch.int32)
        )

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)

        # reset the reset buffer and progress buffer after applying reset
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        """
        reset the reset buffer and progress buffer to zeros for the corresponding environments
        """
        # make sure simulation has not been stopped from the UI
        if not self._env._world.is_playing():
            return
        
        # extract environment indices that need reset and reset them
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # make sure actions buffer is on the same device as the simulation
        self.actions = actions.clone().to(self._device)

        # apply actions to all of the environments
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)

    def get_observations(self) -> dict:
        """
        implement our observations
        """
        # retrieve franka hand joint positions and rotations
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        # retrieve cabinet drawer joint positions and rotations
        handle_pos, handle_rot = self._valves._handles.get_world_poses(clone=False)
        # retrieve franka joint positions and velocities
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)
        # retrieve cabinet joint positions and velocities
        self.valve_dof_pos = self._valves.get_joint_positions(clone=False)
        self.valve_dof_vel = self._valves.get_joint_velocities(clone=False)
        self.franka_dof_pos = franka_dof_pos

        self.franka_grasp_rot, self.franka_grasp_pos, self.handle_grasp_rot, self.handle_grasp_pos = self.compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            handle_rot,
            handle_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

        # retrieve franka left finger joint positions and rotations
        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        # retrieve franka right finger joint positions and rotations
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0
            * (franka_dof_pos - self.franka_dof_lower_limits)
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
            - 1.0
        )
        to_target = self.handle_grasp_pos - self.franka_grasp_pos
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled,
                franka_dof_vel * self.dof_vel_scale,
                to_target,
                self.valve_dof_pos[:, 3].unsqueeze(-1),
                self.valve_dof_vel[:, 3].unsqueeze(-1),
            )
            ,dim=-1,
        )

        # construct the observations dictionary and return
        observations = {
            self._frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        handle_rot,
        handle_pos,
        handle_local_grasp_rot,
        handle_local_grasp_pos,
    ):

        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_handle_rot, global_handle_pos = tf_combine(
            handle_rot, handle_pos, handle_local_grasp_rot, handle_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_handle_rot, global_handle_pos

    def calculate_metrics(self) -> None:
        """
        compute the reward for the task
        """
        self.rew_buf[:] = self.compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.cabinet_dof_pos,
            self.franka_grasp_pos, self.drawer_grasp_pos, self.franka_grasp_rot, self.drawer_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis, self.drawer_inward_axis, self.gripper_up_axis, self.drawer_up_axis,
            self._num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self._max_episode_length, self.franka_dof_pos,
            self.finger_close_reward_scale,
        )

    def compute_franka_reward(
        self, reset_buf, progress_buf, actions, cabinet_dof_pos,
        franka_grasp_pos, drawer_grasp_pos, franka_grasp_rot, drawer_grasp_rot,
        franka_lfinger_pos, franka_rfinger_pos,
        gripper_forward_axis, drawer_inward_axis, gripper_up_axis, drawer_up_axis,
        num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
        finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length, franka_dof_pos, 
        finger_close_reward_scale
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

        # distance from hand to the drawer
        d = torch.norm(franka_grasp_pos - drawer_grasp_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d ** 2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
        axis2 = tf_vector(drawer_grasp_rot, drawer_inward_axis)
        axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
        axis4 = tf_vector(drawer_grasp_rot, drawer_up_axis)

        dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of up axis for gripper
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)

        # bonus if left finger is above the drawer handle and right below
        around_handle_reward = torch.zeros_like(rot_reward)
        around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                           torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                       around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
        # reward for distance of each finger from the drawer
        finger_dist_reward = torch.zeros_like(rot_reward)
        lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - drawer_grasp_pos[:, 2])
        finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > drawer_grasp_pos[:, 2],
                                         torch.where(franka_rfinger_pos[:, 2] < drawer_grasp_pos[:, 2],
                                                     (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)


        finger_close_reward = torch.zeros_like(rot_reward)
        finger_close_reward = torch.where(d <=0.03, (0.04 - franka_dof_pos[:, 7]) + (0.04 - franka_dof_pos[:, 8]), finger_close_reward)

        # regularization on the actions (summed for each environment)
        action_penalty = torch.sum(actions ** 2, dim=-1)

        # how far the cabinet has been opened out
        open_reward = cabinet_dof_pos[:, 3] * around_handle_reward + cabinet_dof_pos[:, 3]  # drawer_top_joint

        rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward \
            + around_handle_reward_scale * around_handle_reward + open_reward_scale * open_reward \
            + finger_dist_reward_scale * finger_dist_reward - action_penalty_scale * action_penalty + finger_close_reward * finger_close_reward_scale

        # bonus for opening drawer properly
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.01, rewards + 0.5, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.2, rewards + around_handle_reward, rewards)
        rewards = torch.where(cabinet_dof_pos[:, 3] > 0.39, rewards + (2.0 * around_handle_reward), rewards)

        # # prevent bad style in opening drawer
        # rewards = torch.where(franka_lfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)
        # rewards = torch.where(franka_rfinger_pos[:, 0] < drawer_grasp_pos[:, 0] - distX_offset,
        #                       torch.ones_like(rewards) * -1, rewards)

        return rewards

    def is_done(self) -> None:
        # reset if drawer is open or max length reached
        self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
