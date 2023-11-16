# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole


class CartpoleTask(RLTask):
    """
    create a new RL environment class for the task
    performing resets, applying actions, collecting observations and computing rewards
    """
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        # these must be defined in the task class
        self._num_observations = 4
        self._num_actions = 1

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

        # reset and actions related variables
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        """
        Cartpole Config
        """
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

    def set_up_scene(self, scene) -> None:
        """
        setting up our simulation world
        triggered automatically by the Isaac Sim when initializing the world
        """
        # first create a single cartpole environment
        self.get_cartpole()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an ArticulationView(cartpole) to hold our collection of environments
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole",
            name="cartpole_view",
            reset_xform_properties=False
        )
        # register the ArticulationView(cartpole) object to the world, so that it can be initialized
        scene.add(self._cartpoles)
        return

    def get_cartpole(self):
        # add a single robot to the stage
        cartpole = Cartpole(
            prim_path=self.default_zero_env_path + "/Cartpole",
            name="Cartpole",
            translation=self._cartpole_positions
        )
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Cartpole",
            get_prim_at_path(cartpole.prim_path),
            self._sim_config.parse_actor_config("Cartpole")
        )

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("cartpole_view"):
            scene.remove_object("cartpole_view", registry_only=True)
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        scene.add(self._cartpoles)

    def post_reset(self):
        """
        automatically triggered by the Isaac Sim framework at the very beginning of simulation
        """
        # retrieve cart and pole joint indices
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")

        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def reset_idx(self, env_ids):
        """
        randomized state to ensure that the RL policy can learn to perform the task from arbitrary starting states
        """
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
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
        actions = actions.to(self._device)

        # compute forces from the actions
        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        # apply actions to all of the environments
        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self) -> dict:
        """
        implement our observations
        """
        # retrieve joint positions and velocities
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        # extract joint states for the cart and pole joints
        self.cart_pos = dof_pos[:, self._cart_dof_idx]
        self.cart_vel = dof_vel[:, self._cart_dof_idx]
        self.pole_pos = dof_pos[:, self._pole_dof_idx]
        self.pole_vel = dof_vel[:, self._pole_dof_idx]

        # populate the observations buffer
        self.obs_buf[:, 0] = self.cart_pos
        self.obs_buf[:, 1] = self.cart_vel
        self.obs_buf[:, 2] = self.pole_pos
        self.obs_buf[:, 3] = self.pole_vel

        # construct the observations dictionary and return
        observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}}
        return observations

    def calculate_metrics(self) -> None:
        """
        compute the reward for the task
        """
        # use states from the observation buffer to compute reward
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        # define the reward function based on pole angle and robot velocities
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.5 * torch.abs(pole_vel)
        # penalize the policy if the cart moves too far on the rail
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # penalize the policy if the pole moves beyond 90 degrees
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # assign rewards to the reward buffer
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        """
        determine the environments need to be reset
        """
        cart_pos = self.obs_buf[:, 0]
        pole_pos = self.obs_buf[:, 2]

        # check for which conditions are met and mark the environments that satisfy the conditions
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        # assign the resets to the reset buffer
        self.reset_buf[:] = resets
