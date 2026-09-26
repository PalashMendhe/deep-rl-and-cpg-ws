import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

'''
#-------------------------------------------------------------#----#
|Angular velocity of each joint                               |12  |
|Joint Angles                                                 |12  |
|Previous joint actions                                       |12  |
|Linear velocity of the whole body in x,y and z direction     |3   |
|Pitch, yaw and roll of the body                              |3   |
|Bot orientation wrt ground                                   |3   |
|Target velocities                                            |3   |
|3D Target velocity vector for the bot to achieve.            |3   |
#-------------------------------------------------------------#----#
'''

import mujoco
from src.terrain.config import TerrainConfig
from typing import NamedTuple


class JumpEvent(NamedTuple):
    apogee_z: float
    clearance_m: float
    hurdle_hit: bool
    success: bool


class go1_env(MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file = '/home/plsh/rl_env2/mujoco_menagerie/unitree_go1/scene.xml',
                 large_step_penalty = 0.5,                                #Penalize the bot when it takes a step too large.
                 contact_penalty = 0.005,                                 #Penalize bot when the external contact force is too large.
                 timestep_reward = 0.001,                                 #Reward the bot for each timestep it is healthy.
                 forward_reward = 3.0,                                    #Reward the bot for each time it moves forward.
                 contact_force = False,                                   #External acting force.
                 contact_force_range = (-1, 1),                           #Range amount for penalty according to contact_penalty.
                 terminate_when_unhealthy = True,                         #Issued when the torso of the body is not in healthy_z_range.
                 healthy_z_range = (0.25, 0.45),                          #The bot is considered healthy in this range.
                 exclude_current_position_from_observation = True,        #For position agnostic behaviour.
                 reset_noise_scale = 0.05,                                #Scale for random perturbations of initial position and velocity.
                 target_velocity = np.array([0.6, 0.0, 0.0]),             #Target velocity for the bot to achieve.
                 _last_action = None,                                        #Last action taken (defaults to zeros of action dim).
                 terrain_config = None,                                   #TerrainConfig or None (flat default).
                 include_ext_obs = True,                                  #Append 4-dim hurdle-relative extras (52->56).
                 jump_reward_w = 2.0,                                     #Weight for airtime/clearance bonus.
                 hurdle_hit_penalty = 5.0,                                #Penalty on hurdle contact.
                 cpg_mode = "off",                                        #off|fixed_residual|parametric|hopf.
                 residual_scale = 0.25,                                   #RL correction scale in CPG modes.
                 cpg_freq_hz = 2.0,                                       #Open-loop trot frequency.
                 cpg_k_fb = 0.5,                                          #Hopf foot-contact feedback gain.
                 cpg_coupling = 2.0,                                      #Hopf Kuramoto coupling strength.
                 landing_bonus_w = 1.0,                                   #Landing-stability bonus weight (Step 6).
                 hurdle_hit_terminate = False,                            #End episode on hurdle contact (else penalty-only).
                 **kwargs
                 ):
      obs_shape = 56 if include_ext_obs else 52
      action_dim = 18 if cpg_mode in ("parametric", "hopf") else 12
      if _last_action is None:
          _last_action = np.zeros(action_dim)
      
      # Initialize the EzPickle utility to enable easy serialization and deserialization of the environment's state.  
      utils.EzPickle.__init__(self, xml_file,
                              large_step_penalty,
                              contact_penalty,
                              timestep_reward,
                              forward_reward,
                              contact_force,
                              contact_force_range,
                              terminate_when_unhealthy,
                              healthy_z_range,
                              exclude_current_position_from_observation,
                              reset_noise_scale,
                              target_velocity,
                              _last_action,
                              terrain_config,
                              include_ext_obs,
                              jump_reward_w,
                              hurdle_hit_penalty,
                              cpg_mode,
                              residual_scale,
                              cpg_freq_hz,
                              cpg_k_fb,
                              cpg_coupling,
                              landing_bonus_w,
                              hurdle_hit_terminate,
                              **kwargs
                                )

      self._xml_file = xml_file
      self._large_step_penalty = large_step_penalty
      self._contact_penalty = contact_penalty
      self._timestep_reward = timestep_reward
      self._forward_reward = forward_reward
      self._contact_force = contact_force
      self._contact_force_range = contact_force_range
      self._terminate_when_unhealthy = terminate_when_unhealthy
      self._healthy_z_range = healthy_z_range
      self._exclude_current_position_from_observation = exclude_current_position_from_observation
      self._reset_noise_scale = reset_noise_scale
      self._target_velocity = target_velocity
      self._last_action = np.asarray(_last_action, dtype=np.float64).flatten()
      self._action_dim = int(action_dim)
      self._terrain = terrain_config if terrain_config is not None else TerrainConfig(mode="flat")
      self._include_ext_obs = include_ext_obs
      self._jump_reward_w = jump_reward_w
      self._hurdle_hit_penalty = hurdle_hit_penalty
      self._hurdle_x = float(self._terrain.hurdle_x)
      self._hurdle_hit_latched = False
      self._last_mod = np.zeros(6)
      self._cpg_mode = cpg_mode
      self._residual_scale = float(residual_scale)
      self._cpg_freq_hz = float(cpg_freq_hz)
      self._cpg_k_fb = float(cpg_k_fb)
      self._cpg_coupling = float(cpg_coupling)
      self._landing_bonus_w = float(landing_bonus_w)
      self._hurdle_hit_terminate = bool(hurdle_hit_terminate)
      self._landed_clean = False
      self._cpg = None
      self._cpg_phase = 0.0

      observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
      render_mode = kwargs.pop("render_mode", None)
      MujocoEnv.__init__(self, model_path=self._xml_file, frame_skip=5, observation_space=observation_space, render_mode=render_mode, **kwargs)
      # MujocoEnv.__init__ resets action_space from the model (12 actuators);
      # re-apply AFTER super().__init__ so parametric/hopf keep 18-dim.
      self.action_space = Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float64)
      home_key_id = 0
      self._home_qpos = self.model.key_qpos[home_key_id].copy()
      self._home_qvel = self.model.key_qvel[home_key_id].copy()
      self._init_cpg()


    def _init_cpg(self):
      # Lazy import so flat-mode envs never pay CPG import cost.
      if self._cpg_mode == "off":
          self._cpg = None
          return
      from src.cpg.fixed_trot import FixedTrotCPG
      from src.cpg.parametric import ParametricCPG
      from src.cpg.hopf import HopfCPGNetwork
      if self._cpg_mode == "fixed_residual":
          self._cpg = FixedTrotCPG(freq_hz=self._cpg_freq_hz)
      elif self._cpg_mode == "parametric":
          self._cpg = ParametricCPG(freq_hz=self._cpg_freq_hz)
      elif self._cpg_mode == "hopf":
          self._cpg = HopfCPGNetwork(freq_hz=self._cpg_freq_hz,
                                      k_fb=self._cpg_k_fb,
                                      coupling_strength=self._cpg_coupling)
      else:
          raise ValueError(f"unknown cpg_mode={self._cpg_mode}")
      self._cpg.reset(0.0)
      self._cpg_phase = 0.0

    def _cpg_targets(self, action):
      """Return (control_target, cpg_phase, jump_mod) for current mode.

      off: legacy home + action*0.25. fixed_residual: trot + residual*scale.
      parametric/hopf: action split into [12 residuals + 6 mod]; hopf also
      injects foot-contact feedback.
      """
      action = np.asarray(action, dtype=np.float64).flatten()
      # Pad short callers (e.g. 12-dim random samples) so every mode is robust.
      if action.shape[0] < self._action_dim:
          action = np.pad(action, (0, self._action_dim - action.shape[0]))
      if self._cpg_mode == "off" or self._cpg is None:
          control_target = self._home_qpos[7:] + action[:12] * 0.25
          self._last_mod = np.zeros(6)
          return control_target, self._cpg_phase, 0.0
      if self._cpg_mode == "fixed_residual":
          base = self._cpg.step(self.dt)
          control_target = base + action[:12] * self._residual_scale
          self._cpg_phase = float(self._cpg.phase)
          self._last_mod = np.zeros(6)
          return control_target, self._cpg_phase, 0.0
      # parametric + hopf share the 18-dim [residual(12) | mod(6)] split.
      residual = action[:12] * self._residual_scale
      mod = action[12:18]
      self._last_mod = mod.copy()
      if self._cpg_mode == "parametric":
          params = self._cpg.params_from_action(mod)
          self._cpg.phase += 2.0 * np.pi * params.frequency_hz * self.dt
          base = self._cpg.rollout_targets(params, self._cpg.phase)
          self._cpg_phase = float(self._cpg.phase)
          return base + residual, self._cpg_phase, float(params.jump_boost)
      # hopf: contact feedback = actual contact minus trot-expected contact.
      # Leg order is [FR, FL, RR, RL]; diagonal pair (FL,RR) shares phase_1.
      contacts = self._get_foot_contacts().astype(np.float64)
      phase01 = (self._cpg_phase / (2.0 * np.pi)) % 1.0
      swing = 1.0 if np.sin(2.0 * np.pi * phase01) > 0 else 0.0
      expected = np.array([swing, 1.0 - swing, 1.0 - swing, swing])
      feedback = contacts - expected
      base = self._cpg.step(self.dt, feedback=feedback, mod=mod)
      self._cpg_phase = float(self._cpg.phase)
      return base + residual, self._cpg_phase, float(self._cpg.params.jump_boost)


    def _get_obs(self):
      #Observation space
      torso_pos = self.data.qpos[:3] #initially 3
      torso_quat = self.data.qpos[3:7] # initilly 4
      joint_pos = self.data.qpos[7:19] # final 12
      # Last-action memory stays 12-dim (residuals) in every mode so the
      # 52-dim proprio prefix is identical across cpg off/fixed/parametric/hopf.
      last_joint_pos = np.asarray(self._last_action, dtype=np.float64).flatten()[:12]
      if last_joint_pos.shape[0] < 12:
          last_joint_pos = np.pad(last_joint_pos, (0, 12 - last_joint_pos.shape[0]))
      torso_lin_vel = self.data.qvel[:3] # final 3
      torso_ang_vel = self.data.qvel[3:6] # final 3
      joint_vel = self.data.qvel[6:18] # final 12

      quat_scipy_order = [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
      euler = R.from_quat(quat_scipy_order).as_euler('xyz', degrees=False) # final 3
      rotation_matrix = R.from_quat(quat_scipy_order).as_matrix()
      gravity_world = np.array([0,0,-1])

      gravity_local = rotation_matrix.T @ gravity_world # final 3

      torso_pos_for_obs = torso_pos[2:] if self._exclude_current_position_from_observation else torso_pos # final 1
      position = np.concatenate((torso_pos_for_obs, euler, joint_pos, gravity_local, last_joint_pos))
      velocity = np.concatenate((torso_lin_vel, torso_ang_vel, joint_vel))

      proprio = np.concatenate((position, velocity ,self._target_velocity))
      if not self._include_ext_obs:
          return proprio
      return np.concatenate((proprio, self._get_ext_obs()))

    def _has_hurdle_geom(self):
      try:
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle")
          return True
      except Exception:
          return False

    def _get_ext_obs(self):
      # Hurdle-relative extras: [dist_to_hurdle_x, hurdle_height_norm, trunk_z - hurdle_top, vertical_vel_z]
      trunk_xy = self.get_body_com("trunk")[:2]
      trunk_z = float(self.data.qpos[2])
      hurdle_active = self._has_hurdle_geom() and self._terrain.mode in ("hurdle", "mixed")
      if hurdle_active:
          dist_x = float(self._hurdle_x - trunk_xy[0])
          height_norm = float(self._terrain.hurdle_height_m / 0.15)
          clearance = float(trunk_z - self._terrain.hurdle_top_z)
      else:
          dist_x = 10.0
          height_norm = 0.0
          clearance = float(trunk_z)
      dist_x = float(np.clip(dist_x, -2.0, 10.0) / 5.0)
      vz = float(self.data.qvel[2])
      return np.array([dist_x, height_norm, clearance, vz], dtype=np.float64)

    def _get_privileged_obs(self):
      # Asymmetric-critic extras: ext_obs + foot contacts + terrain height under trunk.
      contacts = self._get_foot_contacts().astype(np.float64)
      trunk_z = float(self.data.qpos[2])
      return np.concatenate((self._get_ext_obs(), contacts, np.array([trunk_z], dtype=np.float64)))

    def _reposition_bumps(self):
      """Per-episode bump variety for rough/mixed stages (Step 5).

      Jitters each bump's x-centre and scales its height via
      TerrainConfig.sample_bump_poses. No-op when the geoms are absent
      (flat scene) or the terrain mode has no bumps.
      """
      if self._terrain.mode not in ("rough", "mixed", "hurdle"):
          return
      try:
          poses = self._terrain.sample_bump_poses(self.np_random)
      except Exception:
          return
      for name, x, h_scale in poses:
          try:
              gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
          except Exception:
              continue
          size = np.array(self.model.geom_size[gid], dtype=np.float64)
          # XML bumps are boxes centred at z=size[2]; scale z half-extent so
          # the bump stays seated on the floor (pos_z = new half-height).
          new_hz = float(max(0.005, size[2] * h_scale))
          size[2] = new_hz
          self.model.geom_size[gid] = size
          pos = np.array(self.model.geom_pos[gid], dtype=np.float64)
          pos[0] = float(x)
          pos[2] = new_hz
          self.model.geom_pos[gid] = pos

    def _reposition_hurdle(self, x):
      self._hurdle_x = float(x)
      try:
          hurdle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle")
      except Exception:
          return
      size = self._terrain.hurdle_geom_size()
      pos = self._terrain.hurdle_geom_pos(self._hurdle_x)
      self.model.geom_size[hurdle_id] = np.array(size)
      self.model.geom_pos[hurdle_id] = np.array(pos)

    def check_hurdle_collision(self):
      try:
          hurdle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hurdle")
      except Exception:
          return False
      for i in range(self.data.ncon):
          con = self.data.contact[i]
          if con.geom1 == hurdle_id or con.geom2 == hurdle_id:
              return True
      return False

    def reset_model(self):
      qpos = self._home_qpos.copy()         #Joint angles and position of the bot, initialized to home position.
      qvel = self._home_qvel.copy()         #Joint velocities of the bot, initialized to home velocity.

      qpos[7:] += self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=12)
      qvel += self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=self.model.nv)

      self.set_state(qpos, qvel)
      self._last_action = np.zeros(self._action_dim)
      self._last_mod = np.zeros(6)
      # Resample obstacle placement per episode (seeded); no-op on flat scene.
      if self._has_hurdle_geom() and self._terrain.mode in ("hurdle", "mixed"):
          self._reposition_hurdle(self._terrain.sample_hurdle_x(self.np_random))
      elif self._has_hurdle_geom():
          self._reposition_hurdle(self._terrain.hurdle_x)
      self._reposition_bumps()
      self._hurdle_hit_latched = False
      self._landed_clean = False
      if self._cpg is not None:
          self._cpg.reset(0.0)
      self._cpg_phase = 0.0
      return self._get_obs()

    def _get_foot_contacts(self):
    # Map to your specific sensor IDs in MuJoCo
      fl_force = self.data.sensor('FL_Touch').data[0]
      fr_force = self.data.sensor('FR_Touch').data[0]
      rl_force = self.data.sensor('RL_Touch').data[0]
      rr_force = self.data.sensor('RR_Touch').data[0]

      # Apply a threshold (e.g., > 1.0 N) to register true ground support
      return np.array([fl_force, fr_force, rl_force, rr_force]) > 1.0

    def step(self, action):
        xy_position_before = self.get_body_com("trunk")[:2].copy()
        control_target, cpg_phase, jump_mod = self._cpg_targets(action)
        action_full = np.asarray(action, dtype=np.float64).flatten()
        if action_full.shape[0] < self._action_dim:
            action_full = np.pad(action_full, (0, self._action_dim - action_full.shape[0]))
        action12 = action_full[:12]
        
        # Simulate step
        self.do_simulation(control_target, self.frame_skip)
        
        xy_position_after = self.get_body_com("trunk")[:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        
        last12 = np.asarray(self._last_action, dtype=np.float64).flatten()[:12]
        if last12.shape[0] < 12:
            last12 = np.pad(last12, (0, 12 - last12.shape[0]))
        action_smoothness_penalty = np.mean(np.square(action12 - last12))

        # --- 1. EXTRACT STATE (Orientation & Velocity) ---
        torso_quat = self.data.qpos[3:7] 
        quat_scipy_order = [torso_quat[1], torso_quat[2], torso_quat[3], torso_quat[0]]
        euler = R.from_quat(quat_scipy_order).as_euler('xyz', degrees=False)
        
        roll = euler[0]
        pitch = euler[1]
        current_yaw = euler[2]
        
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        # Transform global velocity to local frame
        local_x_vel = xy_velocity[0] * cos_yaw + xy_velocity[1] * sin_yaw
        local_y_vel = -xy_velocity[0] * sin_yaw + xy_velocity[1] * cos_yaw
        
        velocity_reward = float(np.exp(
            -4.0 * (local_x_vel - float(self._target_velocity[0])) ** 2)) 

        # --- 2. NAVIGATION PACKAGE (Active from Step 0) ---
        target_yaw = 0.0
        heading_error = np.arctan2(np.sin(current_yaw - target_yaw), np.cos(current_yaw - target_yaw))
        
        yaw_rate = self.data.qvel[5]
        
        heading_penalty = 0.5 * (heading_error ** 2)
        yaw_penalty = 0.3 * (yaw_rate ** 2)
        drift_penalty = 0.5 * (local_y_vel ** 2)
        
        navigation_penalty = drift_penalty + yaw_penalty + heading_penalty

        # --- 3. KINEMATIC PACKAGE (Scaled by Curriculum) ---
        # A. Posture Penalty
        posture_penalty = 3.0 * (roll**2 + pitch**2)
        
        # B. Foot Clearance Penalty
        fl_z = self.data.site('FL').xpos[2]
        fr_z = self.data.site('FR').xpos[2]
        rl_z = self.data.site('RL').xpos[2]
        rr_z = self.data.site('RR').xpos[2]

        diagonal_1_height = fl_z + rr_z
        diagonal_2_height = fr_z + rl_z

        target_height_diff = 0.10
        actual_height_diff = abs(diagonal_1_height - diagonal_2_height)
        # Hinge-style: reward achieved swing lift, never punish standing
        # support (diff ~ 0). Full bonus once the diagonal pair lifts 0.10 m.
        clearance_penalty = 5.0 * max(0.0, target_height_diff - actual_height_diff) ** 2

        # C. Trot Rhythm Penalty
        FL, FR, RL, RR = self._get_foot_contacts()
        diagonal_1 = int(FL) + int(RR)
        diagonal_2 = int(FR) + int(RL)
        
        is_valid_trot = (diagonal_1 == 2 and diagonal_2 == 0) or (diagonal_1 == 0 and diagonal_2 == 2)
        trot_penalty_flag = 0.0 if is_valid_trot else 1.0

        # D. Effort & Velocity Limits
        effort_penalty = 0.005 * np.sum(np.square(action12))
        leg_joint_velocities = self.data.qvel[6:]
        joint_vel_penalty = 0.0005 * np.sum(np.square(leg_joint_velocities))

        # E. Calculate Raw Kinematic Penalty (Must be summed before scaling)
        raw_kinematic_penalty = (
            (2.0 * trot_penalty_flag) 
            + clearance_penalty 
            + posture_penalty 
            + effort_penalty 
            + joint_vel_penalty
        )

        # F. Apply Curriculum Multiplier
        current_form_weight = getattr(self, "form_weight", 0.0) # Defaults to 0.0 early in training
        scaled_kinematic_penalty = raw_kinematic_penalty * current_form_weight

        # --- 4. JUMP PACKAGE (hurdle / mixed terrain only) ---
        hurdle_active = self._has_hurdle_geom() and self._terrain.mode in ("hurdle", "mixed")
        trunk_z = float(self.data.qpos[2])
        vz = float(self.data.qvel[2])
        trunk_x = float(xy_position_after[0])
        dist_to_hurdle = float(self._hurdle_x - trunk_x)
        clearance_m = float(trunk_z - self._terrain.hurdle_top_z)
        hurdle_hit = bool(hurdle_active and self.check_hurdle_collision())
        if hurdle_hit:
            self._hurdle_hit_latched = True
        contacts_now = self._get_foot_contacts()
        airborne = bool(hurdle_active and not bool(np.any(contacts_now)))
        approaching = bool(hurdle_active and 0.0 < dist_to_hurdle < 1.0)
        approach_bonus = float(np.clip(vz, 0.0, 1.5) * local_x_vel) if approaching else 0.0
        airtime_bonus = float(np.clip(vz, 0.0, 1.5) + max(0.0, clearance_m)) if airborne else 0.0
        jump_bonus = approach_bonus + airtime_bonus
        hurdle_penalty = self._hurdle_hit_penalty if hurdle_hit else 0.0
        crossed = bool(hurdle_active and dist_to_hurdle < -0.3)
        jump_success = bool(crossed and not self._hurdle_hit_latched)
        # Step-6 landing-stability bonus: on first touchdown after a clean
        # crossing, reward staying tall, level, calm and moving forward.
        landing_bonus = 0.0
        if (hurdle_active and crossed and not self._hurdle_hit_latched
                and not self._landed_clean and bool(np.any(contacts_now))):
            height_ok = float(np.clip(1.0 - abs(float(self.data.qpos[2]) - 0.30) / 0.10, 0.0, 1.0))
            level_ok = float(np.clip(1.0 - (abs(roll) + abs(pitch)) / 0.6, 0.0, 1.0))
            calm_ok = float(np.clip(1.0 - abs(vz) / 1.5, 0.0, 1.0))
            fwd_ok = float(np.clip(local_x_vel / 0.6, 0.0, 1.0))
            feet_ok = float(np.sum(contacts_now)) / 4.0
            landing_bonus = float(self._landing_bonus_w
                                  * (0.35 * height_ok + 0.30 * level_ok + 0.15 * calm_ok
                                     + 0.10 * fwd_ok + 0.10 * feet_ok))
            self._landed_clean = True
        current_height = self.data.qpos[2]
        
        reward = (
            self._forward_reward * velocity_reward
            + self._timestep_reward  # survival bonus for staying healthy
            - self._contact_penalty * action_smoothness_penalty
            - navigation_penalty
            - scaled_kinematic_penalty
            + self._jump_reward_w * jump_bonus
            + landing_bonus
            - hurdle_penalty
        )
        
        reward = np.clip(reward, -10.0, 10.0)

        # --- 5. TERMINAL CONDITIONS ---
        terminated = False
        if self._terminate_when_unhealthy and (current_height < self._healthy_z_range[0] or current_height > self._healthy_z_range[1]):
            terminated = True
            reward = -2.0
        elif self._hurdle_hit_terminate and hurdle_hit:
            terminated = True
            reward = -2.0

        # --- 6. STATE UPDATES & LOGGING ---
        self._last_action = action_full.copy()
        observation = self._get_obs()

        info_dict = {
            # Core Movement
            "velocity_reward": velocity_reward,
            "local_x_velocity": local_x_vel,
            "local_y_velocity": local_y_vel,
            "current_height": current_height,
            "distance_from_origin": np.linalg.norm(xy_position_after),
            
            # Penalties (Exposed for tracking dashboard)
            "action_smoothness_penalty": action_smoothness_penalty,
            "navigation_penalty_total": navigation_penalty,
            "kinematic_penalty_scaled": scaled_kinematic_penalty,
            
            # Breakdown of Kinematics
            "raw_trot_flag": trot_penalty_flag,
            "raw_clearance_error": actual_height_diff,
            "raw_posture_penalty": posture_penalty,
            "active_form_weight": current_form_weight,

            # Obstacle / jump tracking
            "dist_to_hurdle": dist_to_hurdle,
            "clearance_m": clearance_m,
            "hurdle_hit": hurdle_hit,
            "jump_bonus": jump_bonus,
            "landing_bonus": landing_bonus,
            "jump_success": jump_success,

            # CPG state
            "cpg_phase": cpg_phase,
            "cpg_freq": self._cpg_freq_hz,
            "cpg_mode": self._cpg_mode,
            "jump_boost": jump_mod,
            "last_mod": self._last_mod.copy(),
        }

        self.velocity_reward = velocity_reward
        self.action_smoothness_penalty = action_smoothness_penalty
        self.reward = reward
        self.terminated = terminated
        self.truncated = False

        return observation, reward, terminated, False, info_dict