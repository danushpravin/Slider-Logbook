from isaacgym import gymapi, gymtorch
from gym import spaces
import numpy as np
import torch
import gym

class MetricsLogger:
    def __init__(self):
        self.episode_data = []
        self.current_episode = []

    def log(self, reward, velocity, height, upright_bonus, action, clipped_action):
        self.current_episode.append({
            "reward": reward,
            "velocity": velocity,
            "height": height,
            "upright_bonus": upright_bonus,
            "action": action,
            "clipped_action": clipped_action
        })

    def end_episode(self):
        if not self.current_episode:
            return {}

        episode = self.current_episode
        self.episode_data.append(episode)
        self.current_episode = []

        total_reward = sum(step["reward"] for step in episode)
        avg_velocity = np.mean([step["velocity"] for step in episode])
        avg_height = np.mean([step["height"] for step in episode])
        action_magnitude = np.mean([np.linalg.norm(step["action"]) for step in episode])

        return {
            "ep_reward_total": total_reward,
            "avg_velocity": avg_velocity,
            "avg_height": avg_height,
            "avg_action_magnitude": action_magnitude,
            "episode_length": len(episode)
        }

class BipedalEnv(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super(BipedalEnv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viewer = None
        
        # === Isaac Gym setup ===
        self.gym = gymapi.acquire_gym()
        self.logger=MetricsLogger()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        self.gymapi = gymapi
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        assert self.sim is not None, "❌ Failed to create sim"
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        env = self.gym.create_env(self.sim, gymapi.Vec3(-1.0, 0.0, 0.0), gymapi.Vec3(1.0, 0.0, 1.0), 1)
        self.env = env

        asset_root = "/home/danush/Desktop/Projects/Slider_Trial5/urdf"
        asset_file = "Slider_Trial5.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0)
        self.actor_handle = self.gym.create_actor(env, self.asset, pose, "robot", 0, 1)

        self.num_dofs = self.gym.get_asset_dof_count(self.asset)
        self.dof_props = self.gym.get_asset_dof_properties(self.asset)
        device = self.device
        self.lower_limits = torch.tensor([d['lower'] for d in self.dof_props], dtype=torch.float32, device=device)
        self.upper_limits = torch.tensor([d['upper'] for d in self.dof_props], dtype=torch.float32, device=device)
        self.gym.set_actor_dof_properties(self.env, self.actor_handle, self.dof_props)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_dofs,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs*2 + 10,), dtype=np.float32)

        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)

        self.max_episode_steps = max_episode_steps
        self.episode_step_count = 0


        self.reset()

    def reset(self):
        # Reset root state (pos, orientation, velocities)
        self.episode_reward = 0
        self.episode_length = 0
        init_root_state = torch.tensor([
            0, 0, 0.65,      # position x,y,z
            0, 0, 0, 1,      # orientation quaternion
            0, 0, 0,       # linear velocity x,y,z
            0, 0, 0          # angular velocity x,y,z
        ], dtype=self.root_states.dtype, device=self.device)

        self.root_states[0, :] = init_root_state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        # Reset DOF positions and velocities
        default_positions = torch.zeros(self.num_dofs, device=self.device, dtype=self.dof_states.dtype)
        # Optionally set some default joint positions here, e.g.
        default_positions[2] = 0.0185
        default_positions[3] = 0.0
        default_positions[7] = -0.0185
        default_positions[8] = 0.0

        self.dof_states[:, 0] = default_positions
        self.dof_states[:, 1] = 0.0  # zero velocities
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.episode_step_count = 0
        return self._get_obs()

    def _get_obs(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        joint_pos = self.dof_states[:, 0].cpu().numpy()
        joint_vel = self.dof_states[:, 1].cpu().numpy()
        root_state = self.root_states[0]

        base_vel = root_state[7:10].cpu().numpy()
        base_ang_vel = root_state[10:13].cpu().numpy()
        base_rot = root_state[3:7].cpu().numpy()

        obs = np.concatenate([joint_pos, joint_vel, base_vel, base_ang_vel, base_rot])
        return obs.astype(np.float32)

    def apply_action(self, action):
        # Clip and scale the action to joint limits
        action = np.clip(action, -1.0, 1.0)  # make sure it's within valid range
        action = torch.tensor(action, dtype=torch.float32, device=self.device)

        # Scale action from [-1, 1] to [lower_limit, upper_limit]
        target_dof_positions = 0.5 * (action + 1.0) * (self.upper_limits - self.lower_limits) + self.lower_limits

        #print(f"[DEBUG] Scaled DOF targets: {target_dof_positions.cpu().numpy()}")
        # Set target DOF positions (using DOF_MODE_POS as defined in your asset options)
        targets = self.dof_states[:, 0].clone()
        targets[:] = target_dof_positions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        #print(f"[DEBUG] Targets tensor set to sim: {targets.cpu().numpy()}")

        # Step the simulator forward by one step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Update tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_reward(self, action, done, info, steps):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_state = self.root_states[0]

        pelvis_height = root_state[2].item()
        reward = pelvis_height
        if done and pelvis_height < 0.25:
            reward -= 1.0 

        return reward

    def check_termination(self):
        root_state = self.root_states[0]
        height = root_state[2].item()

        terminated = bool(height < 0.25)  # robot fell
        return terminated
    
    def step(self, action):
        # Apply action to the simulator
        #print(f"[DEBUG] PPO action input: {action}")
        self.apply_action(action)

        # Increment step count
        self.episode_step_count += 1

        # Get next observation
        obs = self._get_obs()

        # Check for termination
        done = self.check_termination() or self.episode_step_count >= self.max_episode_steps

        # Info dictionary (optional debugging data)
        info = {}

        # Compute reward
        reward = self.compute_reward(action, done, info, self.episode_step_count)
        
        root_state = self.root_states[0]
        velocity = root_state[7].item()
        height = root_state[2].item()
        upright_bonus = 1.0 if height > 0.4 else -1.0

        clipped_action = np.clip(action, -1.5, 1.5)
        self.logger.log(
        reward=reward,
        velocity=velocity,
        height=height,
        upright_bonus=upright_bonus,
        action=action,
        clipped_action=clipped_action
        )

        if done:
            episode_summary = self.logger.end_episode()
            print(f"[Episode Done] Steps: {episode_summary['episode_length']}, "
                f"Total Reward: {episode_summary['ep_reward_total']:.2f}, "
                f"Avg Vel: {episode_summary['avg_velocity']:.2f}, "
                f"Avg Height: {episode_summary['avg_height']:.2f}, "
                f"Avg |Action|: {episode_summary['avg_action_magnitude']:.2f}")
            info.update(episode_summary)

        self.episode_reward += reward
        self.episode_length += 1

        if done:
            info["episode"] = {
            "r": self.episode_reward,
            "l": self.episode_length
         }


        return obs, reward, done, info


    def create_viewer(self):
        if self.viewer is None:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

    def render(self):
        if self.viewer is None:
         # Initialize your viewer here, e.g.:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())  # or however you do it in IsaacGym
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

