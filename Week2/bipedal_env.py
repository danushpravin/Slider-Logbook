from isaacgym import gymapi, gymtorch
from gym import spaces
import numpy as np
import torch
import gym

class BipedalEnv(gym.Env):
    def __init__(self, max_episode_steps=1000):
        super(BipedalEnv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viewer = None
        
        # === Isaac Gym setup ===
        self.gym = gymapi.acquire_gym()

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        assert self.sim is not None, "❌ Failed to create sim"

        #self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        #assert self.viewer is not None, "❌ Failed to create viewer"

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


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_dofs,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dofs*2 + 10,), dtype=np.float32)

        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.max_episode_steps = max_episode_steps
        self.episode_step_count = 0

        self.reset()

    def reset(self):
        self.episode_step_count = 0

        self.dof_states[:, 0] = 0.0
        self.dof_states[:, 1] = 0.0
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.root_states[0, 0:3] = torch.tensor([0.0, 0.0, 0.63], device=self.root_states.device)
        self.root_states[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.root_states.device)
        self.root_states[0, 7:13] = torch.zeros(6, device=self.root_states.device)

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

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
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)

        action_clipped = torch.clip(action, -1.0, 1.0)
        target_dof = 0.5 * (action_clipped + 1.0) * (self.upper_limits - self.lower_limits) + self.lower_limits
        #target_dof = target_dof.cpu().numpy()
        self.gym.set_actor_dof_position_targets(self.env, self.actor_handle, target_dof.cpu().numpy())

    def compute_reward(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_state = self.root_states[0]

        forward_velocity = root_state[7].item()  # velocity in X
        height = root_state[2].item()
        upright_bonus = 1.0 if height > 0.4 else -1.0

        reward = 1.0 * forward_velocity + 0.1 * upright_bonus
        return reward

    def check_termination(self):
        root_state = self.root_states[0]
        height = root_state[2].item()

        terminated = bool(height < 0.3)  # robot fell
        return terminated

    def step(self, action):
        self.episode_step_count += 1

        self.apply_action(action)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        obs = self._get_obs()
        reward = self.compute_reward()
        terminated = self.check_termination()
        truncated = self.episode_step_count >= self.max_episode_steps
        info = {}
        done = terminated or truncated
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

