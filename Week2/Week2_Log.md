# Project Logbook: Bipedal Robot Environment in Isaac Gym

**Date:** [Enter Date]  
**Developer:** Danush Pravin Kumar  
**Project:** Custom Bipedal Robot Environment using Isaac Gym  

---

## Objective  
To create a physics-based reinforcement learning environment for a custom bipedal robot using NVIDIA Isaac Gym and train it using RL algorithms.

---

## Environment Development Summary

### Frameworks Used
- Isaac Gym (Preview 3)  
- `gym`, `gym.spaces`, `torch`, `numpy`  
- `stable-baselines3` for training (initial algorithm: PPO)

### Robot Setup
- Custom URDF: `Slider_Trial5.urdf`
- Loaded into Isaac Gym from local asset path
- `fix_base_link = False` to allow free motion
- DOF limits dynamically read and scaled for action space
- Used position control (`DOF_MODE_POS`) for joint movement

### Simulation Parameters
- Timestep: 1/60 sec  
- Gravity: -9.81 m/s²  
- Ground plane added with Z-axis as up direction  
- One actor created in a single environment  

### Observation Space
- `[joint positions, joint velocities, base linear velocity, base angular velocity, base orientation quaternion]`
- Shape: `(num_dofs * 2 + 10,)`  
- Dynamically constructed based on DOFs

### Action Space
- Continuous `Box` space in `[-1, 1]` for each DOF
- Scaled internally to match actual DOF limits using interpolation

### Reward Function
- Encourages forward velocity along X-axis
- Adds a bonus/penalty based on whether robot is upright (`height > 0.4`)
- `reward = 1.0 * velocity + 0.1 * upright_bonus`

### Termination Condition
- Ends episode if base height drops below `0.3` (robot fell)

---

## Rendering
- Viewer setup using Isaac Gym's `create_viewer` and `draw_viewer`
- `render()` method added for manual visualization and debugging

---

## Training Attempt 1

### Algorithm  
- **Proximal Policy Optimization (PPO)**  
- **Library:** `stable-baselines3`  

### Result  
- Agent displayed **twitching and unstable movements**
- **No consistent walking behavior** observed
- Agent falls down quickly in most episodes
- Likely due to poor reward shaping, insufficient policy complexity, or inadequate observation scaling

---

## Next Steps / To-Dos
- [ ] Improve reward function with more shaping terms (e.g., energy penalty, orientation alignment, joint effort)
- [ ] Normalize observations
- [ ] Implement better curriculum (start from fixed pose and increase complexity)
- [ ] Try alternative RL algorithms (e.g., SAC, PPO with recurrent policy, custom actor-critic)
- [ ] Consider using domain randomization or imitation learning
- [ ] Log and visualize metrics using TensorBoard or WandB
