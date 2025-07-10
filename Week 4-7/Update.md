# Bipedal Locomotion Deep RL Project Logbook

## Project Overview  
This project is part of my master's research at Imperial College London, focused on developing bipedal locomotion using Deep Reinforcement Learning in simulation. The goal is to enable a bipedal robot to balance, stand, and eventually walk using learned control policies.

## Progress Highlights  
- Early attempts struggled to maintain balance beyond 500 timesteps.  
- Transitioned to the NVIDIA Legged Gym simulation environment, which improved stability significantly.  
- Current agents can stand steadily and perform stepping-in-place behaviors.
- Integrated a Linear Inverted Pendulum Model (LIPM) for walking commands, enabling the robot to walk at a commanded speed.  
- Training uses Proximal Policy Optimization (PPO) as the core RL algorithm.

## Key Insights  
- Reward shaping and parameter tuning are crucial to improving learning speed and stability.  
- Small noise added to initial robot state encourages robust adaptive behaviors.  
- Combining classical control models (LIPM) with learned policies accelerates walking capabilities.

## Future Goals  
- Refine reward functions for faster learning and longer walking sequences.  
- Expand locomotion to varied terrains and dynamic environments. 
