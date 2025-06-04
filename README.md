# Slider Robot – Reinforcement Learning Project

## Introduction

This project involves the development of a reinforcement learning (RL) agent capable of learning to control a slider robot within a custom simulation environment. The primary aim is to explore fundamental control strategies using RL, and to build a framework that can be extended for more complex robotic behaviors in the future.

## Objective

The core objective of this project is to design, implement, and train an RL agent that can successfully control a slider robot by interacting with a simulated environment. The intended outcomes of the project include:

- Designing a stable and functional physics-based simulation for the slider robot.
- Developing a reward function that promotes desirable behavior such as balance, movement, or reaching a goal.
- Training an RL agent (e.g., using PPO or similar algorithms) to optimize its policy over time.
- Analyzing the learning behavior through metrics such as reward curves and performance plots.
- Building a robust foundation that can support future extensions like trajectory tracking, obstacle avoidance, or multi-agent interaction.


## Tools and Technologies

This project makes use of the following libraries and frameworks:

- **Python 3.x**
- **PyTorch**
- **Stable-Baselines3**
- **Gym** (custom environment API)
- **NumPy**
- **Matplotlib**
- **TensorBoard** (for training visualization)

## Current Progress

As of Week 3, the following milestones have been completed:

- The initial version of the environment was tested but found to contain several implementation bugs.
- A simplified and stable version of the environment was developed from scratch.
- The new environment successfully supports learning behavior.
- A PPO-based RL agent was trained, and reward progression over time was observed and plotted.
- The agent demonstrates early-stage learning, with rewards gradually increasing.

## Reward Visualization

A plot of the total reward per episode has been generated to visualize the learning trend. This indicates that the agent is improving its performance over time within the simplified environment.

## Future Work

Planned future developments include:

- Refinement of the reward function to improve learning stability and agent performance.
- Experimentation with different RL algorithms and hyperparameters.
- Integration of additional evaluation metrics to assess agent behavior.
- Extension of the environment to support more complex tasks or constraints.
- Creation of video demonstrations and visual simulations of the agent's progress.

## Author

**Danush**  
MSc Design Engineering  
Imperial College London

## License

This project is intended for academic and research purposes only. License and usage terms may be added at a later stage.

---

*This project represents a foundational step in applying reinforcement learning to robotic control systems, with a focus on iterative development, debugging, and performance analysis.*
