# Log Entry - URDF Simulation Debugging in Isaac Gym
**Date:** 2025-05-21  
**Author:** Danush Pravin Kumar  
**Project:** Slider_Trial5 Robot Simulation  

---

## Objective
To load and simulate the `Slider_Trial5` robot model using its provided URDF file in NVIDIA Isaac Gym and ensure it initializes and behaves correctly in the simulated environment.

---

## Tools & Resources
- **Simulation Platform:** NVIDIA Isaac Gym (Python API)  
- **URDF Visualization Tool:** URDF Visualizer by MorningFrog (VSCode extension)  
- **Asset:** `Slider_Trial5.urdf` (provided by project supervisor)  
- **Environment:** Ubuntu 20.04, Python 3.8

---

## Procedure & Observations

### 1. Initial Setup & Load
- Implemented Python code using `gymapi` and `gymutil` modules to load the robot asset and place it in a basic simulation.
- Verified gravity, timestep, and PhysX settings.
- Set camera view and environment bounds.
- Observed that the robot consistently exhibited a strange “death animation” — i.e., the structure would collapse in the exact same way regardless of changes.

### 2. Early Debugging Steps
- **Gravity Tests:** Set gravity to zero. Despite this, the same collapse animation persisted.
- **Initial Pose:** Changed initial position to (0, 0, 1). No improvement; robot still behaved the same.
- **Ground Plane Verification:** Ensured ground normal was correctly set to (0, 0, 1) — i.e., Z-up orientation. The plane was at Z = 0 as expected.

### 3. URDF Validation
- Used the URDF Visualizer by MorningFrog in VSCode to verify that:
  - All joints were fixed or correctly constrained.
  - Model looked structurally sound.
  - No floating links or broken connections were visible in the static render.
- Confirmed that the URDF file was not corrupted or malformed.

### 4. Key Breakthrough
- Identified the issue was caused by the setting:  
  `asset_options.flip_visual_attachments = True`
- Once this line was changed to `False`, the robot loaded and behaved as expected in Isaac Gym.
- With this fix:
  - The robot no longer collapsed.
  - Gravity applied normally.
  - Structure stayed stable in simulation.

### 5. Final Working Parameters
- `asset_options.fix_base_link = False` (free to move)  
- `asset_options.flip_visual_attachments = False` (prevent incorrect visual-joint flips)  
- Initial robot pose set at Z = 1 (above the ground)  
- Ground plane created at Z = 0 with normal (0, 0, 1)  
- Gravity = (0, 0, -9.81)

---

## Outcome
- Successfully debugged and loaded the robot URDF in Isaac Gym.
- Root cause was isolated to `flip_visual_attachments` flag.
- Simulation now runs with expected physics and visual fidelity.

---

## Next Steps
- Implement joint control logic to manipulate the robot.
- Add sensor data extraction and response behavior.
- Build test scenarios for movement and interaction.
