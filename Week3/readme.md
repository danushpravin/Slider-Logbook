# Week 3 Logbook Update – Slider Robot (Reinforcement Learning)

## Overview

This week focused on addressing issues with my RL agent for the slider robot. Previously, the agent was not learning, and after investigation, I found bugs in the environment code that were affecting training stability and reward feedback.

## Key Updates

- ❌ **Old environment scrapped**: The initial version of the environment was flawed and unproductive for learning.
- 🔧 **Rebuilt a simplified environment**:
  - Clean and minimal structure.
  - Verified correct state observations and reward calculations.
  - Ensured stable action application and episode flow.
- ✅ **Agent now learns as expected**:
  - Training shows positive learning trends.
  - Rewards are increasing over time.
- 📈 **Plotted rewards over time**:
  - Graph confirms the learning behavior.
  - Shows steady improvement across episodes.

## Reward Plot

*A sample plot of reward progression over time is included below (add your actual plot here if embedding).*

![Reward Plot](./Slider-Logbook/Week3/Screenshot from 2025-06-04 22-32-41)

---

This version of the environment lays a strong foundation for building more complex behaviors and control logic in the coming weeks.

