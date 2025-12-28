# 🎮 Learn2Play-RL — Deep Reinforcement Learning Game Environments

This repository contains two **2D arcade-style game environments** built with **Pygame**, each paired with a **CNN-based Deep Reinforcement Learning (DQN/Q-Learning)** agent implemented in **PyTorch**.

Each game includes:
- A human-playable mode
- Training + testing pipelines for the AI agent
- Logs and reward-curve plots
- Pretrained model weights (`CNNmodel.pth`)

## 📦 Projects in this Repo

### 🚗 CarGame — Self-Driving Car (Obstacle Avoidance)
A 2D driving environment where an AI agent learns to steer and avoid obstacles using frame-based observations. :contentReference[oaicite:0]{index=0}

**Key features**
- Discrete actions: left / right / do nothing
- Rewards for surviving and passing obstacles; penalties for collisions
- Human-playable mode (keyboard)

📍 Location: `CARGAME/CarGame/` :contentReference[oaicite:1]{index=1}

---

### 🚀 SpaceGame — Arcade Space Shooter (Avoid + Shoot)
A 2D space arcade game where an AI agent learns to move, avoid threats, and shoot using CNN-processed visual frames. :contentReference[oaicite:2]{index=2}

**Key features**
- Discrete actions: move left / move right / shoot / do nothing
- Rewards for eliminating targets; penalties for collisions and failures
- Human-playable mode (keyboard)

📍 Location: `SpaceGame/` :contentReference[oaicite:3]{index=3}

---

## 🧠 How the Agents Learn (High Level)

Both projects follow the same general pattern:

* The environment provides **visual frames**
* Frames are preprocessed and stacked (commonly the last 4 frames)
* A CNN estimates Q-values for each action
* The agent trains via replay memory + ε-greedy exploration

For full details, see each project’s README:

* `CARGAME/CarGame/README.md` 
* `SpaceGame/README.md` 

---

## 🤝 Contributing

Issues and pull requests are welcome. If you’re adding a new game:

* Please follow the existing structure (logs/model/plots/src/scripts)
* Please provide a dedicated `requirements.txt`
* Please include a project-level README with run/train/test instructions

```