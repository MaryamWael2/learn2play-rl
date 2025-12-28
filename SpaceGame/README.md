# 🚀 SpaceGame – Deep Reinforcement Learning Arcade Game

SpaceGame is a 2D arcade-style space game powered by **Deep Reinforcement Learning**.
An AI agent is trained using a **Convolutional Neural Network (CNN)** to navigate the environment, avoid obstacles, and maximize score.

How it works?
* The game environment provides **visual frames**
* Frames are processed by a **CNN**
* The agent uses **Q-Learning / Deep Q-Learning**
* Actions are chosen based on predicted Q-values
* Rewards guide the agent to improve over time

---

## 📂 Project Structure

```
SpaceGame/
|
├── logs/                     # Training & testing logs
│   ├── training.log
│   └── testing.log
│
├── model/
│   └── CNNmodel.pth          # Saved trained CNN agent
│
├── plots/                    # Score curves
│   ├── training_plot.png
│   └── testing_plot.png
│
├── src/
│   ├── agent/
│   │   ├── cnn_agent.py      # RL agent logic
│   │   ├── cnn_model.py      # CNN architecture
│   │   ├── qtrainer.py       # Training loop & optimizer
│   │   └── __init__.py
│   │ 
│   ├── env/
│   │   ├── bullet.py  
│   │   ├── space_game_ai.py     # RL environment (AI-controlled)
│   │   ├── space_game_human.py  # Human-playable version
│   │   ├── ufo_bullet.py 
│   │   ├── ufo.py 
│   │   └── assets/           # Game sprites
│   │       ├── bg.png
│   │       ├── bullet.png
│   │       ├── rocket.png
│   │       ├── stone.png
│   │       └── ufo.jpg
│   │ 
│   ├── scripts/
│   │   ├── play_human.py        # Play manually (keyboard)
│   │   ├── train_cnn_agent.py   # Train the agent
│   │   ├── test_cnn_agent.py    # Run the trained agent
│   │   └── utils.py 
│   │
├── requirements.txt
│
└── README.md
```
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SpaceGame.git
cd SpaceGame
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---
## 🕹️ Play the Game (Human Mode)

```bash
python -m src.scripts.play_human
```

### Controls

* **⬅️ / ➡️ Arrow keys** — Steer left / right
* **SPACE** — Shoot bullet
* **ESC** — Quit

---

## 🤖 Run the Trained AI Agent

```bash
python -m src.scripts.test_cnn_agent
```

This loads the pretrained `CNNmodel.pth` model and runs inference.

Outputs include:

* Logs saved to: `logs/testing.log`
* Reward curve: `plots/testing_plot.png`

---

## 🏋️ Train Your Own RL Agent

```bash
python -m src.scripts.train_cnn_agent
```

Training outputs:

* Logs saved to: `logs/training.log`
* Model checkpoints in: `model/`
* Reward curve: `plots/training_plot.png`

---

## 💡 RL Environment Overview

### **State Space**

The agent receives the **last 4 processed grayscale frames** (stacked), excluding the static background.
Processing is implemented in:

* `space_game_ai.py`
* `cnn_agent.py`

### **Action Space**

Discrete actions:

1. **Move Left**
2. **Move Right**
3. **Shoot Bullet**
4. **Do Nothing**

### **Reward Function**

(from `space_game_ai.py`)

* **+0.1** reward per time step
* **+1** reward for successfully killing a UFO
* **–10** penalty for UFO bullet collision
* **–10** penalty for UFO reaching the end of the screen

---

## 📜 Contributing / Next Steps

Ideas to extend this project:
* Change CNN architecture (`cnn_model.py`)
* Tune reward function
* Add new obstacles or enemies with different strengths 
* Increase observation resolution
* Replace DQN with PPO or A3C

Feel free to use, modify, and distribute.
