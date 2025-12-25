# 🚗 CarGame – Reinforcement Learning Self-Driving Car Environment

This project implements a **2D self-driving car environment** along with:

* A Convolutional Neural Network (CNN) reinforcement learning agent
* Training and testing pipelines
* A human-playable version of the game
* Logging + reward-curve plots
* Pretrained model weights (`CNNmodel.pth`)

The environment is built using **Pygame**, and the RL agent is implemented with **PyTorch**.

---

## 📁 Project Structure

```text
CarGame/
│
├── logs/                     # Training & testing logs
│   ├── training.log
│   └── testing.log
│
├── model/
│   └── CNNmodel.pth          # Saved pretrained CNN agent
│
├── plots/                    # Score curves
│   ├── training_plot.png
│   └── testing_plot.png
│
├── src/
│   ├── agents/
│   │   ├── cnn_agent.py      # Agent logic + training loop
│   │   ├── cnn_model.py      # PyTorch CNN architecture
│   │   └── qtrainer.py       # Q-learning implementation
│   │
│   ├── env/
│   │   ├── obstacle.py  
│   │   ├── car_env_ai.py     # RL environment (AI-controlled)
│   │   ├── car_env_human.py  # Human-playable version
│   │   └── assets/           # Game sprites
│   │       ├── car.png
│   │       ├── obstacle.png
│   │       └── road.jpg
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

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MaryamWael2/learn2play-rl.git
cd learn2play-rl/CARGAME/CarGame
```

### 2. Install Dependencies

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

* `car_env_ai.py`
* `cnn_agent.py`

### **Action Space**

Discrete actions:

1. **Turn Left**
2. **Turn Right**
3. **Do Nothing**

### **Reward Function**

(from `car_env_ai.py`)

* **+0.1** reward per time step
* **+1** reward for successfully passing an obstacle
* **–10** penalty for collisions

---

## 🧠 Algorithm

The project uses **Deep Q-Learning (DQN)** with:

* CNN-based feature extractor
* Replay memory
* ε-greedy exploration
* Target network updates (if enabled)

---

## 🤝 Contributing

Pull requests, feature suggestions, and issues are welcome!
