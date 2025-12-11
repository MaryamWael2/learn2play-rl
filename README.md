```markdown
# 🚗 CarGame – Reinforcement Learning Self-Driving Car Environment

This project implements a **2D self-driving car environment** along with:
- A Convolutional Neural Network (CNN) reinforcement learning agent
- Training and testing pipelines
- A human-playable version of the game
- Logging + plots of reward curves
- Pretrained model weights (`CNNmodel.pth`)

The environment is written using **Pygame**, and the RL agent is implemented using **PyTorch**.

---

## 📁 Project Structure

```

CarGame/
│
├── logs/                     # Training & testing logs
│   ├── training.log
│   └── testing.log
│
├── model/
│   └── CNNmodel.pth          # Saved pretrained CNN agent
│
├── plots/                    # Training / testing reward curves
│   ├── training_plot.png
│   └── testing_plot.png
│
├── src/
│   ├── agents/
│   │   ├── cnn_agent.py      # Agent training & action selection logic
│   │   ├── cnn_model.py      # PyTorch CNN architecture
│   │   └── qtrainer.py       # Q learning algorithm
│   │
│   ├── env/
│   │   ├── obstacle.py  
│   │   ├── car_env_ai.py     # RL environment (agent-controlled)
│   │   ├── car_env_human.py  # Human-playable version
│   │   └── assets/           # Game sprites (car, road, obstacles)
│   │       ├── car.png
│   │       ├── obstacle.png
│   │       └── road.jpg
│   │
│   ├── scripts/
│       ├── play_human.py        # Play manually with keyboard
│       ├── train_cnn_agent.py     # train AI agent
│       ├── test_cnn_agent.py   # Run trained AI agent
│       └── utils.py 
│
├── requirements.txt 
│
└── README.md                 # (this file)

````

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MaryamWael2/learn2play-rl.git
cd learn2play-rl/CARGAME/CarGame
````

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

* **⬅️ / ➡️ arrows** – Steer left and right
* **ESC** – Quit

---

## 🤖 Run the Trained AI Agent

```bash
python -m src.scripts.test_cnn_agent
```

This loads the pretrained `CNNmodel.pth` and runs inference inside `test_cnn_agent.py`.
Training outputs include:
* Logs saved to `logs/testing.log`
* Score curves in `plots/testing.png`

---

## 🏋️ Train Your Own RL Agent

```bash
python -m src.scripts.train_cnn_agent
```

Training outputs include:
* Logs saved to `logs/training.log`
* Model checkpoints in `model/`
* Score curves in `plots/training.png` 

---

## 💡 Reinforcement Learning Overview

### **State**

The agent receives last 4 grayscale frames of the screen excluding the background (processed in `car_env_ai.py` and `cnn_agent.py`).

### **Actions**

Discrete action space:

1. **Turn Left**
2. **Turn Right**
5. **Do Nothing**

### **Reward Function**

Defined in `car_env_ai.py`, includes components such as:

* +0.1 reward per step
* +1 reward when a car passes
* -10 penalty for collisions

### **Algorithm**

* CNN-based Deep Q-Learning (DQN)
* Replay buffer
* ε-greedy exploration

---

## 🤝 Contributing

Pull requests and issues are welcome.

```
