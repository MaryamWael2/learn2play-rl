import torch
import torch.nn as nn
import torch.optim as optim

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.bool).to(device)

        # Add batch dimension if needed
        if len(state.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # Predicted Q values
        pred = self.model(state)

        # Predicted Q values for next states
        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_pred, _ = torch.max(next_pred, dim=1)

        # Compute target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx] + self.gamma * max_next_pred[idx] * (1 - done[idx].float())
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
