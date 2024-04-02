import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        # Build the input, hidden and output layer
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Feed forward NN
    def forward(self, x): # x is our tensor 
        x = F.relu(self.linear1(x)) # Use ReLU activation
        x = self.linear2(x) # No activation needed as we take maximum value at end
        return x

    # Saving the model on the computer
    def save(self, file_name='model.pth'): 
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name) # Save NN

# Training and Optimisation
class QTrainer:
    def __init__(self, model, lr, gamma): # Initializing 
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # Classic Optimizer in NNs
        self.criterion = nn.MSELoss() # Loss function is MSE for Q Nets

    # The trainer, which is used in the agent. It must work for both 1 dimension for short term, and for a tensor as in the long term
    def train_step(self, state, action, reward, next_state, done): 
        # Turn imputs into tensors of the form n * x
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1: # 1 dimension for short term. Need to change to the form 1 * x so it can be used for the next part
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,) # Convert to tuple

        # Start implementing Bellman Equation
            
        pred = self.model(state) # Find predicted Q value with current state

        # Qnew = r + gamma*(next predicted Q)
        # Clone predicted values
        # Then do Q_new = argmax[(action)]

        target = pred.clone() 
        for idx in range(len(done)): # Everything has the same size so done is arbitrary
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Calculating loss function
        self.optimizer.zero_grad() 
        loss = self.criterion(target, pred) # Q_new, Q
        loss.backward() # Backprop to update weights
        self.optimizer.step()
