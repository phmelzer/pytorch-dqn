""" Create DeepQNetwork Class """

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()
        self.name = name
        self.checkpoint_file = os.path.join("../models/", name)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

    def save_checkpoint(self, training_time):
        T.save(self.state_dict(), self.checkpoint_file+"_"+training_time)

    def load_checkpoint(self, training_time):
        self.load_state_dict(T.load(self.checkpoint_file+"_"+training_time))
