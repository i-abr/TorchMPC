import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class iLQR(object):

    def __init__(self, model , T=10, lr=0.1):

        self.T = T
        self.model = model
        self.state_dim = model.num_states
        self.action_dim = model.num_actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        self.lr     = lr
        self.K      = torch.randn(T, self.action_dim, self.state_dim).to(self.device)
        self.k      = torch.randn(T, self.action_dim).to(self.device)
        self.xbar   = torch.randn(T, self.state_dim).to(self.device)

        self.K.requires_grad    = True
        self.xbar.requires_grad = True
        self.k.requires_grad    = True
        self.optim = optim.Adam([self.K, self.k, self.xbar], self.lr)


    def reset(self):
        with torch.no_grad():
            self.K.zero_()
            self.k.zero_()
            self.xbar.zero_()
    def update(self, state, epochs=2):

        for epoch in range(epochs):
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            cost = 0.
            for K, k, xbar in zip(self.K, self.k, self.xbar):
                u = torch.mv(K, xbar - s.squeeze()) + k
                s, r = self.model.step(s, u.unsqueeze(0))
                cost = cost - r
            self.optim.zero_grad()
            cost.backward()
            self.optim.step()
        with torch.no_grad():
            K = self.K[0].cpu().clone().numpy()
            k = self.k[0].cpu().clone().numpy()
            xbar = self.xbar[0].cpu().clone().numpy()
            self.k[:-1] = self.k[1:].clone()
            self.k[-1].zero_()
            self.K[:-1] = self.K[1:].clone()
            self.K[-1].zero_()
            self.xbar[:-1] = self.xbar[1:].clone()
            self.xbar[-1].zero_()
            return np.dot(K, xbar - state) + k
