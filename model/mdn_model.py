import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MDNDyn(nn.Module):
    def __init__(self, num_inputs, num_outputs,
                    n_hidden=200, n_gaussians=10):
        super(MDNDyn, self).__init__()

        self.linear1 = nn.Linear(num_inputs, n_hidden)

        # self.z_h = nn.Sequential(
        #     nn.Linear(num_inputs, n_hidden),
        #     nn.ReLU()
        # )

        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians * num_outputs)
        self.z_mu = nn.Linear(n_hidden, n_gaussians * num_outputs)

        self.num_outputs = num_outputs
        self.num_gauss   = n_gaussians
        self.num_inputs  = num_inputs

    def sample(self, x, u):
        pi, sigma, mu = self.forward(x, u)
        pi_picked = torch.multinomial(pi, 1)
        # pi_picked = torch.argmax(pi, dim=1, keepdim=True)
        res = []
        for i, r in enumerate(pi_picked):
            res.append(
                mu[i, r]#torch.normal(mu[i, r], sigma[i, r])
            )

        return torch.cat(res)

    def logits(self, x, u, y):
        pi, sigma, mu = self.forward(x, u)
        y_expand = y.unsqueeze(1).expand_as(mu)

        log_pi = torch.log(pi)
        log_pdf = -torch.log(sigma).sum(2)-0.5*((y_expand - mu) * torch.reciprocal(sigma)).pow(2).sum(2)
        return torch.logsumexp(log_pdf + log_pi, dim=1, keepdim=True)

    def forward(self, x, u):
        # z_h = self.z_h(torch.cat([x, u], dim=1))
        z_h = torch.sin(self.linear1(torch.cat([x, u], dim=1)))
        pi = nn.functional.softmax(self.z_pi(z_h) + 1e-3, -1)
        sigma = torch.clamp(self.z_sigma(z_h), -20, 4).exp()

        mu = self.z_mu(z_h).view(-1, self.num_gauss, self.num_outputs)
        mu = mu + x.unsqueeze(1).expand_as(mu)
        return pi, sigma.view(-1, self.num_gauss, self.num_outputs), mu


class MDNModel(nn.Module):

    def __init__(self, num_states, num_actions, def_layers=[200, 200],
                                                n_gaussians=10):

        super(MDNModel, self).__init__()


        self.num_states  = num_states
        self.num_actions = num_actions

        self.mdn_model = MDNDyn(num_states+num_actions, num_states,
                                    n_hidden=def_layers[0], n_gaussians=n_gaussians)
        self.n_params = []

        layers = [num_states] + def_layers + [1]
        for i, (insize, outsize) in enumerate(zip(layers[:-1], layers[1:])):
            var = 'rew_layer' + str(i)
            setattr(self, var, nn.Linear(insize, outsize))
            self.n_params.append(i)

    def predict_reward(self, s):
        #rew = torch.cat([s, a], axis=1)
        rew = s
        for i in self.n_params[:-1]:
            w = getattr(self, 'rew_layer' + str(i))
            rew = w(rew)
            rew = F.relu(rew)
            # rew = torch.sin(rew)

        w = getattr(self, 'rew_layer' + str(self.n_params[-1]))
        rew = w(rew)
        return rew

    def forward(self, s, a, ns):
        """
        dx, rew = forward(s, a)
        dx is the change in the state
        """

        rew = self.predict_reward(s)
        logprob = self.mdn_model.logits(s, a, ns)

        return logprob, rew

    def step(self, x, u):
        ns = self.mdn_model.sample(x, u)
        rew = self.predict_reward(x)
        return ns, rew
