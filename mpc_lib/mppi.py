import torch
from torch.distributions import Normal

class MPPI(object):

    def __init__(self, model, samples=40, T=10, lam=0.1, eps=0.2):

        self.model          = model
        self.num_actions    = model.num_actions
        self.t_H            = T
        self.lam            = lam
        self.samples        = samples

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'

        self.a = torch.zeros(T, self.num_actions).to(self.device)
        self.eps = Normal(torch.zeros(self.samples, self.num_actions).to(self.device),
                            (torch.ones(self.samples, self.num_actions) * eps).to(self.device))

    def reset(self):
        self.a.zero_()

    def update(self, state):

        with torch.no_grad():
            self.a[:-1] = self.a[1:].clone()
            self.a[-1].zero_()

            s0 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            s = s0.repeat(self.samples, 1)

            sk, da, log_prob = [], [], []
            for t in range(self.t_H):
                eps = self.eps.sample()
                v = self.a[t].expand_as(eps) + eps
                s, rew = self.model.step(s, v)
                log_prob.append(self.eps.log_prob(eps).sum(1))
                da.append(eps)
                sk.append(rew.squeeze())

            sk = torch.stack(sk)
            sk = torch.cumsum(sk.flip(0), 0).flip(0)
            log_prob = torch.stack(log_prob)

            sk = sk.div(self.lam) + self.lam*log_prob
            sk = sk - torch.max(sk, dim=1, keepdim=True)[0]
            w = torch.exp(sk) + 1e-5
            w.div_(torch.sum(w, dim=1, keepdim=True))
            for t in range(self.t_H):
                self.a[t] = self.a[t] + torch.mv(da[t].T, w[t])

            return self.a[0].cpu().clone().numpy()
