import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints

from simple_model.Policy import Policy, SNetwork, physicians_policy

cols = ['t', 'X_t', 'A_t']


class SimpleModel(nn.Module):
    def __init__(self, learn_initial_state, phy_pol, increment_factor, rho, use_cuda=False):
        super().__init__()
        if phy_pol:
            self.policy = physicians_policy
        else:
            self.policy = Policy(2, 2)
        self.s1_network = SNetwork(4, 2)
        self.s0_network = SNetwork(3, 2)
        if increment_factor is None:
            increment_factor = (20, 20)
        self.increment_factor = torch.tensor(increment_factor)
        self.learn_initial_state = learn_initial_state
        if rho is None:
            self.rho = 15 / np.sqrt(20 * 40)
        else:
            self.rho = rho
        self.use_cuda = use_cuda
        self.mu = nn.Parameter(torch.rand(2).float())
        self.log_diag0 = nn.Parameter(torch.rand(1).float())
        self.log_diag1 = nn.Parameter(torch.rand(1).float())
        self.tril = nn.Parameter(torch.rand(1))
        if use_cuda:
            self.cuda()

    def model(self, mini_batch):
        pyro.module("simple_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            if self.learn_initial_state:
                mu = self.mu
                scale_tril = torch.stack([
                    torch.exp(self.log_diag0), torch.tensor([0]),
                    self.tril, torch.exp(self.log_diag1)
                ], dim=-1).view(-1, 2, 2)
                s_0 = pyro.sample("s_0", dist.MultivariateNormal(mu, scale_tril=scale_tril))
            else:
                mu = torch.tensor((70, 110)).float()
                sigma = torch.tensor([[20, self.rho*np.sqrt(40*20)], [self.rho*np.sqrt(40*20), 40]]).float()
                s_0 = pyro.sample("s_0", dist.MultivariateNormal(mu, covariance_matrix=sigma))
            x_0 = pyro.sample("x_0", dist.Normal(s_0[:, 0], 1), obs=mini_batch[:, 0, cols.index('X_t')])
            a_0 = pyro.sample("a_0", dist.Categorical(logits=self.policy(s_0)), obs=mini_batch[:, 0, cols.index('A_t')])
            increment_factor = torch.stack(len(mini_batch)*[self.increment_factor])
            s_1 = pyro.sample("s_1", dist.MultivariateNormal(s_0 + torch.column_stack(2*[a_0])*increment_factor,
                                                             torch.tensor([[1, 0.5], [0.5, 1]]).float()))
            x_1 = pyro.sample("x_1", dist.Normal(s_1[:, 0], 1), obs=mini_batch[:, 1, cols.index('X_t')])
            a_1 = pyro.sample("a_1", dist.Categorical(logits=self.policy(s_1)), obs=mini_batch[:, 1, cols.index('A_t')])

    def guide(self, mini_batch):
        pyro.module("simple_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s0_loc, s0_tril, s0_diag = self.s0_network(torch.column_stack((
                mini_batch[:, 0, cols.index('X_t')], mini_batch[:, 1, cols.index('X_t')],
                mini_batch[:, 0, cols.index('A_t')])))
            scale_tril_s0 = torch.stack([
                s0_diag[:, 0], torch.zeros(s0_loc.size(0)),
                s0_tril[:, 0], s0_diag[:, 1]
            ], dim=-1).view(-1, 2, 2)
            s_0 = pyro.sample("s_0", dist.MultivariateNormal(loc=s0_loc, scale_tril=scale_tril_s0))
            s1_loc, s1_tril, s1_diag = self.s1_network(torch.column_stack((
                s_0[:, 0], s_0[:, 1], mini_batch[:, 0, cols.index('A_t')], mini_batch[:, 1, cols.index('X_t')])))
            scale_tril_s1 = torch.stack([
                s1_diag[:, 0], torch.zeros(s1_loc.size(0)),
                s1_tril[:, 0], s1_diag[:, 1]
            ], dim=-1).view(-1, 2, 2)
            s1 = pyro.sample("s_1", dist.MultivariateNormal(loc=s1_loc, scale_tril=scale_tril_s1))

