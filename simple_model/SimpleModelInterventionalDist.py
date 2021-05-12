import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
from simple_model.SimpleModel import SimpleModel
from simple_model.Policy import Policy, SNetwork, physicians_policy

cols = ['t', 'X_t', 'A_t', 'id']


class SimpleModelInterventionalDist(nn.Module):
    def __init__(self, learned_initial_state, phy_pol, increment_factor, rho, use_cuda=False):
        super().__init__()
        if increment_factor is None:
            increment_factor = (20, 20)
        x, y  = increment_factor
        self.increment_factor = torch.tensor(increment_factor)
        model = SimpleModel(learn_initial_state=True, increment_factor=increment_factor, phy_pol=phy_pol, rho=rho)
        model.load_state_dict(torch.load(learned_initial_state + f"/model-state-{x}-{y}-final"))
        self.s1_network = model.s1_network
        self.s0_network = model.s0_network
        if rho is None:
            self.rho = 15 / np.sqrt(20 * 40)
        else:
            self.rho = rho
        self.use_cuda = use_cuda
        self.mu = model.mu
        self.log_diag0 = model.log_diag0
        self.log_diag1 = model.log_diag1
        self.tril = model.tril
        if use_cuda:
            self.cuda()

    def model(self, mini_batch):
        pyro.module("simple_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            mu = self.mu
            scale_tril = torch.stack([
                torch.exp(self.log_diag0), torch.tensor([0]),
                self.tril, torch.exp(self.log_diag1)
            ], dim=-1).view(-1, 2, 2)
            s_0 = pyro.sample("s_0", dist.MultivariateNormal(mu, scale_tril=scale_tril))
            x_0 = pyro.sample("x_0", dist.Normal(s_0[:, 0], 1), obs=mini_batch[:, 0, cols.index('X_t')])
            a_0 = mini_batch[:, 0, cols.index('A_t')]
            increment_factor = torch.stack(len(mini_batch)*[self.increment_factor])
            s_1 = pyro.sample("s_1", dist.MultivariateNormal(s_0 + torch.column_stack(2*[a_0])*increment_factor,
                                                             torch.tensor([[1, 0.5], [0.5, 1]]).float()))
            x_1 = pyro.sample("x_1", dist.Normal(s_1[:, 0], 1), obs=mini_batch[:, 1, cols.index('X_t')])

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

