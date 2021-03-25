import pyro
import pyro.distributions as dist
import torch
from torch import nn

from simple_model.Policy import Policy, SNetwork

cols = ['t', 'X_t', 'A_t']


class SimpleModel(nn.Module):
    def __init__(self, use_cuda=False, increment_factor=None, policy=None):
        super().__init__()
        if policy is None:
            self.policy = Policy(2, 2)
        else:
            self.policy = policy
        self.st_network = SNetwork(4, 2)
        if increment_factor is None:
            increment_factor = (20, 20)
        self.increment_factor = torch.tensor(increment_factor)
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, mini_batch):
        pyro.module("simple_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            mu, sigma = torch.tensor((70, 110)).float(), torch.tensor([[20, 15], [15, 40]]).float()
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
            mu, sigma = torch.tensor((70, 110)).float(), torch.tensor([[20, 15], [15, 40]]).float()
            s_0 = pyro.sample("s_0", dist.MultivariateNormal(mu, covariance_matrix=sigma))
            st_loc, st_tril, st_diag = self.st_network(torch.column_stack((
                s_0[:, 0], s_0[:, 1], mini_batch[:, 0, cols.index('A_t')], mini_batch[:, 1, cols.index('X_t')])))
            z = torch.zeros(st_loc.size(0))
            scale_tril = torch.stack([
                st_diag[:, 0], z,
                st_tril[:, 0], st_diag[:, 1]
            ], dim=-1).view(-1, 2, 2)
            s1 = pyro.sample("s_1", dist.MultivariateNormal(loc=st_loc, scale_tril=scale_tril))

