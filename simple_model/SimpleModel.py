import pyro
import pyro.distributions as dist
import torch
from torch import nn

from simple_model.Policy import Policy, SNetwork

cols = ['t', 'X_t', 'A_t']


class SimpleModel(nn.Module):
    def __init__(self, use_cuda=False, increment_factor=None):
        super().__init__()
        self.policy = Policy(2, 2)
        self.s_network = SNetwork(4, 2)
        if increment_factor is None:
            increment_factor = (10, 10)
        self.increment_factor = increment_factor
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, mini_batch):
        pyro.module("simple_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s_0_1 = pyro.sample("s_0_1", dist.Normal(70, 20))
            s_0_2 = pyro.sample("s_0_2", dist.Normal(110, 40))
            x_0_1 = pyro.sample("x_0", dist.Normal(s_0_1, 1), obs=mini_batch[:, 0, cols.index('X_t')])
            action = pyro.sample("a_0", dist.Categorical(logits=self.policy(torch.column_stack((s_0_1, s_0_2)))), obs=mini_batch[:, 0, cols.index('A_t')])
            s_1_1 = pyro.sample("s_1_1", dist.Normal(s_0_1 + action*self.increment_factor[0], 10))
            s_1_2 = pyro.sample("s_1_2", dist.Normal(s_0_2 + action*self.increment_factor[1], 15))
            x_1_1 = pyro.sample("x_1", dist.Normal(s_1_1, 1), obs=mini_batch[:, 1, cols.index('X_t')])

    def guide(self, mini_batch):
        pyro.module("simple_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s_0_1 = pyro.sample("s_0_1", dist.Normal(70, 20))
            s_0_2 = pyro.sample("s_0_2", dist.Normal(110, 40))
            s_1_loc, s_1_scale = self.s_network(torch.column_stack((s_0_1, s_0_2, mini_batch[:, 0, cols.index('A_t')], mini_batch[:, 1, cols.index('X_t')])))
            s_1_1 = pyro.sample("s_1_1", dist.Normal(s_1_loc[:, 0], torch.exp(s_1_scale[:, 0])))
            s_1_2 = pyro.sample("s_1_2", dist.Normal(s_1_loc[:, 1], torch.exp(s_1_scale[:, 1])))

