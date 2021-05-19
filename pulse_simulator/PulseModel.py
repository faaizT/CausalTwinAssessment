import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
from pulse_simulator.utils.Networks import Net
import pyro.contrib.examples.polyphonic_data_loader as poly
from pulse_simulator.utils.MIMICDataset import (
    cols,
    pulse_cols,
    action_cols,
    static_cols
)
from pulse_simulator.utils.Networks import Policy
from max_likelihood.utils.HelperNetworks import Combiner
import logging


class PulseModel(nn.Module):
    def __init__(
        self, rnn_dim=40, rnn_dropout_rate=0.0, st_vec_dim=len(pulse_cols), n_act=25, use_cuda=False
    ):
        super().__init__()
        self.tanh = nn.Tanh()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.st_vec_dim = st_vec_dim
        self.actions = Policy(
            input_dim=st_vec_dim + len(static_cols),
            output_dim=n_act,
            hidden_1_dim=8,
            use_cuda=use_cuda,
        )
        if use_cuda:
            self.cuda()

    def emission(self, sim_mini_batch, mini_batch, t):
        for column in cols:
            if column not in static_cols:
                pyro.sample(
                    f"x{t}_{column}",
                    dist.Normal(sim_mini_batch[:, 1, pulse_cols.index(column)], 0.01),
                    obs=mini_batch[:, t, cols.index(column)],
                )

    def model(
        self,
        mini_batch,
        sim_mini_batch,
        actions_obs,
        static_data
    ):
        T_max = mini_batch.size(1)
        pyro.module("pulse", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            inp = torch.column_stack((sim_mini_batch[:,0,:], static_data.squeeze()))
            for t in range(T_max - 1):
                at = pyro.sample(f"a{t}",
                    dist.Categorical(logits=self.actions(inp).squeeze()),
                    obs=actions_obs[:,t,action_cols.index("A_t")]
                )
                self.emission(sim_mini_batch, mini_batch, t+1)

    def guide(
        self,
        mini_batch,
        sim_mini_batch,
        actions_obs,
        static_data
    ):
        pass