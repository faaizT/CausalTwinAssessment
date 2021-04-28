import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
import pyro.contrib.examples.polyphonic_data_loader as poly
from pulse.utils.MIMICDataset import dummy_cols as cols, action_cols
from simple_model.Policy import Policy
from pulse.utils.PulseInterface import *
from max_likelihood.utils.HelperNetworks import Combiner


class PulseModel(nn.Module):
    def __init__(self, rnn_dim=40, rnn_dropout_rate=0.0, st_vec_dim=1, n_act=2, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.s0_hr_mean = nn.Parameter(torch.zeros(1))
        self.s0_hr_var = nn.Parameter(torch.ones(1))
        self.policy = Policy(
            input_dim=st_vec_dim,
            hidden_1_dim=20,
            hidden_2_dim=20,
            output_dim=1,
            use_cuda=use_cuda,
        )
        self.rnn = nn.RNN(
            input_size=len(cols),
            hidden_size=rnn_dim,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=False,
            num_layers=1,
            dropout=rnn_dropout_rate,
        )
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.combiner = Combiner(z_dim=st_vec_dim, rnn_dim=rnn_dim, out_dim=st_vec_dim)
        self.s_q_0 = nn.Parameter(torch.zeros(st_vec_dim))
        if use_cuda:
            self.cuda()
    
    def emission(self, st_hr, t, mini_batch, mini_batch_mask):
        xt_hr = pyro.sample(f"x{t}_hr", dist.Normal(st_hr, 0.001).mask(mini_batch_mask[:,t]), obs=mini_batch[:,t,cols.index("HR")])

    def create_pool(self, st):
        pool = PulsePhysiologyEnginePool()
        for i in range(st.size(0)):
            p1 = pool.create_patient(i)
            p1.hr = st[i].detach().item()
        return pool

    def transition(self, pool, at):
        for p in pool.get_patients():
            p.actions.append(at[p.id].detach().item())
        pool.advance_time_s(60)

    def pull_data(self, pool):
        data = []
        for p in pool.get_patients():
            data.append(p.results())
        return torch.FloatTensor(data)
    
    def model(self, mini_batch, actions_obs, mini_batch_mask, mini_batch_seq_lengths, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("pulse", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            st_hr = pyro.sample(f"s0_hr", dist.Normal(self.s0_hr_mean, self.s0_hr_var))
            self.emission(st_hr, 0, mini_batch, mini_batch_mask)
            pool = self.create_pool(st_hr)
            for t in range(T_max - 1):
                action = self.policy(st_hr.unsqueeze(1)).squeeze()
                at = pyro.sample(f"a{t}", 
                    dist.Normal(action, 0.001).mask(mini_batch_mask[:,t+1]), 
                    obs=actions_obs[:,t,action_cols.index("median_dose_vaso")]
                )
                self.transition(pool, at)
                st_hr = pyro.sample(f"s{t+1}_hr", dist.Normal(self.pull_data(pool), 0.01).mask(mini_batch_mask[:, t+1]))
                self.emission(st_hr, t+1, mini_batch, mini_batch_mask)
            
    def guide(self, mini_batch, actions_obs, mini_batch_mask, mini_batch_seq_lengths, mini_batch_reversed):
        T_max = mini_batch.size(1)
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        st_prev = self.s_q_0.expand(mini_batch.size(0), self.s_q_0.size(0))
        pyro.module("pulse", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            for t in range(T_max):
                st_loc, st_scale = self.combiner(st_prev, rnn_output[:, t, :])
                st_hr = pyro.sample(f"s{t}_hr", dist.Normal(st_loc.squeeze(), torch.exp(st_scale).squeeze()+0.001).mask(mini_batch_mask[:,t]))
                st_prev = st_hr.unsqueeze(1)