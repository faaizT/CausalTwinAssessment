import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
import pyro.contrib.examples.polyphonic_data_loader as poly
from gumbel_max_sim.utils.State import State
from gumbel_max_sim.utils.Action import Action
from gumbel_max_sim.utils.MDP import MdpPyro
from gumbel_max_sim.utils.ObservationalDataset import cols
from .utils.Distributions import CategoricalVals
from simple_model.Policy import Policy
from gumbel_max_sim.utils.Networks import Combiner, Combiner_without_rnn, Net


class GumbelMaxModel(nn.Module):
    def __init__(
        self,
        use_cuda=False,
        st_vec_dim=8,
        n_act=8,
        rnn_dim=40,
        rnn_dropout_rate=0.0,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy = Policy(
            input_dim=st_vec_dim,
            hidden_1_dim=20,
            hidden_2_dim=20,
            output_dim=n_act,
            use_cuda=use_cuda,
        )
        self.policy.to(self.device)
        self.s0_diab_logits = nn.Parameter(torch.zeros(2))
        self.s0_diab_logits.to(self.device)
        self.s0_hr = Net(input_dim=1, output_dim=3)
        self.s0_sysbp = Net(input_dim=1, output_dim=3)
        self.s0_glucose = Net(input_dim=1, output_dim=5)
        self.s0_percoxyg = Net(input_dim=1, output_dim=2)
        self.rnn = nn.RNN(
            input_size=len(cols),
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=1,
            dropout=rnn_dropout_rate,
        )
        self.combiner = Combiner(z_dim=st_vec_dim, rnn_dim=rnn_dim, use_cuda=use_cuda)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.s0_diab_net = Net(input_dim=rnn_dim, hidden_dim=20, output_dim=2, use_cuda=use_cuda)
        self.s_q_0_hr = nn.Parameter(torch.zeros(3))
        self.s_q_0_sysbp = nn.Parameter(torch.zeros(3))
        self.s_q_0_percoxyg = nn.Parameter(torch.zeros(2))
        self.s_q_0_glucose_no_diab = nn.Parameter(torch.zeros(5))
        self.s_q_0_glucose_diab = nn.Parameter(torch.zeros(5))
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, actions_obs, mini_batch_mask, mini_batch_seq_lengths, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("gumbel_max", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s0_diab = pyro.sample(
                f"s0_diab_state", dist.Categorical(logits=self.s0_diab_logits)
            ).float()
            s0_diab = s0_diab.unsqueeze(dim=1)
            s0_hr = pyro.sample(
                f"s0_hr",
                dist.Categorical(logits=self.s0_hr(s0_diab).unsqueeze(dim=0))
            )
            s0_sysbp = pyro.sample(
                f"s0_sysbp",
                dist.Categorical(logits=self.s0_sysbp(s0_diab).unsqueeze(dim=0))
            )
            s0_glucose = pyro.sample(
                f"s0_glucose",
                dist.Categorical(logits=self.s0_glucose(s0_diab).unsqueeze(dim=0))
            )
            s0_percoxyg = pyro.sample(
                f"s0_percoxyg",
                dist.Categorical(logits=self.s0_percoxyg(s0_diab).unsqueeze(dim=0))
            )
            a_prev = Action(action_idx=torch.zeros(len(mini_batch)))
            state_categ = torch.column_stack(
                (
                    s0_hr,
                    s0_sysbp,
                    s0_percoxyg,
                    s0_glucose,
                    a_prev.antibiotic,
                    a_prev.vasopressors,
                    a_prev.ventilation,
                    s0_diab.reshape(-1),
                )
            )
            mdp = MdpPyro(init_state_categ=state_categ, device=self.device)
            mdp.emission(mini_batch, mini_batch_mask, t=0)
            for t in range(T_max - 1):
                at = pyro.sample(
                    f"a{t}",
                    dist.Categorical(
                        logits=self.policy(mdp.state.get_state_tensor())
                    ).mask(mini_batch_mask[:, t + 1]),
                    obs=actions_obs[:, t].reshape(-1),
                )
                action = Action(action_idx=at)
                mdp.transition(action, mini_batch_mask, t + 1)
                mdp.emission(mini_batch, mini_batch_mask, t + 1)


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
        pyro.module("gumbel_max", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s_q_0_diab = pyro.sample(
                f"s0_diab_state", dist.Categorical(logits=self.s0_diab_net(rnn_output[:, 0, :]))
            )
            s_q_0_hr = pyro.sample("s_q_0_hr", dist.Categorical(logits=self.s_q_0_hr), infer={'is_auxiliary': True})
            s_q_0_sysbp = pyro.sample("s_q_0_sysbp", dist.Categorical(logits=self.s_q_0_sysbp), infer={'is_auxiliary': True})
            s_q_0_percoxyg = pyro.sample("s_q_0_percoxyg", dist.Categorical(logits=self.s_q_0_percoxyg), infer={'is_auxiliary': True})
            s_q_0_glucose_no_diab = pyro.sample("s_q_0_glucose_no_diab", dist.Categorical(logits=self.s_q_0_glucose_no_diab), infer={'is_auxiliary': True})
            s_q_0_glucose_diab = pyro.sample("s_q_0_glucose_diab", dist.Categorical(logits=self.s_q_0_glucose_diab), infer={'is_auxiliary': True})
            s_q_0_glucose = s_q_0_diab * s_q_0_glucose_diab + (1-s_q_0_diab) * s_q_0_glucose_no_diab
            s_q_0_at = torch.zeros(len(mini_batch))
            s_q_0_at.to(self.device)
            s_prev = torch.column_stack((s_q_0_diab, s_q_0_hr, s_q_0_sysbp, s_q_0_percoxyg, s_q_0_glucose, s_q_0_at, s_q_0_at, s_q_0_at)).float()
            for t in range(T_max):
                hr_logits, sysbp_logits, percoxyg_logits, glucose_logits, antibiotic_logits, vaso_logits, vent_logits = self.combiner(s_prev, rnn_output[:, t, :])
                hr_state = pyro.sample(
                    f"s{t}_hr",
                    dist.Categorical(logits=hr_logits).mask(mini_batch_mask[:, t]))
                sysbp_state = pyro.sample(
                    f"s{t}_sysbp",
                    dist.Categorical(logits=sysbp_logits).mask(mini_batch_mask[:, t]))
                glucose_state = pyro.sample(
                    f"s{t}_glucose",
                    dist.Categorical(logits=glucose_logits).mask(mini_batch_mask[:, t]))
                percoxyg_state = pyro.sample(
                    f"s{t}_percoxyg",
                    dist.Categorical(logits=percoxyg_logits).mask(mini_batch_mask[:, t]))
                if t > 0:
                    antibiotic_state = pyro.sample(
                        f"s{t}_antibiotic",
                        dist.Categorical(logits=antibiotic_logits).mask(mini_batch_mask[:, t]))
                    vent_state = pyro.sample(
                        f"s{t}_vent",
                        dist.Categorical(logits=vent_logits).mask(mini_batch_mask[:, t]))
                    vaso_state = pyro.sample(
                        f"s{t}_vaso",
                        dist.Categorical(logits=vaso_logits).mask(mini_batch_mask[:, t]))
                    s_prev = torch.column_stack((s_q_0_diab, hr_state, sysbp_state, percoxyg_state, glucose_state, antibiotic_state, vent_state, vaso_state)).float()
                else:
                    s_prev = torch.column_stack((s_q_0_diab, hr_state, sysbp_state, percoxyg_state, glucose_state, s_q_0_at, s_q_0_at, s_q_0_at)).float()