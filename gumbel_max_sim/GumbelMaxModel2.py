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
from .utils.Distributions import CategoricalVals
from simple_model.Policy import Policy
from gumbel_max_sim.utils.Networks import Combiner, Combiner_without_rnn, Net

cols = [
    "hr_state",
    "sysbp_state",
    "percoxyg_state",
    "glucose_state",
    "antibiotic_state",
    "vaso_state",
    "vent_state",
]


class GumbelMaxModel(nn.Module):
    def __init__(
        self,
        use_cuda=False,
        st_vec_dim=8,
        n_act=8,
        use_rnn=True,
        rnn_dim=20,
        rnn_dropout_rate=0.0,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.use_rnn = use_rnn
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
        if use_rnn:
            self.rnn = nn.RNN(
                input_size=len(cols),
                hidden_size=rnn_dim,
                nonlinearity="relu",
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                dropout=rnn_dropout_rate,
            )
            self.combiner = Combiner(z_dim=st_vec_dim, rnn_dim=rnn_dim, out_dim=2)
            self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
            self.s_q_0_diab = nn.Parameter(torch.zeros(2))
            self.s_q_0_hr = nn.Parameter(torch.zeros(3))
            self.s_q_0_sysbp = nn.Parameter(torch.zeros(3))
            self.s_q_0_percoxyg = nn.Parameter(torch.zeros(2))
            self.s_q_0_glucose_no_diab = nn.Parameter(torch.zeros(5))
            self.s_q_0_glucose_diab = nn.Parameter(torch.zeros(5))
        else:
            T_max = 5
            self.s0_diab_guide = Net(
                input_dim=len(cols) * T_max, hidden_dim=20, output_dim=2
            )
            self.s0_diab_guide.to(self.device)
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
                f"x0_hr",
                dist.Categorical(logits=self.s0_hr(s0_diab).unsqueeze(dim=0)),
                obs=mini_batch[:, 0, cols.index("hr_state")],
            )
            s0_sysbp = pyro.sample(
                f"x0_sysbp",
                dist.Categorical(logits=self.s0_sysbp(s0_diab).unsqueeze(dim=0)),
                obs=mini_batch[:, 0, cols.index("sysbp_state")],
            )
            s0_glucose = pyro.sample(
                f"x0_glucose",
                dist.Categorical(logits=self.s0_glucose(s0_diab).unsqueeze(dim=0)),
                obs=mini_batch[:, 0, cols.index("glucose_state")],
            )
            s0_percoxyg = pyro.sample(
                f"x0_percoxyg",
                dist.Categorical(logits=self.s0_percoxyg(s0_diab).unsqueeze(dim=0)),
                obs=mini_batch[:, 0, cols.index("percoxyg_state")],
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
            for t in range(T_max - 1):
                at = pyro.sample(
                    f"a{t}",
                    dist.Categorical(
                        logits=self.policy(mdp.state.get_state_tensor())
                    ).mask(mini_batch_mask[:, t + 1]),
                    obs=actions_obs[:, t].reshape(-1),
                )
                action = Action(action_idx=at)
                mdp.transition(action, mini_batch, mini_batch_mask, t + 1)

    def guide(self, mini_batch, actions_obs, mini_batch_mask, mini_batch_seq_lengths, mini_batch_reversed):
        T_max = mini_batch.size(1)
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        if self.use_rnn:
            h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
            # push the observed x's through the rnn;
            # rnn_output contains the hidden state at each time step
            rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
            # reverse the time-ordering in the hidden state and un-pack it
            rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        pyro.module("gumbel_max", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            if self.use_rnn:
                s_q_0_diab = pyro.sample("s_q_0_diab", dist.Categorical(logits=self.s_q_0_diab), infer={'is_auxiliary': True})
                s_q_0_hr = pyro.sample("s_q_0_hr", dist.Categorical(logits=self.s_q_0_hr), infer={'is_auxiliary': True})
                s_q_0_sysbp = pyro.sample("s_q_0_sysbp", dist.Categorical(logits=self.s_q_0_sysbp), infer={'is_auxiliary': True})
                s_q_0_percoxyg = pyro.sample("s_q_0_percoxyg", dist.Categorical(logits=self.s_q_0_percoxyg), infer={'is_auxiliary': True})
                s_q_0_glucose_no_diab = pyro.sample("s_q_0_glucose_no_diab", dist.Categorical(logits=self.s_q_0_glucose_no_diab), infer={'is_auxiliary': True})
                s_q_0_glucose_diab = pyro.sample("s_q_0_glucose_diab", dist.Categorical(logits=self.s_q_0_glucose_diab), infer={'is_auxiliary': True})
                s_q_0_glucose = s_q_0_diab * s_q_0_glucose_diab + (1-s_q_0_diab) * s_q_0_glucose_no_diab
                s_q_0_at = torch.zeros(len(mini_batch))
                s_q_0_at.to(self.device)
                s_prev = torch.column_stack((s_q_0_diab, s_q_0_hr, s_q_0_sysbp, s_q_0_percoxyg, s_q_0_glucose, s_q_0_at, s_q_0_at, s_q_0_at))
                s0_diab_state = pyro.sample("s0_diab_state", dist.Categorical(logits=self.combiner(s_prev, rnn_output[:, 0, :])))
            else:
                s0_diab_state = pyro.sample(
                    "s0_diab_state",
                    dist.Categorical(
                        logits=self.s0_diab_guide(
                            torch.column_stack(
                                (
                                    mini_batch[:, 0, :],
                                    mini_batch[:, 1, :],
                                    mini_batch[:, 2, :],
                                    mini_batch[:, 3, :],
                                    mini_batch[:, 4, :],
                                )
                            )
                        )
                    ),
                )
