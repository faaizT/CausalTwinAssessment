import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.MDP import MDP
from .utils.Distributions import CategoricalVals
from simple_model.Policy import Policy
from .utils.Networks import Combiner, Combiner_without_rnn

cols = [
    "hr_state",
    "sysbp_state",
    "percoxyg_state",
    "glucose_state",
    "antibiotic_state",
    "vaso_state",
    "vent_state",
    "A_t",
]


class GumbelMaxModel(nn.Module):
    def __init__(
        self,
        use_cuda=False,
        st_vec_dim=8,
        n_st=1440,
        n_act=8,
        use_rnn=False,
        rnn_dim=400,
        rnn_dropout_rate=0.0,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = nn.RNN(
                input_size=st_vec_dim,
                hidden_size=rnn_dim,
                nonlinearity="relu",
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                dropout=rnn_dropout_rate,
            )
            self.combiner = Combiner(z_dim=st_vec_dim, rnn_dim=rnn_dim, out_dim=n_st)
        else:
            self.combiner = Combiner_without_rnn(
                z_dim=st_vec_dim, hidden_dim=400, out_dim=n_st
            )
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.policy = Policy(
            input_dim=st_vec_dim, hidden_1_dim=20, hidden_2_dim=20, output_dim=n_act
        )
        self.s0_probs = nn.Parameter(torch.zeros(n_st))
        self.s0_probs_guide = nn.Parameter(torch.zeros(n_st))
        if use_cuda:
            self.cuda()

    def emission(self, mini_batch, state, t, i):
        hr_probs = torch.ones(3) * 0.05
        hr_probs[state.hr_state] = 0.9
        xt_hr = pyro.sample(
            f"x{t}_hr_{i}",
            dist.Categorical(probs=hr_probs),
            obs=mini_batch[i, t, cols.index("hr_state")],
        )
        sysbp_probs = torch.ones(3) * 0.05
        sysbp_probs[state.sysbp_state] = 0.9
        xt_sysbp = pyro.sample(
            f"x{t}_sysbp_{i}",
            dist.Categorical(probs=sysbp_probs),
            obs=mini_batch[i, t, cols.index("sysbp_state")],
        )
        percoxyg_probs = torch.ones(2) * 0.05
        percoxyg_probs[state.percoxyg_state] = 0.95
        xt_percoxyg = pyro.sample(
            f"x{t}_percoxyg_{i}",
            dist.Categorical(probs=percoxyg_probs),
            obs=mini_batch[i, t, cols.index("percoxyg_state")],
        )
        glucose_probs = torch.ones(5) * 0.05
        glucose_probs[state.glucose_state] = 0.8
        xt_glucose = pyro.sample(
            f"x{t}_glucose_{i}",
            dist.Categorical(probs=glucose_probs),
            obs=mini_batch[i, t, cols.index("glucose_state")],
        )

    def model(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("gumbel_max", self)
        for i in pyro.plate("s_minibatch", len(mini_batch)):
            st = pyro.sample(f"s0_{i}", dist.Categorical(logits=self.s0_probs))
            mdp = MDP(init_state_idx=int(st.item()), init_state_idx_type="full")
            state = State(state_idx=int(st.item()), idx_type="full")
            st_vec = state.get_full_state_vector()
            for t in range(T_max):
                self.emission(mini_batch, state, t, i)
                # TODO: FIX THIS
                if (mini_batch[i, t, :] == -1).sum() > 0:
                    break
                at = pyro.sample(
                    f"a{t}_{i}",
                    dist.Categorical(logits=self.policy(torch.tensor(st_vec).float())),
                    obs=mini_batch[i, t, cols.index("A_t")],
                )
                action = Action(action_idx=at.item())
                if t < T_max - 1:
                    mdp.transition(action)
                    next_state_id = mdp.state.get_state_idx(idx_type="full")
                    st_probs = torch.zeros(1440)
                    st_probs[next_state_id] = 1.0
                    st = pyro.sample(f"s{t+1}_{i}", dist.Categorical(probs=st_probs))
                    state = State(state_idx=int(st.item()), idx_type="full")
                    st_vec = state.get_full_state_vector()

    def guide(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        h_0_contig = self.h_0
        s_prev_id = pyro.sample(
            "s_q_0",
            dist.Categorical(logits=self.s0_probs_guide),
            infer={"is_auxiliary": True},
        )
        s_prev = State(
            state_idx=s_prev_id.item(), idx_type="full"
        ).get_full_state_vector()
        pyro.module("gumbel_max", self)
        for i in pyro.plate("s_minibatch", len(mini_batch)):
            for t in range(T_max):
                if (mini_batch_reversed[i, t, :] == -1).sum() < 8:
                    break
            if self.use_rnn:
                rnn_output, _ = self.rnn(
                    (mini_batch_reversed[i, t:, :]).reshape((1, T_max - t, 8)),
                    h_0_contig,
                )
                rnn_output = torch.flip(rnn_output, [1])
            for t in range(T_max):
                # TODO: FIX THIS
                if (mini_batch[i, t, :] == -1).sum() > 1:
                    break
                if self.use_rnn:
                    logits = self.combiner(
                        torch.tensor(s_prev).float(),
                        rnn_output[0, t, :],
                    )
                else:
                    logits = self.combiner(
                        torch.tensor(s_prev).float(),
                    )
                st = pyro.sample(f"s{t}_{i}", dist.Categorical(logits=logits))
                s_prev = State(
                    state_idx=st.item(), idx_type="full"
                ).get_full_state_vector()
