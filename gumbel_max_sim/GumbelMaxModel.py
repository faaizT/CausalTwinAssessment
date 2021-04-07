import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
from sepsisSimDiabetes.State import State
from .utils.Distributions import CategoricalVals
from simple_model.Policy import Policy
from .utils.Networks import Combiner

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
        self, use_cuda=False, st_dim=8, n_act=8, rnn_dim=30, rnn_dropout_rate=0.0
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.rnn = nn.RNN(
            input_size=st_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=1,
            dropout=rnn_dropout_rate,
        )
        self.combiner = Combiner(z_dim=st_dim, rnn_dim=rnn_dim, out_dim=1440)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.policy = Policy(
            input_dim=st_dim, hidden_1_dim=20, hidden_2_dim=20, output_dim=n_act
        )
        self.s0_probs = nn.Parameter(torch.zeros(1440))
        self.s_q_0 = nn.Parameter(torch.zeros(st_dim))
        self.s0_probs_guide = nn.Parameter(torch.zeros(1440))
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("gumbel_max", self)
        for i in pyro.plate("s_minibatch", len(mini_batch)):
            for t in range(T_max):
                # TODO: FIX THIS
                if (mini_batch[i, t, :] == -1).sum() > 1:
                    break
                st = pyro.sample(f"s{t}_{i}", dist.Categorical(logits=self.s0_probs))
                state = State(state_idx=int(st.item()), idx_type="full")
                st_vec = state.get_full_state_vector()
                if mini_batch[i, t, cols.index("A_t")] != -1:
                    a_0 = pyro.sample(
                        f"a{t}_{i}",
                        dist.Categorical(
                            logits=self.policy(torch.tensor(st_vec).float())
                        ),
                        obs=mini_batch[i, t, cols.index("A_t")],
                    )
                probs = torch.ones(3) * 0.05
                probs[st_vec[0]] = 0.9
                x_0_hr = pyro.sample(
                    f"x{t}_hr_{i}",
                    dist.Categorical(probs=probs),
                    obs=mini_batch[i, t, cols.index("hr_state")],
                )

    def guide(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        h_0_contig = self.h_0
        s_prev_id = dist.Categorical(logits=self.s0_probs_guide).sample()
        s_prev = State(
            state_idx=s_prev_id.item(), idx_type="full"
        ).get_full_state_vector()
        pyro.module("gumbel_max", self)
        for i in pyro.plate("s_minibatch", len(mini_batch)):
            for t in range(T_max):
                if (mini_batch_reversed[i, t, :] == -1).sum() < 8:
                    break
            rnn_output, _ = self.rnn(
                (mini_batch_reversed[i, t:, :]).reshape((1, T_max - t, 8)), h_0_contig
            )
            rnn_output = torch.flip(rnn_output, [1])
            for t in range(T_max):
                # TODO: FIX THIS
                if (mini_batch[i, t, :] == -1).sum() > 1:
                    break
                logits = self.combiner(
                    torch.tensor(s_prev).reshape(1, 8).float(),
                    rnn_output[0, t, :].reshape(1, 30),
                )
                st = pyro.sample(f"s{t}_{i}", dist.Categorical(logits=logits))
                s_prev = State(
                    state_idx=st.item(), idx_type="full"
                ).get_full_state_vector()
