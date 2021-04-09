import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
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
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.policy = Policy(
            input_dim=st_vec_dim, hidden_1_dim=20, hidden_2_dim=20, output_dim=n_act, use_cuda=use_cuda
        )
        self.policy.to(device)
        self.s0_diab_logits = nn.Parameter(torch.zeros(2))
        self.s0_diab_logits.to(device)
        self.s0_hr = Net(input_dim=1, output_dim=3)
        self.s0_sysbp = Net(input_dim=1, output_dim=3)
        self.s0_glucose = Net(input_dim=1, output_dim=5)
        self.s0_percoxyg = Net(input_dim=1, output_dim=2)
        
        self.s0_diab_guide = Net(input_dim=14, output_dim=2) 
        self.s0_diab_guide.to(device)
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, mini_batch_reversed):
        # T_max = mini_batch.size(1)
        T_max = 2
        pyro.module("gumbel_max", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s0_diab = pyro.sample(f"s0_diab_state", dist.Categorical(logits=self.s0_diab_logits)).float()
            s0_diab = s0_diab.unsqueeze(dim=1)
            print(s0_diab.size())
            s0_hr = pyro.sample(f"x0_hr", dist.Categorical(logits=self.s0_hr(s0_diab).unsqueeze(dim=0)), obs=mini_batch[:, 0, cols.index("hr_state")])
            s0_sysbp = pyro.sample(f"x0_sysbp", dist.Categorical(logits=self.s0_sysbp(s0_diab).unsqueeze(dim=0)), obs=mini_batch[:, 0, cols.index("sysbp_state")])
            s0_glucose = pyro.sample(f"x0_glucose", dist.Categorical(logits=self.s0_glucose(s0_diab).unsqueeze(dim=0)), obs=mini_batch[:, 0, cols.index("glucose_state")])
            s0_percoxyg = pyro.sample(f"x0_percoxyg", dist.Categorical(logits=self.s0_percoxyg(s0_diab).unsqueeze(dim=0)), obs=mini_batch[:, 0, cols.index("percoxyg_state")])
            a_prev = Action(action_idx=torch.zeros(len(mini_batch)))
            state_categ = torch.column_stack((s0_hr, s0_sysbp, s0_percoxyg, s0_glucose, a_prev.antibiotic, a_prev.vasopressors, a_prev.ventilation, s0_diab.reshape(-1)))
            mdp = MdpPyro(init_state_categ=state_categ)
            # state = State(state_idx=int(st.item()), idx_type="full")
            # st_vec = state.get_full_state_vector()
            for t in range(T_max):
                # self.emission(mini_batch, state, t, i)
                # TODO: FIX THIS
                # if (mini_batch[i, t, :] == -1).sum() > 0:
                #     break
                at = pyro.sample(
                    f"a{t}",
                    dist.Categorical(logits=self.policy(mdp.state.get_state_tensor())),
                    obs=mini_batch[:, t, cols.index("A_t")],
                )
                action = Action(action_idx=at)
                if t < T_max - 1:
                    mdp.transition(action, mini_batch, t+1)
                    # next_state_id = mdp.state.get_state_idx(idx_type="full")
                    # st_probs = torch.zeros(1440)
                    # st_probs[next_state_id] = 1.0
                    # st = pyro.sample(f"s{t+1}_{i}", dist.Categorical(probs=st_probs))
                    # state = State(state_idx=int(st.item()), idx_type="full")
                    # st_vec = state.get_full_state_vector()

    def guide(self, mini_batch, mini_batch_reversed):
        # T_max = mini_batch.size(1)
        T_max = 2
        pyro.module("gumbel_max", self)
        logits = self.s0_diab_guide(
                torch.column_stack((
                mini_batch[:, 0, :7], mini_batch[:, 1, :7])))
        print("logits: shape", logits.size())
        s0_diab_state = pyro.sample(
            "s0_diab_state",
            dist.Categorical(logits=self.s0_diab_guide(
                torch.column_stack((
                mini_batch[:, 0, :7], mini_batch[:, 1, :7]))))
                )
        # s_prev = State(
        #     state_idx=s_prev_id.item(), idx_type="full"
        # ).get_full_state_vector()
        # for i in pyro.plate("s_minibatch", len(mini_batch)):
        #     for t in range(T_max):
        #         if (mini_batch_reversed[i, t, :] == -1).sum() < 8:
        #             break
        #     if self.use_rnn:
        #         rnn_output, _ = self.rnn(
        #             (mini_batch_reversed[i, t:, :]).reshape((1, T_max - t, 8)),
        #             h_0_contig,
        #         )
        #         rnn_output = torch.flip(rnn_output, [1])
        #     for t in range(T_max):
        #         # TODO: FIX THIS
        #         if (mini_batch[i, t, :] == -1).sum() > 1:
        #             break
        #         if self.use_rnn:
        #             logits = self.combiner(
        #                 torch.tensor(s_prev).float(),
        #                 rnn_output[0, t, :],
        #             )
        #         else:
        #             logits = self.combiner(
        #                 torch.tensor(s_prev).float(),
        #             )
        #         st = pyro.sample(f"s{t}_{i}", dist.Categorical(logits=logits))
        #         s_prev = State(
        #             state_idx=st.item(), idx_type="full"
        #         ).get_full_state_vector()
