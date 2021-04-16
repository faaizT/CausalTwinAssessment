import numpy as np
import pyro
from pyroapi import handlers, infer
from pyro.ops.indexing import Vindex
import logging
import pyro.distributions as dist
import torch
from torch import nn
from torch.distributions import constraints
import pyro.contrib.examples.polyphonic_data_loader as poly
from gumbel_max_sim.utils.State import State
from gumbel_max_sim.utils.Action import Action
from gumbel_max_sim.utils.MDP import MdpPyro
from gumbel_max_sim.utils.Simulators import *
from gumbel_max_sim.utils.ObservationalDataset import cols
from .utils.Distributions import CategoricalVals
from simple_model.Policy import Policy
from gumbel_max_sim.utils.Networks import Combiner, Combiner_without_rnn, Net


class GumbelMaxModel(nn.Module):
    def __init__(
        self,
        simulator_name,
        use_cuda=False,
        st_vec_dim=8,
        n_act=8,
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
        self.simulator_name = simulator_name
        self.policy.to(self.device)
        self.s0_diab_logits = nn.Parameter(torch.zeros(2))
        self.s0_diab_logits.to(self.device)
        self.s0_hr = nn.Parameter(torch.zeros((2, 3)))
        self.s0_sysbp = nn.Parameter(torch.zeros((2, 3)))
        self.s0_glucose = nn.Parameter(torch.zeros((2, 5)))
        self.s0_percoxyg = nn.Parameter(torch.zeros((2, 2)))
        self.action_probs = nn.Parameter(torch.zeros((2,3,3,2,5,2,2,2,8)))
        self.posterior_hr = nn.Parameter(torch.zeros((2,3,3,2,5,3,3,2,5,2,2,2,3)))

        self.s0_diab_net = Net(input_dim=len(cols)*2, hidden_dim=20, output_dim=2, use_cuda=use_cuda)
        self.s0_hr_net = Net(input_dim=len(cols)*2 , hidden_dim=20, output_dim=3, use_cuda=use_cuda)
        self.s0_sysbp_net = Net(input_dim=len(cols)*2 , hidden_dim=20, output_dim=3, use_cuda=use_cuda)
        self.s0_percoxyg_net = Net(input_dim=len(cols)*2 , hidden_dim=20, output_dim=2, use_cuda=use_cuda)
        self.s0_glucose_net = Net(input_dim=len(cols)*2, hidden_dim=20, output_dim=5, use_cuda=use_cuda)

        self.s1_hr_net = Net(input_dim=len(cols), hidden_dim=20, output_dim=3, use_cuda=use_cuda)
        self.s1_sysbp_net = Net(input_dim=len(cols), hidden_dim=20, output_dim=3, use_cuda=use_cuda)
        self.s1_percoxyg_net = Net(input_dim=len(cols), hidden_dim=20, output_dim=2, use_cuda=use_cuda)
        self.s1_glucose_net = Net(input_dim=len(cols), hidden_dim=20, output_dim=5, use_cuda=use_cuda)

        self.s_q_0 = nn.Parameter(torch.zeros(st_vec_dim-1))
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, actions_obs, mini_batch_mask, mini_batch_seq_lengths, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("gumbel_max", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s0_diab = pyro.sample(
                f"s0_diab_state",
                dist.Categorical(logits=self.s0_diab_logits).mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            s0_hr = pyro.sample(
                f"s0_hr",
                dist.Categorical(logits=Vindex(self.s0_hr)[s0_diab, :])
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            s0_sysbp = pyro.sample(
                f"s0_sysbp",
                dist.Categorical(logits=Vindex(self.s0_sysbp)[s0_diab, :])
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            s0_glucose = pyro.sample(
                f"s0_glucose",
                dist.Categorical(logits=Vindex(self.s0_glucose)[s0_diab, :])
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            s0_percoxyg = pyro.sample(
                f"s0_percoxyg",
                dist.Categorical(logits=Vindex(self.s0_percoxyg)[s0_diab, :])
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            a_prev = Action(action_idx=torch.zeros(len(mini_batch)))
            state = State(hr_state=s0_hr, sysbp_state=s0_sysbp, percoxyg_state=s0_percoxyg, glucose_state=s0_glucose, antibiotic_state=a_prev.antibiotic, vaso_state=a_prev.vasopressors, vent_state=a_prev.ventilation, diabetic_idx=s0_diab)
            mdp = get_simulator(name=self.simulator_name, init_state=state, device=self.device)
            mdp.emission(mini_batch, mini_batch_mask, t=0)
            for t in pyro.markov(range(T_max-1)):
                at = pyro.sample(
                    f"a{t}",
                    dist.Categorical(
                        logits=Vindex(self.action_probs)[state.diabetic_idx,
                                                         state.hr_state,
                                                         state.sysbp_state,
                                                         state.percoxyg_state,
                                                         state.glucose_state,
                                                         state.antibiotic_state.to(torch.long),
                                                         state.vaso_state.to(torch.long),
                                                         state.vent_state.to(torch.long),
                                                         :]
                    ).mask(mini_batch_mask[:, t + 1]),
                    obs=actions_obs[:, t].reshape(-1),
                ).expand(len(mini_batch))
                action = Action(action_idx=at)
                mdp.transition(action, mini_batch_mask, t + 1)
                mdp.emission(mini_batch, mini_batch_mask, t + 1)
                state = mdp.state


    def guide(self, mini_batch, actions_obs, mini_batch_mask, mini_batch_seq_lengths, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("gumbel_max", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            s_q_0_diab = pyro.sample(
                f"s0_diab_state",
                dist.Categorical(logits=self.s0_diab_net(torch.column_stack((mini_batch[:,0,:], mini_batch[:,1,:]))))
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            hr_state = pyro.sample(
                f"s0_hr",
                dist.Categorical(logits=self.s0_hr_net(torch.column_stack((mini_batch[:,0,:], mini_batch[:,1,:]))))
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            sysbp_state = pyro.sample(
                f"s0_sysbp", dist.Categorical(logits=self.s0_sysbp_net(torch.column_stack((mini_batch[:,0,:], mini_batch[:,1,:]))))
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            percoxyg_state = pyro.sample(
                f"s0_percoxyg", dist.Categorical(logits=self.s0_percoxyg_net(torch.column_stack((mini_batch[:,0,:], mini_batch[:,1,:]))))
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            glucose_state = pyro.sample(
                f"s0_glucose", dist.Categorical(logits=self.s0_glucose_net(torch.column_stack((mini_batch[:,0,:], mini_batch[:,1,:]))))
                .mask(mini_batch_mask[:, 0]),
                infer={"enumerate": "parallel"}
            )
            for t in pyro.markov(range(1, T_max)):
                hr_logits=Vindex(self.posterior_hr)[s_q_0_diab, hr_state, sysbp_state, percoxyg_state, glucose_state, mini_batch[:, t, 0].to(torch.long), mini_batch[:, t, 1].to(torch.long), mini_batch[:, t, 2].to(torch.long), mini_batch[:, t, 3].to(torch.long), mini_batch[:, t, 4].to(torch.long), mini_batch[:, t, 5].to(torch.long), mini_batch[:, t, 6].to(torch.long)]
                hr_state = pyro.sample(
                    f"s{t}_hr",
                    dist.Categorical(logits=self.s1_hr_net(mini_batch[:, t, :]))
                    .mask(mini_batch_mask[:, t]),
                    infer={"enumerate": "parallel"}
                )
                sysbp_state = pyro.sample(
                    f"s{t}_sysbp",
                    dist.Categorical(logits=self.s1_sysbp_net(mini_batch[:, t, :]))
                    .mask(mini_batch_mask[:, t]),
                    infer={"enumerate": "parallel"}
                )
                percoxyg_state = pyro.sample(
                    f"s{t}_percoxyg",
                    dist.Categorical(logits=self.s1_percoxyg_net(mini_batch[:, t, :]))
                    .mask(mini_batch_mask[:, t]),
                    infer={"enumerate": "parallel"}
                )
                glucose_state = pyro.sample(
                    f"s{t}_glucose",
                    dist.Categorical(logits=self.s1_glucose_net(mini_batch[:, t, :]))
                    .mask(mini_batch_mask[:, t]),
                    infer={"enumerate": "parallel"}
                )