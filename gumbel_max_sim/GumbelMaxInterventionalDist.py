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
from gumbel_max_sim.GumbelMaxModel import GumbelMaxModel


class GumbelMaxInterventionalDist(nn.Module):
    def __init__(
        self,
        model_path,
        simulator_name,
        use_cuda=False,
        st_vec_dim=8,
        n_act=8,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.simulator_name = simulator_name
        model = GumbelMaxModel(simulator_name, use_cuda, st_vec_dim, n_act)
        model.load_state_dict(torch.load(model_path + f"/model-state-final"))
        self.s0_diab_logits = model.s0_diab_logits
        self.s0_diab_logits.to(self.device)
        self.s0_hr = model.s0_hr
        self.s0_sysbp = model.s0_sysbp
        self.s0_glucose = model.s0_glucose
        self.s0_percoxyg = model.s0_percoxyg

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
                obs=mini_batch[:, 0, cols.index("hr_state")]
            ).to(torch.long)
            s0_sysbp = pyro.sample(
                f"s0_sysbp",
                dist.Categorical(logits=Vindex(self.s0_sysbp)[s0_diab, :])
                .mask(mini_batch_mask[:, 0]),
                obs=mini_batch[:, 0, cols.index("sysbp_state")]
            ).to(torch.long)
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
                obs=mini_batch[:, 0, cols.index("percoxyg_state")]
            ).to(torch.long)
            a_prev = Action(action_idx=torch.zeros(len(mini_batch)))
            state = State(hr_state=s0_hr, sysbp_state=s0_sysbp, percoxyg_state=s0_percoxyg, glucose_state=s0_glucose, antibiotic_state=a_prev.antibiotic, vaso_state=a_prev.vasopressors, vent_state=a_prev.ventilation, diabetic_idx=s0_diab)
            mdp = get_simulator(name=self.simulator_name, init_state=state, device=self.device)
            for t in pyro.markov(range(T_max-1)):
                at = actions_obs[:, t]
                action = Action(action_idx=at)
                mdp.transition(action, mini_batch, mini_batch_mask, t + 1)
                state = mdp.state