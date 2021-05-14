import numpy as np
import pandas as pd
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


class SimulatorDataGenerator(nn.Module):
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
        self.s0_policy = torch.ones(n_act)
        self.patient_idx = 0
        if use_cuda:
            self.cuda()

    def transition(self, state, action):
        mdp = get_simulator(name=self.simulator_name, init_state=state, device=self.device)
        hr_probs, sysbp_probs, glucose_probs, percoxyg_probs = mdp.transition_probs(action)
        hr_state = dist.Categorical(probs=hr_probs).sample().to(torch.long).squeeze()
        sysbp_state = dist.Categorical(probs=sysbp_probs).sample().to(torch.long).squeeze()
        glucose_state = dist.Categorical(probs=glucose_probs).sample().to(torch.long).squeeze()
        percoxyg_state = dist.Categorical(probs=percoxyg_probs).sample().to(torch.long).squeeze()
        state.hr_state = hr_state
        state.sysbp_state = sysbp_state
        state.glucose_state = glucose_state
        state.percoxyg_state = percoxyg_state
        state.antibiotic_state = action.antibiotic.squeeze()
        state.vent_state = action.ventilation.squeeze()
        state.vaso_state = action.vasopressors.squeeze()
        return state

    def generate_trajectory(self):
        T_max = 2
        s0_diab = dist.Categorical(logits=self.s0_diab_logits).sample([10]).to(torch.long).squeeze()
        s0_hr = dist.Categorical(logits=Vindex(self.s0_hr)[s0_diab, :]).sample().to(torch.long).squeeze()
        s0_sysbp = dist.Categorical(logits=Vindex(self.s0_sysbp)[s0_diab, :]).sample().to(torch.long).squeeze()
        s0_glucose = dist.Categorical(logits=Vindex(self.s0_glucose)[s0_diab, :]).sample().to(torch.long).squeeze()
        s0_percoxyg = dist.Categorical(logits=Vindex(self.s0_percoxyg)[s0_diab, :]).sample().to(torch.long).squeeze()
        a_prev = Action(action_idx=torch.zeros(10))
        state = State(hr_state=s0_hr, sysbp_state=s0_sysbp, percoxyg_state=s0_percoxyg, glucose_state=s0_glucose, antibiotic_state=a_prev.antibiotic.squeeze(), vaso_state=a_prev.vasopressors.squeeze(), vent_state=a_prev.ventilation.squeeze(), diabetic_idx=s0_diab)
        df = state.get_dataframe()
        df['id'] = np.array(range(10)) + self.patient_idx
        df['t'] = 0
        for t in pyro.markov(range(T_max-1)):
            at = dist.Categorical(logits=self.s0_policy).sample([10])
            action = Action(action_idx=at.to(torch.float))
            state = self.transition(state, action)
            df2 = state.get_dataframe()
            df2['id'] = np.array(range(10)) + self.patient_idx
            df2['t'] = 1
        self.patient_idx += 10
        return pd.concat([df, df2], ignore_index=True).sort_values('id')
    
    def save_trajectories(self, n_trajectories, exportdir):
        df = pd.DataFrame()
        for i in range(int(n_trajectories/10)):
            df = pd.concat([df, self.generate_trajectory()], ignore_index=True)
        df.to_csv(f"{exportdir}/data_{self.simulator_name}.csv", index=False)