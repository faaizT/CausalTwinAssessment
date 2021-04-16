from gumbel_max_sim.utils.State import State
from gumbel_max_sim.utils.Action import Action
from sepsisSimDiabetes.MDP import MDP
from gumbel_max_sim.utils.ObservationalDataset import cols
import torch
import pyro
import pyro.distributions as dist
from pyroapi import infer
from pyro.ops.indexing import Vindex
import logging


class MdpPyro(MDP):
    def __init__(self, init_state, device):
        self.state = init_state
        self.batch_size = self.state.antibiotic_state.size(0)
        self.device = device

    def transition_antibiotics_on(self):
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .5
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.5, 0.5]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*self.batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.5, 0.5]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*self.batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self):
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
        """
        antibiotic_state = torch.column_stack([self.state.antibiotic_state]*9).reshape(self.batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        """
        percoxyg_probs = torch.FloatTensor([
            [0.3, 0.7],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*self.batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self):
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        """
        vent_state = torch.column_stack([self.state.vent_state]*4).reshape(self.batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.1, 0.9]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)

    def transition_vaso_on(self):
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.5, 0.4],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.5, 0.5, 0.0, 0.0, 0.0], 
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0], 
            [0.0, 0.0, 0.0, 0.5, 0.5], 
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*self.batch_size), torch.stack([glucose_probs_diab]*self.batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.05, 0.95]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)

    def transition_fluctuate(self, action):
        '''
        all (non-treatment) states fluctuate +/- 1 w.p. .1
        exception: glucose flucuates +/- 1 w.p. .3 if diabetic
        '''
        antibiotics = torch.column_stack([action.antibiotic]*2*9).reshape(2,self.batch_size,3,3)
        antibitiotic_state = torch.column_stack([self.state.antibiotic_state]*2*9).reshape(2,self.batch_size,3,3)
        hr_fluctuate = (1-antibiotics)*(1-antibitiotic_state)
        hr_probs = hr_fluctuate*torch.FloatTensor([
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9]
        ]) + (1-hr_fluctuate)*torch.eye(3)
        ventilation = torch.column_stack([action.ventilation]*2*4).reshape(2,self.batch_size,2,2)
        vent_state = torch.column_stack([self.state.vent_state]*2*4).reshape(2,self.batch_size,2,2)
        percoxyg_fluctuate = (1-ventilation)*(1-vent_state)
        percoxyg_probs = percoxyg_fluctuate*torch.FloatTensor([
            [0.9, 0.1],
            [0.1, 0.9]
        ]) + (1-percoxyg_fluctuate)*torch.eye(2)
        vaso = torch.column_stack([action.vasopressors]*2*25).reshape(2,self.batch_size,5,5)
        glucose_fluctuate = (1-vaso)
        glucose_probs_no_diab = torch.FloatTensor([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.8, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.9]
        ])
        glucose_probs_diab = torch.FloatTensor([
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.3, 0.4, 0.3, 0.0, 0.0],
            [0.0, 0.3, 0.4, 0.3, 0.0],
            [0.0, 0.0, 0.3, 0.4, 0.3],
            [0.0, 0.0, 0.0, 0.3, 0.7]
        ])
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*self.batch_size), torch.stack([glucose_probs_diab]*self.batch_size)))
        glucose_probs = glucose_fluctuate*glucose_probs + (1-glucose_fluctuate)*torch.eye(5)
        vaso = torch.column_stack([action.vasopressors]*2*9).reshape(2,self.batch_size,3,3)
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_fluctuate = (1-vaso)*(1-vaso_state)*hr_fluctuate
        sysbp_probs = sysbp_fluctuate*torch.FloatTensor([
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9]
        ]) + (1-sysbp_fluctuate)*torch.eye(3)
        return hr_probs.to(self.device), sysbp_probs.to(self.device), glucose_probs.to(self.device), percoxyg_probs.to(self.device)

    def transition_probs(self, action):
        antibiotics = torch.column_stack([action.antibiotic]*2*9).reshape(2, self.batch_size,3,3)
        antibiotic_state = torch.column_stack([self.state.antibiotic_state]*2*9).reshape(2, self.batch_size,3,3)
        hr_antibiotics_on, sysbp_antibiotics_on = self.transition_antibiotics_on()
        hr_antibiotics_off, sysbp_antibiotics_off = self.transition_antibiotics_off()
        hr_probs = antibiotics*hr_antibiotics_on + \
                   antibiotic_state*(1-antibiotics)*hr_antibiotics_off + \
                   (1-antibiotic_state)*(1-antibiotics)*torch.eye(3)
        sysbp_probs = antibiotics*sysbp_antibiotics_on + \
                      antibiotic_state*(1-antibiotics)*sysbp_antibiotics_off + \
                      (1-antibiotic_state)*(1-antibiotics)*torch.eye(3)
        ventilation = torch.column_stack([action.ventilation]*2*4).reshape(2,self.batch_size,2,2)
        vent_state = torch.column_stack([self.state.vent_state]*2*4).reshape(2,self.batch_size,2,2)
        percoxyg_vent_on = self.transition_vent_on()
        percoxyg_vent_off = self.transition_vent_off()
        percoxyg_probs = ventilation*percoxyg_vent_on + \
                         vent_state*(1-ventilation)*percoxyg_vent_off + \
                         (1-vent_state)*(1-ventilation)*torch.eye(2)
        sysbp_vaso_on, glucose_vaso_on = self.transition_vaso_on()
        sysbp_vaso_off, glucose_vaso_off = self.transition_vaso_off(), torch.stack(2*[torch.stack([torch.eye(5)]*self.batch_size)])
        vaso = torch.column_stack([action.vasopressors]*2*9).reshape(2,self.batch_size,3,3)
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_probs = torch.matmul(sysbp_probs, vaso*sysbp_vaso_on + vaso_state*(1-vaso)*sysbp_vaso_off + (1-vaso_state)*(1-vaso)*torch.eye(3))
        vaso = torch.column_stack([action.vasopressors]*2*25).reshape(2,self.batch_size,5,5)
        vaso_state = torch.column_stack([self.state.vaso_state]*2*25).reshape(2,self.batch_size,5,5)
        glucose_probs = vaso*glucose_vaso_on + vaso_state*(1-vaso)*glucose_vaso_off + (1-vaso_state)*(1-vaso)*torch.eye(5)
        hr_fluctuate, sysbp_fluctuate, glucose_fluctuate, percoxyg_fluctuate = self.transition_fluctuate(action)
        sysbp_probs = torch.matmul(sysbp_probs, sysbp_fluctuate)
        hr_probs = torch.matmul(hr_probs, hr_fluctuate)
        glucose_probs = torch.matmul(glucose_probs, glucose_fluctuate)
        percoxyg_probs = torch.matmul(percoxyg_probs, percoxyg_fluctuate)
        hr_probs = Vindex(hr_probs)[self.state.diabetic_idx, torch.arange(self.batch_size), self.state.hr_state.to(torch.long), :]
        sysbp_probs = Vindex(sysbp_probs)[self.state.diabetic_idx, torch.arange(self.batch_size), self.state.sysbp_state.to(torch.long), :]
        glucose_probs = Vindex(glucose_probs)[self.state.diabetic_idx, torch.arange(self.batch_size), self.state.glucose_state.to(torch.long), :]
        percoxyg_probs = Vindex(percoxyg_probs)[self.state.diabetic_idx, torch.arange(self.batch_size), self.state.percoxyg_state.to(torch.long), :]
        return hr_probs, sysbp_probs, glucose_probs, percoxyg_probs

    def transition(self, action, mini_batch_mask, t):
        hr_probs, sysbp_probs, glucose_probs, percoxyg_probs = self.transition_probs(action)
        hr_state = pyro.sample(
            f"s{t}_hr",
            dist.Categorical(probs=hr_probs).mask(mini_batch_mask[:, t]),
            infer={"enumerate": "parallel"})
        self.state.hr_state = hr_state
        sysbp_state = pyro.sample(
            f"s{t}_sysbp",
            dist.Categorical(probs=sysbp_probs).mask(mini_batch_mask[:, t]),
            infer={"enumerate": "parallel"})
        self.state.sysbp_state = sysbp_state
        glucose_state = pyro.sample(
            f"s{t}_glucose",
            dist.Categorical(probs=glucose_probs).mask(mini_batch_mask[:, t]),
            infer={"enumerate": "parallel"})
        self.state.glucose_state = glucose_state
        percoxyg_state = pyro.sample(
            f"s{t}_percoxyg",
            dist.Categorical(probs=percoxyg_probs).mask(mini_batch_mask[:, t]),
            infer={"enumerate": "parallel"})
        self.state.percoxyg_state = percoxyg_state
        self.state.antibiotic_state = action.antibiotic
        self.state.vent_state = action.ventilation
        self.state.vaso_state = action.vasopressors


    def emission(self, mini_batch, mini_batch_mask, t):
        hr_sysbp_probs = torch.FloatTensor([
            [0.95, 0.05, 0.00],
            [0.05, 0.90, 0.05],
            [0.00, 0.05, 0.95]
        ])
        percoxyg_probs = torch.FloatTensor([
            [0.95, 0.05],
            [0.05, 0.95]
        ])
        glucose_probs = torch.FloatTensor([
            [0.95, 0.05, 0.00, 0.00, 0.00],
            [0.05, 0.90, 0.05, 0.00, 0.00],
            [0.00, 0.05, 0.90, 0.05, 0.00],
            [0.00, 0.00, 0.05, 0.90, 0.05],
            [0.00, 0.00, 0.00, 0.05, 0.95]
        ])
        xt_hr_state = pyro.sample(
            f"x{t}_hr",
            dist.Categorical(probs=Vindex(hr_sysbp_probs)[self.state.hr_state, :]).mask(mini_batch_mask[:, t]),
            obs=mini_batch[:, t, cols.index("hr_state")])
        xt_sysbp_state = pyro.sample(
            f"x{t}_sysbp",
            dist.Categorical(probs=Vindex(hr_sysbp_probs)[self.state.sysbp_state, :]).mask(mini_batch_mask[:, t]),
            obs=mini_batch[:, t, cols.index("sysbp_state")])
        xt_glucose_state = pyro.sample(
            f"x{t}_glucose",
            dist.Categorical(probs=Vindex(glucose_probs)[self.state.glucose_state, :]).mask(mini_batch_mask[:, t]),
            obs=mini_batch[:, t, cols.index("glucose_state")])
        xt_percoxyg_state = pyro.sample(
            f"x{t}_percoxyg",
            dist.Categorical(probs=Vindex(percoxyg_probs)[self.state.percoxyg_state, :]).mask(mini_batch_mask[:, t]),
            obs=mini_batch[:, t, cols.index("percoxyg_state")])