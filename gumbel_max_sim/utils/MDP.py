from gumbel_max_sim.utils.State import State
from gumbel_max_sim.utils.Action import Action
from sepsisSimDiabetes.MDP import MDP
from gumbel_max_sim.GumbelMaxModel import cols
import torch
import pyro
import pyro.distributions as dist


class MdpPyro(MDP):
    def __init__(self, init_state_idx=None, init_state_categ=None, init_state_idx_type="full"):
        self.state = State(state_idx=init_state_idx, state_categs=init_state_categ)
        self.batch_size = self.state.hr_state.size(0)

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
        hr_probs = torch.stack([hr_probs]*self.batch_size)

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [0.0, 0.5, 0.5]
        ])
        sysbp_probs = torch.stack([sysbp_probs]*self.batch_size)

        return hr_probs, sysbp_probs


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
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0], 
            [0.0, 0.9, 0.1], 
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        return hr_probs, sysbp_probs

    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .7
        """
        percoxyg_probs = torch.FloatTensor([
            [0.3, 0.7],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack([percoxyg_probs]*self.batch_size)
        return percoxyg_probs


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
        return percoxyg_probs

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
        diabetic_idx =  torch.column_stack([self.state.diabetic_idx]*9).reshape(self.batch_size,3,3)
        sysbp_probs = diabetic_idx*torch.FloatTensor([
            [0.1, 0.5, 0.4],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ]) + (1-diabetic_idx)*torch.FloatTensor([
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])

        diabetic_idx = torch.column_stack([self.state.diabetic_idx]*25).reshape(self.batch_size,5,5)
        glucose_probs = diabetic_idx*torch.FloatTensor([
            [0.5, 0.5, 0.0, 0.0, 0.0], 
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0], 
            [0.0, 0.0, 0.0, 0.5, 0.5], 
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ]) + (1-diabetic_idx)*torch.eye(5)
        return sysbp_probs, glucose_probs

    def transition_vaso_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        diabetic_idx = torch.column_stack([self.state.diabetic_idx]*9).reshape(self.batch_size,3,3)
        sysbp_probs = diabetic_idx*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.05, 0.95]
        ]) + (1-diabetic_idx)*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ])
        vaso_state = torch.column_stack([self.state.vaso_state]*9).reshape(self.batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs

    def transition_fluctuate(self, action):
        '''
        all (non-treatment) states fluctuate +/- 1 w.p. .1
        exception: glucose flucuates +/- 1 w.p. .3 if diabetic
        '''
        antibiotics = torch.column_stack([action.antibiotic]*9).reshape(self.batch_size,3,3)
        antibitiotic_state = torch.column_stack([self.state.antibiotic_state]*9).reshape(self.batch_size,3,3)
        hr_sysbp_fluctuate = (1-antibiotics)*(1-antibitiotic_state)
        hr_sysbp_probs = hr_sysbp_fluctuate*torch.FloatTensor([
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.1, 0.9]
        ]) + (1-hr_sysbp_fluctuate)*torch.eye(3)
        ventilation = torch.column_stack([action.ventilation]*4).reshape(self.batch_size,2,2)
        vent_state = torch.column_stack([self.state.vent_state]*4).reshape(self.batch_size,2,2)
        percoxyg_fluctuate = (1-ventilation)*(1-vent_state)
        percoxyg_probs = percoxyg_fluctuate*torch.FloatTensor([
            [0.9, 0.1],
            [0.1, 0.9]
        ]) + (1-percoxyg_fluctuate)*torch.eye(2)
        vaso = torch.column_stack([action.vasopressors]*25).reshape(self.batch_size,5,5)
        vaso_state = torch.column_stack([self.state.vaso_state]*25).reshape(self.batch_size,5,5)
        diab = torch.column_stack([self.state.diabetic_idx]*25).reshape(self.batch_size,5,5)
        glucose_fluctuate = (1-vaso)*(1-vaso_state)
        glucose_probs = glucose_fluctuate*diab*torch.FloatTensor([
            [0.7, 0.3, 0.0, 0.0, 0.0],
            [0.3, 0.4, 0.3, 0.0, 0.0],
            [0.0, 0.3, 0.4, 0.3, 0.0],
            [0.0, 0.0, 0.3, 0.4, 0.3],
            [0.0, 0.0, 0.0, 0.3, 0.7]
        ]) + glucose_fluctuate*(1-diab)*torch.FloatTensor([
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0, 0.0],
            [0.0, 0.1, 0.8, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.9]
        ]) + (1-glucose_fluctuate)*torch.eye(5)
        return hr_sysbp_probs, glucose_probs, percoxyg_probs

    def transition_probs(self, action):
        antibiotics = torch.column_stack([action.antibiotic]*9).reshape(self.batch_size,3,3)
        hr_antibiotics_on, sysbp_antibiotics_on = self.transition_antibiotics_on()
        hr_antibiotics_off, sysbp_antibiotics_off = self.transition_antibiotics_off()
        hr_probs = antibiotics*hr_antibiotics_on + (1-antibiotics)*hr_antibiotics_off
        sysbp_probs = antibiotics*sysbp_antibiotics_on + (1-antibiotics)*sysbp_antibiotics_off
        ventilation = torch.column_stack([action.ventilation]*4).reshape(self.batch_size,2,2)
        percoxyg_vent_on = self.transition_vent_on()
        percoxyg_vent_off = self.transition_vent_off()
        percoxyg_probs = ventilation*percoxyg_vent_on + (1-ventilation)*percoxyg_vent_off
        sysbp_vaso_on, glucose_vaso_on = self.transition_vaso_on()
        sysbp_vaso_off, glucose_vaso_off = self.transition_vaso_off(), torch.stack([torch.eye(5)]*self.batch_size)
        vaso = torch.column_stack([action.vasopressors]*9).reshape(self.batch_size,3,3)
        sysbp_probs = torch.matmul(sysbp_probs, vaso*sysbp_vaso_on + (1-vaso)*sysbp_vaso_off)
        vaso = torch.column_stack([action.vasopressors]*25).reshape(self.batch_size,5,5)
        glucose_probs = vaso*glucose_vaso_on + (1-vaso)*glucose_vaso_off
        hr_sysbp_fluctuate, glucose_fluctuate, percoxyg_fluctuate = self.transition_fluctuate(action)
        sysbp_probs = torch.matmul(sysbp_probs, hr_sysbp_fluctuate)
        hr_probs = torch.matmul(hr_probs, hr_sysbp_fluctuate)
        glucose_probs = torch.matmul(glucose_probs, glucose_fluctuate)
        percoxyg_probs = torch.matmul(percoxyg_probs, percoxyg_fluctuate)
        hr_idx = torch.column_stack([self.state.hr_state]*3).reshape((self.batch_size,1,3))
        hr_probs = hr_probs.gather(1, hr_idx.to(dtype=torch.int64)).reshape((self.batch_size, 3))
        sysbp_idx = torch.column_stack([self.state.sysbp_state]*3).reshape((self.batch_size,1,3))
        sysbp_probs = sysbp_probs.gather(1, sysbp_idx.to(dtype=torch.int64)).reshape((self.batch_size, 3))
        glucose_idx = torch.column_stack([self.state.glucose_state]*5).reshape((self.batch_size,1,5))
        glucose_probs = glucose_probs.gather(1, glucose_idx.to(dtype=torch.int64)).reshape((self.batch_size, 5))
        percoxyg_idx = torch.column_stack([self.state.percoxyg_state]*2).reshape((self.batch_size,1,2))
        percoxyg_probs = percoxyg_probs.gather(1, percoxyg_idx.to(dtype=torch.int64)).reshape((self.batch_size, 2))
        return hr_probs, sysbp_probs, glucose_probs, percoxyg_probs

    def transition(self, action, mini_batch, mini_batch_mask, t):
        hr_probs, sysbp_probs, glucose_probs, percoxyg_probs = self.transition_probs(action)
        hr_state = pyro.sample(
            f"x{t}_hr", 
            dist.Categorical(probs=hr_probs).mask(mini_batch_mask[:, t]), 
            obs=mini_batch[:, t, cols.index("hr_state")])
        self.state.hr_state = hr_state
        sysbp_state = pyro.sample(
            f"x{t}_sysbp", 
            dist.Categorical(probs=sysbp_probs).mask(mini_batch_mask[:, t]), 
            obs=mini_batch[:, t, cols.index("sysbp_state")])
        self.state.sysbp_state = sysbp_state
        glucose_state = pyro.sample(
            f"x{t}_glucose", 
            dist.Categorical(probs=glucose_probs).mask(mini_batch_mask[:, t]), 
            obs=mini_batch[:, t, cols.index("glucose_state")])
        self.state.glucose_state = glucose_state
        percoxyg_state = pyro.sample(
            f"x{t}_percoxyg", 
            dist.Categorical(probs=percoxyg_probs).mask(mini_batch_mask[:, t]), 
            obs=mini_batch[:, t, cols.index("percoxyg_state")])
        self.state.percoxyg_state = percoxyg_state
        self.state.antibiotic_state = action.antibiotic
        self.state.vent_state = action.ventilation
        self.state.vaso_state = action.vasopressors

