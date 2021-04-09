'''
Action:             0                1                  2 ...
hr_probs:       [       ]         [         ]       [         ]
sysbp_probs:    
 .
 .
 .



'''





from gumbel_max_sim.utils.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.MDP import MDP
import torch


class MdpPyro(MDP):
    def __init__(self, init_state_idx=None, init_state_categ=None, init_state_idx_type="full"):
        self.state = State(state_idx=init_state_idx, state_categs=init_state_categ)
        self.batch_size = self.state.hr_state.size(0)

    # def pyro_transition()

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
        hr_probs = hr_probs.gather(1, self.state.hr_state.reshape(self.batch_size, 1, 1).to(dtype=torch.int64))
        sysbp_probs = sysbp_probs.gather(1, self.state.sysbp_state.reshape(self.batch_size, 1, 1).to(dtype=torch.int64))
        glucose_probs = glucose_probs.gather(1, self.state.glucose_state.reshape(self.batch_size, 1, 1).to(dtype=torch.int64))
        percoxyg_probs = percoxyg_probs.gather(1, self.state.percoxyg_state.reshape(self.batch_size, 1, 1).to(dtype=torch.int64))
        return hr_probs, sysbp_probs, glucose_probs, percoxyg_probs


