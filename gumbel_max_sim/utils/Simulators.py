from gumbel_max_sim.utils.State import State
from gumbel_max_sim.utils.Action import Action
from gumbel_max_sim.utils.MDP import MdpPyro
from gumbel_max_sim.utils.ObservationalDataset import cols
import torch
import pyro
import pyro.distributions as dist
from pyroapi import infer
from pyro.ops.indexing import Vindex
import logging


def get_simulator(name, init_state, device):
    if name == 'real':
        return MdpPyro(init_state, device)
    elif name == 'vaso1':
        return MDPVasoIntensity1(init_state, device)
    elif name == 'vaso2':
        return MDPVasoIntensity2(init_state, device)
    elif name == 'antibiotic1':
        return MDPAnitibioticIntensity1(init_state, device)
    elif name == 'antibiotic2':
        return MDPAnitibioticIntensity2(init_state, device)
    elif name == 'vent1':
        return MDPVentIntensity1(init_state, device)
    elif name == 'vent2':
        return MDPVentIntensity2(init_state, device)
    elif name == 'vaso_minus1':
        return MDPVasoIntensity_minus1(init_state, device)
    elif name == 'vaso_minus2':
        return MDPVasoIntensity_minus2(init_state, device)
    elif name == 'vent_minus1':
        return MDPVentIntensity_minus1(init_state, device)
    elif name == 'vent_minus2':
        return MDPVentIntensity_minus2(init_state, device)
    elif name == 'antibiotic_minus1':
        return MDPAnitibioticIntensity_minus1(init_state, device)
    elif name == 'antibiotic_minus2':
        return MDPAnitibioticIntensity_minus2(init_state, device)
    else:
        raise Exception('Incorrect Simulator Name')


class MDPVentIntensity2(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)
    
    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. 1.
        """
        percoxyg_probs = torch.FloatTensor([
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*self.batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self):
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .7
        """
        vent_state = torch.column_stack([self.state.vent_state]*4).reshape(self.batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.7, 0.3]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPVentIntensity1(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)
    
    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .8
        """
        percoxyg_probs = torch.FloatTensor([
            [0.2, 0.8],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*self.batch_size)])
        return percoxyg_probs.to(self.device)

    def transition_vent_off(self):
        """
        ventilation state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .4
        """
        vent_state = torch.column_stack([self.state.vent_state]*4).reshape(self.batch_size,2,2)
        percoxyg_probs = vent_state*torch.FloatTensor([
            [1.0, 0.0],
            [0.4, 0.6]
        ]) + (1-vent_state)*torch.eye(2)
        percoxyg_probs = torch.stack([percoxyg_probs]*2)
        return percoxyg_probs.to(self.device)


class MDPVentIntensity_minus1(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. .2
        """
        percoxyg_probs = torch.FloatTensor([
            [0.8, 0.2],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*self.batch_size)])
        return percoxyg_probs.to(self.device)

    # def transition_vent_off(self):
    #     """
    #     ventilation state off
    #     if ventilation was on: percent oxygen: normal -> lo w.p. 0.05
    #     """
    #     vent_state = torch.column_stack([self.state.vent_state]*4).reshape(self.batch_size,2,2)
    #     percoxyg_probs = vent_state*torch.FloatTensor([
    #         [1.0, 0.0],
    #         [0.05, 0.95]
    #     ]) + (1-vent_state)*torch.eye(2)
    #     percoxyg_probs = torch.stack([percoxyg_probs]*2)
    #     return percoxyg_probs.to(self.device)


class MDPVentIntensity_minus2(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_vent_on(self):
        """
        ventilation state on
        percent oxygen: low -> normal w.p. 0.0
        """
        percoxyg_probs = torch.FloatTensor([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        percoxyg_probs = torch.stack(2*[torch.stack([percoxyg_probs]*self.batch_size)])
        return percoxyg_probs.to(self.device)

    # def transition_vent_off(self):
    #     """
    #     ventilation state off
    #     if ventilation was on: percent oxygen: normal -> lo w.p. 0
    #     """
    #     vent_state = torch.column_stack([self.state.vent_state]*4).reshape(self.batch_size,2,2)
    #     percoxyg_probs = vent_state*torch.FloatTensor([
    #         [1.0, 0.0],
    #         [0.0, 1.0]
    #     ]) + (1-vent_state)*torch.eye(2)
    #     percoxyg_probs = torch.stack([percoxyg_probs]*2)
    #     return percoxyg_probs.to(self.device)


class MDPAnitibioticIntensity2(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_antibiotics_on(self):
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .9
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*self.batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*self.batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self):
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .6
        """
        antibiotic_state = torch.column_stack([self.state.antibiotic_state]*9).reshape(self.batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.6],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity1(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_antibiotics_on(self):
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .7
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7, 0.3]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*self.batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7, 0.3]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*self.batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    def transition_antibiotics_off(self):
        """
        antibiotics state off
        if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .4
        """
        antibiotic_state = torch.column_stack([self.state.antibiotic_state]*9).reshape(self.batch_size,3,3)
        hr_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        hr_probs = torch.stack(2*[hr_probs])
        sysbp_probs = antibiotic_state*torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ]) + (1-antibiotic_state)*torch.eye(3)
        sysbp_probs = torch.stack(2*[sysbp_probs])
        return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity_minus1(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_antibiotics_on(self):
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .3
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.3, 0.7]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*self.batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.3, 0.7]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*self.batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    # def transition_antibiotics_off(self):
    #     """
    #     antibiotics state off
    #     if antibiotics was on: heart rate, sys bp: normal -> hi w.p. 0.05
    #     """
    #     antibiotic_state = torch.column_stack([self.state.antibiotic_state]*9).reshape(self.batch_size,3,3)
    #     hr_probs = antibiotic_state*torch.FloatTensor([
    #         [1.0, 0.0, 0.0],
    #         [0.0, 0.95, 0.05],
    #         [0.0, 0.0, 1.0]
    #     ]) + (1-antibiotic_state)*torch.eye(3)
    #     hr_probs = torch.stack(2*[hr_probs])
    #     sysbp_probs = antibiotic_state*torch.FloatTensor([
    #         [1.0, 0.0, 0.0],
    #         [0.0, 0.95, 0.05],
    #         [0.0, 0.0, 1.0]
    #     ]) + (1-antibiotic_state)*torch.eye(3)
    #     sysbp_probs = torch.stack(2*[sysbp_probs])
    #     return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPAnitibioticIntensity_minus2(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_antibiotics_on(self):
        """
        antibiotics state on
        heart rate, sys bp: hi -> normal w.p. .1
        """
        hr_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9]
        ])
        hr_probs = torch.stack(2*[torch.stack([hr_probs]*self.batch_size)])

        sysbp_probs = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9]
        ])
        sysbp_probs = torch.stack(2*[torch.stack([sysbp_probs]*self.batch_size)])

        return hr_probs.to(self.device), sysbp_probs.to(self.device)

    # def transition_antibiotics_off(self):
    #     """
    #     antibiotics state off
    #     if antibiotics was on: heart rate, sys bp: normal -> hi w.p. 0.0
    #     """
    #     antibiotic_state = torch.column_stack([self.state.antibiotic_state]*9).reshape(self.batch_size,3,3)
    #     hr_probs = antibiotic_state*torch.FloatTensor([
    #         [1.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0]
    #     ]) + (1-antibiotic_state)*torch.eye(3)
    #     hr_probs = torch.stack(2*[hr_probs])
    #     sysbp_probs = antibiotic_state*torch.FloatTensor([
    #         [1.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0]
    #     ]) + (1-antibiotic_state)*torch.eye(3)
    #     sysbp_probs = torch.stack(2*[sysbp_probs])
    #     return hr_probs.to(self.device), sysbp_probs.to(self.device)


class MDPVasoIntensity2(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)
    
    def transition_vaso_on(self):
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal w.p. .2, low -> high w.p. .5, 
                    normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .1, lo -> hi w.p. .8
            raise blood glucose by 3 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.1, 0.8],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.2, 0.5],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.3, 0.0, 0.0, 0.7, 0.0],
            [0.0, 0.3, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.3, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.3, 0.7],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*self.batch_size), torch.stack([glucose_probs_diab]*self.batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .5
            for diabetics, blood pressure falls by 1 w.p. .3
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity1(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_vaso_on(self):
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal w.p. .3, low -> high w.p. .4, 
                    normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .3, lo -> hi w.p. .6
            raise blood glucose by 2 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.1, 0.3, 0.6],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.3, 0.3, 0.4],
            [0.0, 0.3, 0.7],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        glucose_probs_diab = torch.FloatTensor([
            [0.2, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.8, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.2, 0.8],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        glucose_probs_no_diab = torch.eye(5)
        glucose_probs = torch.stack((torch.stack([glucose_probs_no_diab]*self.batch_size), torch.stack([glucose_probs_diab]*self.batch_size)))
        return sysbp_probs.to(self.device), glucose_probs.to(self.device)

    def transition_vaso_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .3
            for diabetics, blood pressure falls by 1 w.p. .2
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
            [0.0, 0.2, 0.8]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.3, 0.7, 0.0],
            [0.0, 0.3, 0.7]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity_minus1(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_vaso_on(self):
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal w.p. .3, normal -> high w.p. .3,
        for diabetic:
            raise blood pressure: normal -> hi w.p. .4,
                lo -> normal w.p. .2, lo -> hi w.p. .2
            raise blood glucose by 1 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.6, 0.2, 0.2],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.7, 0.3, 0.0],
            [0.0, 0.7, 0.3],
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
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. 0.05
            for diabetics, blood pressure falls by 1 w.p. 0.0
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0],
            [0.0, 0.05, 0.95]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)


class MDPVasoIntensity_minus2(MdpPyro):
    def __init__(self, init_state, device):
        super().__init__(init_state, device)

    def transition_vaso_on(self):
        """
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal w.p. .1, normal -> high w.p. .1,
        for diabetic:
            raise blood pressure: normal -> hi w.p. .2,
                lo -> normal w.p. .1, lo -> hi w.p. .1
            raise blood glucose by 1 w.p. .5
        """
        sysbp_probs_diab = torch.FloatTensor([
            [0.8, 0.1, 0.1],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [0.9, 0.1, 0.0],
            [0.0, 0.9, 0.1],
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
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. 0.0
            for diabetics, blood pressure falls by 1 w.p. 0.0
        '''
        sysbp_probs_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs_no_diab = torch.FloatTensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        sysbp_probs = torch.stack((torch.stack([sysbp_probs_no_diab]*self.batch_size), torch.stack([sysbp_probs_diab]*self.batch_size)))
        vaso_state = torch.column_stack([self.state.vaso_state]*2*9).reshape(2,self.batch_size,3,3)
        sysbp_probs = vaso_state*sysbp_probs + (1-vaso_state)*torch.eye(3)
        return sysbp_probs.to(self.device)