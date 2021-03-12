import pyro
import torch
from observational_model.PatientState import PatientState
import numpy as np
import pyro.distributions as dist

min_sysbp_men = 90
max_sysbp_men = 160
min_sysbp_women = 100
max_sysbp_women = 165
min_diabp_men = 60
max_diabp_men = 110
min_diabp_women = 65
max_diabp_women = 120
min_weight_men = 50
max_weight_men = 140
min_weight_women = 40
max_weight_women = 90
min_height_men = 130
max_height_men = 200
min_height_women = 120
max_height_women = 180


class Model:
    def __init__(self, current_state=None):
        if current_state is None:
            self.current_state = InitialStateGenerator().generate_state()
        else:
            self.current_state = current_state
        self.current_gender = getattr(self.current_state, 'gender')
        self.current_hr = getattr(self.current_state, 'hr')
        self.current_rr = getattr(self.current_state, 'rr')
        self.current_sysbp = getattr(self.current_state, 'sysbp')
        self.current_diabp = getattr(self.current_state, 'diabp')
        self.current_fio2 = getattr(self.current_state, 'fio2')
        self.current_weight = getattr(self.current_state, 'weight')
        self.current_height = getattr(self.current_state, 'height')
        self.current_wbc_count = getattr(self.current_state, 'wbc_count')
        self.current_socio_econ = getattr(self.current_state, 'socio_econ')
        self.current_pain_stimulus = getattr(self.current_state, 'pain_stimulus')

    def transition_to_next_state(self, action):
        self.current_hr += action*(np.random.normal(0,1)**2)*(1+self.current_socio_econ)/3
        self.current_rr += action*(np.random.normal(0,1)**2)*(1+self.current_socio_econ)/3
        if self.current_gender == 0:
            self.current_sysbp += action/3*(np.random.uniform(0, max_sysbp_men-self.current_sysbp))
            self.current_diabp += action/3*(np.random.uniform(0, max_diabp_men-self.current_diabp))
        else:
            self.current_sysbp += action / 3 * (np.random.uniform(0, max_sysbp_women - self.current_sysbp))
            self.current_diabp += action / 3 * (np.random.uniform(0, max_diabp_women - self.current_diabp))
        self.current_wbc_count *= (1-action/3*np.random.normal(0.1,0.001))
        if np.random.uniform(0, 1) > (1 - action / 3) and self.current_pain_stimulus > 0:
            self.current_pain_stimulus = np.random.randint(0, self.current_pain_stimulus)
        self.current_state = PatientState(self.current_gender, self.current_hr, self.current_rr,
                                          self.current_sysbp, self.current_diabp, self.current_fio2,
                                          self.current_weight, self.current_height, self.current_wbc_count,
                                          self.current_socio_econ, self.current_pain_stimulus)

    def get_state(self):
        return self.current_state

    def copy(self):
        return Model(self.current_state.copy())


class InitialStateGenerator:
    def __init__(self):
        self.gender = pyro.sample("z_0_gender", dist.Bernoulli(0.5))
        # 0 is male, 1 female
        self.hr = torch.where(self.gender == 0,
                              pyro.sample("z_0_hr", dist.Normal(72,16)),
                              pyro.sample("z_0_hr", dist.Normal(78,16)))
        self.rr = torch.where(self.gender == 0,
                              pyro.sample("z_0_rr", dist.Normal(18, 5)),
                              pyro.sample("z_0_rr", dist.Normal(20, 5)))
        self.weight = torch.where(self.gender == 0,
                                  pyro.sample("z_0_weight", dist.Uniform(min_weight_men, max_weight_men)),
                                  pyro.sample("z_0_weight", dist.Uniform(min_weight_women, max_weight_women)))
        self.height = torch.where(self.gender == 0,
                                  pyro.sample("z_0_height", dist.Uniform(min_height_men, max_height_men)),
                                  pyro.sample("z_0_height", dist.Uniform(min_height_women, max_height_women)))
        self.sysbp = torch.where(self.gender == 0,
                                 pyro.sample("z_0_sysbp", dist.Uniform(min_sysbp_men, max_sysbp_men)),
                                 pyro.sample("z_0_sysbp", dist.Uniform(min_sysbp_women, max_sysbp_women)))
        self.diabp = torch.where(self.gender == 0,
                                 pyro.sample("z_0_diabp", dist.Uniform(min_diabp_men, max_diabp_men)),
                                 pyro.sample("z_0_diabp", dist.Uniform(min_diabp_women, max_diabp_women)))
        self.fio2 = 0.21 + pyro.sample("z_0_fio2", dist.Normal(0.2,0.05))**2
        self.wbc_count = pyro.sample("z_0_wbc_count", dist.Normal(7500,1000))
        # Higher socio_econ value suggests richer patients
        self.socio_econ = np.floor(pyro.sample("z_0_socio_econ", dist.Uniform(0,3)))
        # Higher value suggests higher pain stimulus
        self.pain_stimulus = np.floor(pyro.sample("z_0_pain_stimulus", dist.Uniform(0,10)))

    def generate_state(self):
        return PatientState(self.gender, self.hr, self.rr,
                            self.sysbp, self.diabp, self.fio2,
                            self.weight, self.height, self.wbc_count,
                            self.socio_econ, self.pain_stimulus)


def physician_policy(xt, ut):
    if xt.gender == 0:
        if (xt.hr >= 80 and xt.diabp >= 97.5 and xt.sysbp >= 142.5 and ut.facial_expression <= 2) or ut.socio_econ == 0:
            return 0
        if xt.hr >= 72 and xt.diabp >= 85 and xt.sysbp >= 125 and ut.facial_expression <= 4:
            return 1
        if xt.hr >= 64 and xt.diabp >= 72.5 and xt.sysbp >= 107.5 and ut.facial_expression <= 6:
            return 2
        return 3
    else:
        if (xt.hr >= 86 and xt.diabp >= 106.25 and xt.sysbp >= 148.75 and ut.facial_expression <= 2) or ut.socio_econ == 0:
            return 0
        if xt.hr >= 78 and xt.diabp >= 92.5 and xt.sysbp >= 132.5 and ut.facial_expression <= 4:
            return 1
        if xt.hr >= 70 and xt.diabp >= 78.75 and xt.sysbp >= 116.25 and ut.facial_expression <= 6:
            return 2
        return 3
