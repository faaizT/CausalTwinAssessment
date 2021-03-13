import pyro
import torch

from max_likelihood.ObservationalDataset import cols
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


class S_0:
    def __init__(self, data=None):
        if data is None:
            self.gender = pyro.sample("s_0_gender", dist.Bernoulli(0.5), infer={'is_auxiliary': True})
        else:
            self.gender = pyro.sample("s_0_gender", dist.Bernoulli(0.5), obs=data[:, 0, cols.index('xt_gender')])
        # # 0 is male, 1 female
        self.hr = pyro.sample("s_0_hr", dist.Normal((1-self.gender)*72 + self.gender*78, 16))
        self.rr = pyro.sample("s_0_rr", dist.Normal((1-self.gender)*18 + self.gender*20, 5))
        max_sysbp = (1 - self.gender) * max_sysbp_men + self.gender * max_sysbp_women
        max_diabp = (1 - self.gender) * max_diabp_men + self.gender * max_diabp_women
        min_sysbp = (1 - self.gender) * min_sysbp_men + self.gender * min_sysbp_women
        min_diabp = (1 - self.gender) * min_diabp_men + self.gender * min_diabp_women
        self.sysbp = pyro.sample("s_0_sysbp", dist.Normal((min_sysbp+max_sysbp)/2, 25))
        self.diabp = pyro.sample("s_0_diabp", dist.Normal((min_diabp+max_diabp)/2, 25))
        max_weight = (1 - self.gender) * max_weight_men + self.gender * max_weight_women
        min_weight = (1 - self.gender) * min_weight_men + self.gender * min_weight_women
        self.weight = pyro.sample("s_0_weight", dist.Normal((min_weight + max_weight)/2, 25))
        max_height = (1 - self.gender) * max_height_men + self.gender * max_height_women
        min_height = (1 - self.gender) * min_height_men + self.gender * min_height_women
        self.height = pyro.sample("s_0_height", dist.Normal((min_height + max_height)/2, 35))
        self.wbc_count = pyro.sample("s_0_wbc_count", dist.Normal(7500, 1000))
        pain_stimulus_probs = torch.ones((self.sysbp.size(0), 10))/10
        self.pain_stimulus = pyro.sample("s_0_pain_stimulus", dist.Categorical(pain_stimulus_probs))
        socio_econ_probs = torch.ones((self.sysbp.size(0), 3)) / 3
        self.socio_econ = pyro.sample("s_0_socio_econ", dist.Categorical(socio_econ_probs))
        self.fio2 = pyro.sample("s_0_fio2", dist.Normal(0.29, 0.05))

    def generate_state(self):
        return PatientState(gender=self.gender, hr=self.hr, rr=self.rr,
                            sysbp=self.sysbp, diabp=self.diabp, weight=self.weight,
                            height=self.height, wbc_count=self.wbc_count, pain_stimulus=self.pain_stimulus,
                            socio_econ=self.socio_econ, fio2=self.fio2)
