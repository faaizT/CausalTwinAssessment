import pyro
import numpy as np
import torch
from torch import nn
from max_likelihood.ObservationalDataset import cols, ObservationalDataset
from observational_model.PatientState import PatientState
import pyro.distributions as dist
from observational_model.Model import InitialStateGenerator

# TODO: move to appropriate location
# TODO: deal with tensors in St, Xt and Ut
from observational_model.PatientState import Xt
from max_likelihood.PolicyNetwork import PolicyNetwork

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


class SimulatorModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.policy = PolicyNetwork()

    @staticmethod
    def transition_to_next_state(state: PatientState, action, t):
        current_hr = state.hr
        current_rr = state.rr
        current_gender = state.gender
        current_sysbp = state.sysbp
        current_diabp = state.diabp
        current_wbc_count = state.wbc_count
        current_pain_stimulus = state.pain_stimulus
        current_fio2 = state.fio2
        current_height = state.height
        current_weight = state.weight
        current_socio_econ = state.socio_econ

        max_sysbp = torch.where(current_gender == 0, max_sysbp_men, max_sysbp_women)
        max_diabp = torch.where(current_gender == 0, max_diabp_men, max_diabp_women)

        current_sysbp = pyro.deterministic(f"z_{t}_sysbp",
                                           current_sysbp + action / 3 * (pyro.sample(f"z_{t}_sysbp_eps", dist.Uniform(0, max_sysbp - current_sysbp))))
        current_diabp = pyro.deterministic(f"z_{t}_diabp",
                                           current_diabp + action / 3 * (pyro.sample(f"z_{t}_diabp_eps", dist.Uniform(0, max_diabp - current_diabp))))
        current_hr = pyro.deterministic(f"z_{t}_hr",
                                        current_hr + action*(pyro.sample(f"z_{t}_hr_eps", dist.Normal(0,1))**2)*(1+current_socio_econ)/3)
        current_rr = pyro.deterministic(f"z_{t}_rr",
                                        current_rr + action*(pyro.sample(f"z_{t}_rr_eps", dist.Normal(0,1))**2)*(1+current_socio_econ)/3)
        current_wbc_count = pyro.deterministic(f"z_{t}_wbc_count",
                                               current_wbc_count * (1-action/3*pyro.sample(f"z_{t}_wbc_count_eps", dist.Normal(0.1,0.001))))
        pain_stimulus_change = pyro.sample(f"z_{t}_pain_stimulus_change", dist.Uniform(0,1))

        pain_stimulus_eps_ul = pyro.deterministic(f"z_{t}_pain_stimulus_eps_ul",
                                                  torch.where(current_pain_stimulus > 0, current_pain_stimulus, torch.ones(1)))

        current_pain_stimulus = torch.where((pain_stimulus_change > (1 - action / 3)) & (current_pain_stimulus > 0),
            pyro.deterministic(f"z_{t}_pain_stimulus", np.floor(pyro.sample(f"z_{t}_pain_stimulus_raw", dist.Uniform(0, pain_stimulus_eps_ul)))),
            current_pain_stimulus
            )
        return PatientState(current_gender, current_hr, current_rr,
                            current_sysbp, current_diabp, current_fio2,
                            current_weight, current_height, current_wbc_count,
                            current_socio_econ, current_pain_stimulus)

    @staticmethod
    def get_xt_from_state(st, t, obs_data):
        gender = pyro.sample(f"x_{t}_gender", dist.Delta(st.gender), obs=obs_data[:, t, cols.index('xt_gender')])
        hr_eps = pyro.sample(f"x_{t}_hr_epsilon", dist.Normal(0, 5))
        hr = pyro.sample(f"x_{t}_hr", dist.Delta(st.hr + hr_eps), obs=obs_data[:, t, cols.index('xt_hr')])
        sysbp_eps = pyro.sample(f"x_{t}_sysbp_eps", dist.Normal(0, 4))
        sysbp = pyro.sample(f"x_{t}_sysbp", dist.Delta(st.sysbp + sysbp_eps), obs=obs_data[:, t, cols.index('xt_sysbp')])
        diabp_eps = pyro.sample(f"x_{t}_diabp_eps", dist.Normal(0, 4))
        diabp = pyro.sample(f"x_{t}_diabp", dist.Delta(st.diabp + diabp_eps), obs=obs_data[:, t, cols.index('xt_diabp')])
        return Xt(gender, hr, sysbp, diabp)

    def model(self, mini_batch):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = InitialStateGenerator()
            s_t = initial_state_generator.generate_state()
            for t in range(T_max):
                SimulatorModel.get_xt_from_state(s_t, t, obs_data=mini_batch)
                action_eps = pyro.sample(f"A_{t}_ind", dist.Multinomial(total_count=1, probs=self.policy(s_t.as_tensor())))
                action = pyro.sample(f"A_{t}", dist.Delta((action_eps==1).nonzero()[:, -1]), obs=mini_batch[:, t, cols.index('A_t')])
                s_t = SimulatorModel.transition_to_next_state(s_t, action, t+1)

    def guide(self, mini_batch):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = InitialStateGenerator()
            s_t = initial_state_generator.generate_state()
            for t in range(T_max):





############################################################################
e = ObservationalDataset("/Users/faaiz/exportdir/observational-data-5000-0.csv")
from torch.utils.data.sampler import SubsetRandomSampler

validation_split = 0.25
shuffle_dataset = True
dataset_size = len(e)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
policy = PolicyNetwork()
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(e, batch_size=16,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(e, batch_size=16,
                                                sampler=valid_sampler)

for batch in validation_loader:
    break
print(batch.size())

with pyro.plate("s_minibatch", len(batch)):
    initial_state_generator = InitialStateGenerator()
    s_t = initial_state_generator.generate_state()
    z  = pyro.sample("A_{1}", dist.Multinomial(1, policy(s_t.as_tensor())))
    a = (z==1).nonzero()[:, -1]
    SimulatorModel.transition_to_next_state(s_t, action=a, t=1)
    # print(s_t.as_tensor().size())
    # print(s_t.as_tensor())
    # p = policy(s_t.as_tensor())
    print(a.size())
    print(a)
    print(z)
    # action = batch[:, 1, cols.index('A_t')]
    # print(action.size())






