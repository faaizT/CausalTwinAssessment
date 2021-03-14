import torch
from pyroapi import pyro

from max_likelihood.utils.ObservationalDataset import cols
from observational_model.PatientState import PatientState, Xt
import pyro.distributions as dist


def transition_to_next_state(state: PatientState, action, t):
    current_hr = pyro.sample(f"s_{t}_hr",
                             dist.Normal(action * 8 + state.hr, action ** 2 * 4 + 0.1).rv.mul(
                                 (1 + state.socio_econ) / 3).dist)
    current_rr = pyro.sample(f"s_{t}_rr",
                             dist.Normal(action * 2 + state.rr, action ** 2 * 1 + 0.1).rv.mul(
                                 (1 + state.socio_econ) / 3).dist)
    current_sysbp = pyro.sample(f"s_{t}_sysbp",
                                dist.Normal(action * 5 * (1 - state.gender) + state.gender * action * 3 + state.sysbp,
                                            action * 5 + 0.1))
    current_diabp = pyro.sample(f"s_{t}_diabp",
                                dist.Normal(action * 5 * (1 - state.gender) + state.gender * action * 3 + state.diabp,
                                            action * 5 + 0.1))
    current_height = pyro.sample(f"s_{t}_height", dist.Normal(state.height, 0.001))
    current_weight = pyro.sample(f"s_{t}_weight", dist.Normal(state.weight, 0.001))
    current_wbc_count = pyro.sample(f"s_{t}_wbc_count",
                                    dist.Normal(state.wbc_count - action / 3 * 1000, ((action + 1) / 4) ** 2 * 1000))
    current_fio2 = pyro.sample(f"s_{t}_fio2", dist.Normal(state.fio2, torch.tensor(0.001)))
    probs = torch.zeros((state.pain_stimulus.size(0), 10))
    probs[torch.arange(state.pain_stimulus.size(0)), state.pain_stimulus] = 1
    pain_stimulus_non_zero = torch.where(state.pain_stimulus > 0, 1, 0)
    probs[torch.arange(state.pain_stimulus.size(0)), state.pain_stimulus - pain_stimulus_non_zero] = 1
    current_pain_stimulus = pyro.sample(f"s_{t}_pain_stimulus", dist.Categorical(probs=probs))
    return PatientState(gender=state.gender, hr=current_hr, rr=current_rr,
                        sysbp=current_sysbp, diabp=current_diabp, height=current_height,
                        weight=current_weight, wbc_count=current_wbc_count, socio_econ=state.socio_econ,
                        pain_stimulus=current_pain_stimulus, fio2=current_fio2)


def get_xt_from_state(st, t, obs_data):
    hr = pyro.sample(f"x_{t}_hr", dist.Normal(st.hr, 5), obs=obs_data[:, t, cols.index('xt_hr')])
    sysbp = pyro.sample(f"x_{t}_sysbp", dist.Normal(st.sysbp, 4), obs=obs_data[:, t, cols.index('xt_sysbp')])
    diabp = pyro.sample(f"x_{t}_diabp", dist.Normal(st.diabp, 4), obs=obs_data[:, t, cols.index('xt_diabp')])
    return Xt(gender=st.gender, hr=hr, sysbp=sysbp, diabp=diabp)