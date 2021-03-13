import pyro
import pyro.distributions as dist
import torch
from torch import nn

from max_likelihood.HelperNetworks import PolicyNetwork, Combiner, PainStimulusClassifier
from max_likelihood.ObservationalDataset import cols
from max_likelihood.S_0 import S_0
from observational_model.PatientState import PatientState
# TODO: move to appropriate location
# TODO: deal with tensors in St, Xt and Ut
# TODO: Comments
from observational_model.PatientState import Xt

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

    def __init__(self, input_dim=len(cols), rnn_dim=20, st_dim=11, actions=4, rnn_dropout_rate=0.0, use_cuda=False):
        super().__init__()
        self.policy = PolicyNetwork(input_dim=st_dim, output_dim=actions)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=1, dropout=rnn_dropout_rate)
        self.pain_stimulus_classifier = PainStimulusClassifier(st_dim, rnn_dim, pain_stimulus_dim=10)
        self.combiner = Combiner(z_dim=st_dim, rnn_dim=rnn_dim, out_dim=st_dim-3)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    @staticmethod
    def transition_to_next_state(state: PatientState, action, t):
        current_hr = pyro.sample(f"s_{t}_hr", dist.Normal(action*8, action**2*4+0.1))
        current_rr = pyro.sample(f"s_{t}_rr", dist.Normal(action*2, action**2*1+0.1))
        current_sysbp = pyro.sample(f"s_{t}_sysbp", dist.Normal(action*5*(1-state.gender) + state.gender*action*3, action*5 + 0.1))
        current_diabp = pyro.sample(f"s_{t}_diabp", dist.Normal(action*5*(1-state.gender) + state.gender*action*3, action*5 + 0.1))
        current_height = pyro.sample(f"s_{t}_height", dist.Normal(state.height, 0.001))
        current_weight = pyro.sample(f"s_{t}_weight", dist.Normal(state.weight, 0.001))
        current_wbc_count = pyro.sample(f"s_{t}_wbc_count", dist.Normal(state.wbc_count - action/3*1000, ((action+1)/4)**2*1000))
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

    @staticmethod
    def get_xt_from_state(st, t, obs_data):
        hr = pyro.sample(f"x_{t}_hr", dist.Normal(st.hr, 5), obs=obs_data[:, t, cols.index('xt_hr')])
        sysbp = pyro.sample(f"x_{t}_sysbp", dist.Normal(st.sysbp, 4), obs=obs_data[:, t, cols.index('xt_sysbp')])
        diabp = pyro.sample(f"x_{t}_diabp", dist.Normal(st.diabp, 4), obs=obs_data[:, t, cols.index('xt_diabp')])
        return Xt(gender=st.gender, hr=hr, sysbp=sysbp, diabp=diabp)

    def model(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = S_0(mini_batch)
            s_t = initial_state_generator.generate_state()
            for t in range(T_max):
                SimulatorModel.get_xt_from_state(s_t, t, obs_data=mini_batch)
                action = pyro.sample(f"A_{t}", dist.Categorical(self.policy(s_t.as_tensor())[0]), obs=mini_batch[:, t, cols.index('A_t')])
                if t < T_max - 1:
                    s_t = SimulatorModel.transition_to_next_state(s_t, action, t + 1)

    def guide(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = S_0()
            s_t = initial_state_generator.generate_state()
            h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
            rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
            rnn_output = torch.flip(rnn_output, [1])
            for t in range(1,T_max):
                pain_stimulus = pyro.sample(f"s_{t}_pain_stimulus", dist.Categorical(self.pain_stimulus_classifier(s_t.as_tensor(), rnn_output[:, t-1, :])[0]))
                s_loc, s_scale = self.combiner(s_t.as_tensor(), rnn_output[:, t-1, :])
                hr = pyro.sample(f"s_{t}_hr", dist.Normal((s_loc[:, :, 0]).reshape(-1), (s_scale[:, :, 0]).reshape(-1)))
                rr = pyro.sample(f"s_{t}_rr", dist.Normal((s_loc[:, :, 1]).reshape(-1), (s_scale[:, :, 1]).reshape(-1)))
                sysbp = pyro.sample(f"s_{t}_sysbp", dist.Normal((s_loc[:, :, 2]).reshape(-1), (s_scale[:, :, 2]).reshape(-1)))
                diabp = pyro.sample(f"s_{t}_diabp", dist.Normal((s_loc[:, :, 3]).reshape(-1), (s_scale[:, :, 3]).reshape(-1)))
                fio2 = pyro.sample(f"s_{t}_fio2", dist.Normal((s_loc[:, :, 4]).reshape(-1), (s_scale[:, :, 4]).reshape(-1)))
                weight = pyro.sample(f"s_{t}_weight", dist.Normal((s_loc[:, :, 5]).reshape(-1), (s_scale[:, :, 5]).reshape(-1)))
                height = pyro.sample(f"s_{t}_height", dist.Normal((s_loc[:, :, 6]).reshape(-1), (s_scale[:, :, 6]).reshape(-1)))
                wbc_count = pyro.sample(f"s_{t}_wbc_count", dist.Normal((s_loc[:, :, 7]).reshape(-1), (s_scale[:, :, 7]).reshape(-1)))
                s_t = PatientState(gender=s_t.gender, hr=hr, rr=rr,
                                   sysbp=sysbp, diabp=diabp, weight=weight,
                                   height=height, wbc_count=wbc_count, socio_econ=s_t.socio_econ,
                                   pain_stimulus=pain_stimulus, fio2=fio2)