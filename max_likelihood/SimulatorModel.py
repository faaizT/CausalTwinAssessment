import pyro
import pyro.distributions as dist
import torch
from torch import nn

from max_likelihood.utils.HelperNetworks import PolicyNetwork, Combiner, PainStimulusClassifier, CombinerWithNoise
from max_likelihood.utils.ObservationalDataset import cols
from max_likelihood.utils.S_0 import S_0
from max_likelihood.utils.TransitionModel import transition_to_next_state, get_xt_from_state
from observational_model.PatientState import PatientState
# TODO: move to appropriate location
# TODO: deal with tensors in St, Xt and Ut
# TODO: Comments
# TODO: IMPORTANT - correct the actions being passed into rnn

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

    def __init__(self, input_dim=len(cols), rnn_dim=30, st_dim=11, actions=4, rnn_dropout_rate=0.0, noise_dim=10,
                 transition_model=transition_to_next_state, use_cuda=False):
        super().__init__()
        self.policy = PolicyNetwork(input_dim=st_dim, output_dim=actions)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=1, dropout=rnn_dropout_rate)
        self.pain_stimulus_classifier = PainStimulusClassifier(st_dim, rnn_dim, pain_stimulus_dim=10)
        self.combiner = CombinerWithNoise(z_dim=st_dim, rnn_dim=rnn_dim, out_dim=st_dim - 3, noise_dim=noise_dim)
        self.noise_dim = noise_dim
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))
        self.use_cuda = use_cuda
        self.transition_model = transition_model
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = S_0(mini_batch)
            s_t = initial_state_generator.generate_state()
            for t in range(T_max):
                get_xt_from_state(s_t, t, obs_data=mini_batch)
                action = pyro.sample(f"A_{t}", dist.Categorical(self.policy(s_t.as_tensor())[0]), obs=mini_batch[:, t, cols.index('A_t')])
                if t < T_max - 1:
                    s_t = self.transition_model(s_t, action, t + 1)

    def guide(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = S_0()
            s_t = initial_state_generator.generate_state()
            h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
            rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
            rnn_output = torch.flip(rnn_output, [1])
            for t in range(1, T_max):
                epsilon = dist.Normal(0, 1).sample((1, len(mini_batch), self.noise_dim))
                s_loc = self.combiner(s_t.as_tensor(), rnn_output[:, t-1, :], epsilon)
                hr = pyro.sample(f"s_{t}_hr", dist.Normal((s_loc[:, :, 0]).reshape(-1), 0.001))
                rr = pyro.sample(f"s_{t}_rr", dist.Normal((s_loc[:, :, 1]).reshape(-1), 0.001))
                sysbp = pyro.sample(f"s_{t}_sysbp", dist.Normal((s_loc[:, :, 2]).reshape(-1), 0.001))
                diabp = pyro.sample(f"s_{t}_diabp", dist.Normal((s_loc[:, :, 3]).reshape(-1), 0.001))
                fio2 = pyro.sample(f"s_{t}_fio2", dist.Normal((s_loc[:, :, 4]).reshape(-1), 0.001))
                weight = pyro.sample(f"s_{t}_weight", dist.Normal((s_loc[:, :, 5]).reshape(-1), 0.001))
                height = pyro.sample(f"s_{t}_height", dist.Normal((s_loc[:, :, 6]).reshape(-1), 0.001))
                wbc_count = pyro.sample(f"s_{t}_wbc_count", dist.Normal((s_loc[:, :, 7]).reshape(-1), 0.001))
                pain_stimulus = pyro.sample(f"s_{t}_pain_stimulus", dist.Categorical(self.pain_stimulus_classifier(s_t.as_tensor(), rnn_output[:, t-1, :])[0]))
                s_t = PatientState(gender=s_t.gender, hr=hr, rr=rr,
                                   sysbp=sysbp, diabp=diabp, weight=weight,
                                   height=height, wbc_count=wbc_count, socio_econ=s_t.socio_econ,
                                   pain_stimulus=pain_stimulus, fio2=fio2)