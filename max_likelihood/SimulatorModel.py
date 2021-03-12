import pyro
import numpy as np
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch import nn
from max_likelihood.ObservationalDataset import cols, ObservationalDataset
from observational_model.PatientState import PatientState
import pyro.distributions as dist
from observational_model.Model import InitialStateGenerator
import pyro.contrib.examples.polyphonic_data_loader as poly

# TODO: move to appropriate location
# TODO: deal with tensors in St, Xt and Ut
# TODO: Comments
from observational_model.PatientState import Xt
from max_likelihood.HelperNetworks import PolicyNetwork, Combiner, PainStimulusClassifier

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

    def __init__(self, input_dim=5, rnn_dim=20, st_dim=11, actions=4, rnn_dropout_rate=0.0, use_cuda=False):
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
        current_gender = pyro.deterministic(f"s_{t}_gender", state.gender)
        current_fio2 = pyro.deterministic(f"s_{t}_fio2", state.fio2)
        current_socio_econ = pyro.deterministic(f"s_{t}_socio_econ", state.socio_econ)
        current_height = pyro.deterministic(f"s_{t}_height", state.height)
        current_weight = pyro.deterministic(f"s_{t}_weight", state.weight)
        max_sysbp = torch.where(current_gender == 0, max_sysbp_men, max_sysbp_women)
        max_diabp = torch.where(current_gender == 0, max_diabp_men, max_diabp_women)

        current_sysbp = pyro.deterministic(f"s_{t}_sysbp",
                                           state.sysbp + action / 3 * (pyro.sample(f"s_{t}_sysbp_eps", dist.Uniform(0, max_sysbp - state.sysbp))))
        current_diabp = pyro.deterministic(f"s_{t}_diabp",
                                           state.diabp + action / 3 * (pyro.sample(f"s_{t}_diabp_eps", dist.Uniform(0, max_diabp - state.diabp))))
        current_hr = pyro.deterministic(f"s_{t}_hr",
                                        state.hr + action*(pyro.sample(f"s_{t}_hr_eps", dist.Normal(0,1))**2)*(1+current_socio_econ)/3)
        current_rr = pyro.deterministic(f"s_{t}_rr",
                                        state.rr + action*(pyro.sample(f"s_{t}_rr_eps", dist.Normal(0,1))**2)*(1+current_socio_econ)/3)
        current_wbc_count = pyro.deterministic(f"s_{t}_wbc_count",
                                               state.wbc_count * (1-action/3*pyro.sample(f"s_{t}_wbc_count_eps", dist.Normal(0.1,0.001))))
        pain_stimulus_change = pyro.sample(f"s_{t}_pain_stimulus_change", dist.Uniform(0,1))

        pain_stimulus_eps_ul = pyro.deterministic(f"s_{t}_pain_stimulus_eps_ul",
                                                  torch.where(state.pain_stimulus > 0, state.pain_stimulus, torch.ones(1)))

        current_pain_stimulus = torch.where((pain_stimulus_change > (1 - action / 3)) & (state.pain_stimulus > 0),
            pyro.deterministic(f"s_{t}_pain_stimulus", np.floor(pyro.sample(f"s_{t}_pain_stimulus_raw", dist.Uniform(0, pain_stimulus_eps_ul)))),
            state.pain_stimulus
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

    def model(self, mini_batch, mini_batch_reversed):
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

    def guide(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("simulator_model", self)
        with pyro.plate("s_minibatch", len(mini_batch)):
            initial_state_generator = InitialStateGenerator()
            s_t = initial_state_generator.generate_state()
            h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
            rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
            rnn_output = torch.flip(rnn_output, [1])
            for t in range(T_max):
                gender = pyro.sample(f"s_{t+1}_gender", dist.Delta(s_t.gender))
                socio_econ = pyro.sample(f"s_{t+1}_socio_econ", dist.Delta(s_t.socio_econ))
                pain_stimulus_probs = self.pain_stimulus_classifier(s_t.as_tensor(), rnn_output[:, t, :])
                pain_stimulus_eps = pyro.sample(f"s_{t+1}_pain_stimulus_eps", dist.Multinomial(total_count=1, probs=pain_stimulus_probs))
                pain_stimulus = pyro.sample(f"s_{t+1}_pain_stimulus", dist.Delta((pain_stimulus_eps==1).nonzero()[:, -1]))
                s_loc, s_scale = self.combiner(s_t.as_tensor(), rnn_output[:, t, :])
                hr = pyro.sample(f"s_{t+1}_hr", dist.Normal((s_loc[:, :, 0]).reshape(-1), (s_scale[:, :, 0]).reshape(-1)))
                rr = pyro.sample(f"s_{t+1}_rr", dist.Normal((s_loc[:, :, 1]).reshape(-1), (s_scale[:, :, 1]).reshape(-1)))
                sysbp = pyro.sample(f"s_{t+1}_sysbp", dist.Normal((s_loc[:, :, 2]).reshape(-1), (s_scale[:, :, 2]).reshape(-1)))
                diabp = pyro.sample(f"s_{t+1}_diabp", dist.Normal((s_loc[:, :, 3]).reshape(-1), (s_scale[:, :, 3]).reshape(-1)))
                fio2 = pyro.sample(f"s_{t+1}_fio2", dist.Normal((s_loc[:, :, 4]).reshape(-1), (s_scale[:, :, 4]).reshape(-1)))
                weight = pyro.sample(f"s_{t+1}_weight", dist.Normal((s_loc[:, :, 5]).reshape(-1), (s_scale[:, :, 5]).reshape(-1)))
                height = pyro.sample(f"s_{t+1}_height", dist.Normal((s_loc[:, :, 6]).reshape(-1), (s_scale[:, :, 6]).reshape(-1)))
                wbc_count = pyro.sample(f"s_{t+1}_wbc_count", dist.Normal((s_loc[:, :, 7]).reshape(-1), (s_scale[:, :, 7]).reshape(-1)))
                s_t = PatientState(gender, hr, rr, sysbp, diabp, fio2, weight, height, wbc_count, socio_econ, pain_stimulus)


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
policy = PolicyNetwork(11, 4)
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(e, batch_size=16,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(e, batch_size=16,
                                                sampler=valid_sampler)
adam_params = {"lr": 0.001}
optimizer = ClippedAdam(adam_params)
dmm = SimulatorModel()
svi = SVI(dmm.model, dmm.guide, optimizer, Trace_ELBO())
def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        x_reversed = torch.flip(x, [1])
        epoch_loss += svi.step(x.float(), x_reversed.float())

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        x_reversed = torch.flip(x, [1])
        test_loss += svi.evaluate_loss(x.float(), x_reversed.float())
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


NUM_EPOCHS = 8
train_elbo = []
test_elbo = []
TEST_FREQUENCY = 2
# training loop
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader, use_cuda=False)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, validation_loader, use_cuda=False)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

