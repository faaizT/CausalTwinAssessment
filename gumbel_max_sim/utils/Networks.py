import torch
import torch as T


class Combiner(T.nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim,
                hr_dim=3, sysbp_dim=3, percoxyg_dim=2,
                glucose_dim=5, antibiotics_dim=2, vaso_dim=2,
                vent_dim=2, use_cuda=False):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = T.nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_hr = T.nn.Linear(rnn_dim, hr_dim)
        self.lin_hidden_to_sysbp = T.nn.Linear(rnn_dim, sysbp_dim)
        self.lin_hidden_to_percoxyg = T.nn.Linear(rnn_dim, percoxyg_dim)
        self.lin_hidden_to_glucose = T.nn.Linear(rnn_dim, glucose_dim)
        self.lin_hidden_to_antibiotic = T.nn.Linear(rnn_dim, antibiotics_dim)
        self.lin_hidden_to_vaso = T.nn.Linear(rnn_dim, vaso_dim)
        self.lin_hidden_to_vent = T.nn.Linear(rnn_dim, vent_dim)
        # initialize the two non-linearities used in the neural network
        self.leakyRelu = T.nn.LeakyReLU()
        if use_cuda:
            self.cuda()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, x_{t:T})
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.leakyRelu(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        hr_logits = self.lin_hidden_to_hr(h_combined)
        sysbp_logits = self.lin_hidden_to_sysbp(h_combined)
        percoxyg_logits = self.lin_hidden_to_percoxyg(h_combined)
        glucose_logits = self.lin_hidden_to_glucose(h_combined)
        antibiotic_logits = self.lin_hidden_to_antibiotic(h_combined)
        vaso_logits = self.lin_hidden_to_vaso(h_combined)
        vent_logits = self.lin_hidden_to_vent(h_combined)
        return hr_logits, sysbp_logits, percoxyg_logits, glucose_logits, antibiotic_logits, vaso_logits, vent_logits


class Combiner_without_rnn(T.nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """

    def __init__(self, z_dim, hidden_dim, out_dim, use_cuda=False):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = T.nn.Linear(z_dim, hidden_dim)
        self.lin_hidden_to_logits = T.nn.Linear(hidden_dim, out_dim)
        # initialize the two non-linearities used in the neural network
        self.leakyRelu = T.nn.LeakyReLU()
        if use_cuda:
            self.cuda()

    def forward(self, z_t_1):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, x_{t:T})
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = self.leakyRelu(self.lin_z_to_hidden(z_t_1))
        # use the combined hidden state to compute the mean used to sample z_t
        output = self.lin_hidden_to_logits(h_combined)
        return output

class Net(T.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=8, use_cuda=False):
        super().__init__()
        self.lin1 = T.nn.Linear(input_dim, hidden_dim)
        self.lin2 = T.nn.Linear(hidden_dim, output_dim)
        self.leakyRelu = T.nn.LeakyReLU()
        if use_cuda:
            self.cuda()

    def forward(self, z):
        z = self.leakyRelu(self.lin1(z))
        output = self.lin2(z)
        return output
