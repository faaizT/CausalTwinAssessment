import torch
import torch as T


class PolicyNetwork(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=25, hidden_2_dim=25):
        super(PolicyNetwork, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.hid2 = T.nn.Linear(hidden_1_dim, hidden_2_dim)
        self.outp = T.nn.Linear(hidden_2_dim, output_dim)
        self.softmax = T.nn.Softmax(dim=2)
        self.leaky_relu = T.nn.LeakyReLU()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.outp.weight)
        T.nn.init.zeros_(self.outp.bias)

    def forward(self, x):
        z = self.leaky_relu(self.hid1(x))
        z = self.leaky_relu(self.hid2(z))
        z = self.softmax(self.outp(z))
        return z


class Combiner(T.nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """
    def __init__(self, z_dim, rnn_dim, out_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = T.nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = T.nn.Linear(rnn_dim, out_dim)
        self.lin_hidden_to_scale = T.nn.Linear(rnn_dim, out_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = T.nn.Tanh()

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, x_{t:T})
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.lin_hidden_to_scale(h_combined)
        # return loc, scale which can be fed into Normal
        return loc, scale


class CombinerWithNoise(T.nn.Module):

    def __init__(self, noise_dim, z_dim, rnn_dim, out_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = T.nn.Linear(z_dim + noise_dim, rnn_dim)
        self.lin_hidden_to_loc = T.nn.Linear(rnn_dim, out_dim)

        # initialize the two non-linearities used in the neural network
        self.tanh = T.nn.Tanh()

    def forward(self, z_t_1, h_rnn, noise):
        # combine the rnn hidden state with a transformed version of z_t_1 and noise
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(torch.cat((z_t_1, noise), dim=2))) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        return loc


class PainStimulusClassifier(T.nn.Module):
    """
    Parameterizes q(z_t_pain_stimulus | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN (see the pytorch module `rnn` below)
    """
    def __init__(self, z_dim, rnn_dim, pain_stimulus_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = T.nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_pain_stimulus = T.nn.Linear(rnn_dim, pain_stimulus_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = T.nn.Tanh()
        self.softmax = T.nn.Softmax(dim=2)

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN h(x_{t:T}) we return the probabilities of pain_stimulus
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        probabilities = self.softmax(self.lin_hidden_to_pain_stimulus(h_combined))
        # return loc, scale which can be fed into Normal
        return probabilities
