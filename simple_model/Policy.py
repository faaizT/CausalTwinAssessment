import torch as T


class Policy(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=5):
        super(Policy, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.outp = T.nn.Linear(hidden_1_dim, output_dim)
        self.tanh = T.nn.Tanh()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.outp.weight)
        T.nn.init.zeros_(self.outp.bias)

    def forward(self, x):
        z = self.tanh(self.hid1(x))
        z = self.outp(z)
        return z


class SNetwork(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=5):
        super(SNetwork, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.hid_to_loc = T.nn.Linear(hidden_1_dim, output_dim)
        self.hid_to_logscale = T.nn.Linear(hidden_1_dim, output_dim)
        self.tanh = T.nn.Tanh()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid_to_loc.weight)
        T.nn.init.zeros_(self.hid_to_loc.bias)
        T.nn.init.xavier_uniform_(self.hid_to_logscale.weight)
        T.nn.init.zeros_(self.hid_to_logscale.bias)

    def forward(self, x):
        z = self.tanh(self.hid1(x))
        z_loc = self.hid_to_loc(z)
        z_logscale = self.hid_to_logscale(z)
        return z_loc, z_logscale