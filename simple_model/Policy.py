import torch
import torch as T


class Policy(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=5, hidden_2_dim=5):
        super(Policy, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.hid2 = T.nn.Linear(hidden_1_dim, hidden_2_dim)
        self.outp = T.nn.Linear(hidden_2_dim, output_dim)
        self.leakyRelu = T.nn.LeakyReLU()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.outp.weight)
        T.nn.init.zeros_(self.outp.bias)

    def forward(self, x):
        z = self.leakyRelu(self.hid1(x))
        z = self.leakyRelu(self.hid2(z))
        z = self.outp(z)
        return z


class SNetwork(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=5):
        super(SNetwork, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.hid_to_loc = T.nn.Linear(hidden_1_dim, output_dim)
        self.hid_to_tril = T.nn.Linear(hidden_1_dim, int(output_dim*(output_dim-1)/2))
        self.hid_to_diag = T.nn.Linear(hidden_1_dim, output_dim)
        self.tanh = T.nn.Tanh()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid_to_loc.weight)
        T.nn.init.zeros_(self.hid_to_loc.bias)
        T.nn.init.xavier_uniform_(self.hid_to_tril.weight)
        T.nn.init.zeros_(self.hid_to_tril.bias)
        T.nn.init.xavier_uniform_(self.hid_to_diag.weight)
        T.nn.init.zeros_(self.hid_to_diag.bias)

    def forward(self, x):
        z = self.tanh(self.hid1(x))
        z_loc = self.hid_to_loc(z)
        z_tril = self.hid_to_tril(z)
        z_diag = torch.exp(self.hid_to_diag(z))
        return z_loc, z_tril, z_diag
