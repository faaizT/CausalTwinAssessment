import torch
import torch as T
import pyro.distributions as dist


class Policy(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=8, hidden_2_dim=8, use_cuda=False):
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
        if use_cuda:
            self.cuda()

    def forward(self, x):
        z = self.leakyRelu(self.hid1(x))
        z = self.leakyRelu(self.hid2(z))
        z = self.outp(z)
        return z


def physicians_policy(st):
    xt = dist.Normal(st[:, 0], 1).sample()
    ut = dist.Normal(st[:, 1], 1).sample()
    prob_action = torch.where((xt <= 90) & (ut <= 150), torch.minimum(240-xt-ut, torch.tensor(240).float()).float()/240, torch.tensor(0).float()).float()
    return torch.column_stack((torch.zeros(st.size(0)).float(), torch.logit(prob_action, eps=1e-6)))


class SNetwork(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=8, hidden_2_dim=8):
        super(SNetwork, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.hid2 = T.nn.Linear(hidden_1_dim, hidden_2_dim)
        self.hid_to_loc = T.nn.Linear(hidden_2_dim, output_dim)
        self.hid_to_tril = T.nn.Linear(hidden_2_dim, int(output_dim*(output_dim-1)/2))
        self.hid_to_diag = T.nn.Linear(hidden_2_dim, output_dim)
        self.leakyRelu = T.nn.LeakyReLU()

    def forward(self, x):
        z = self.leakyRelu(self.hid1(x))
        z = self.leakyRelu(self.hid2(z))
        z_loc = self.hid_to_loc(z)
        z_tril = self.hid_to_tril(z)
        z_diag = torch.square(self.hid_to_diag(z)) + 0.01
        return z_loc, z_tril, z_diag
