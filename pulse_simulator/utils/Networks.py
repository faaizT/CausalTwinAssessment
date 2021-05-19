import torch
import torch as T


class Policy(T.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_1_dim=8, use_cuda=False):
        super(Policy, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.outp = T.nn.Linear(hidden_1_dim, output_dim)
        self.leakyRelu = T.nn.LeakyReLU()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.outp.weight)
        T.nn.init.zeros_(self.outp.bias)
        if use_cuda:
            self.cuda()

    def forward(self, x):
        z = self.leakyRelu(self.hid1(x))
        z = self.outp(z)
        return z


class Net(T.nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim1=8, hidden_dim2=8, use_cuda=False
    ):
        super().__init__()
        self.lin1 = T.nn.Linear(input_dim, hidden_dim1)
        self.lin2 = T.nn.Linear(hidden_dim1, hidden_dim2)
        self.lin2_to_loc = T.nn.Linear(hidden_dim2, output_dim)
        self.lin2_to_scale = T.nn.Linear(hidden_dim2, output_dim)
        self.leakyRelu = T.nn.LeakyReLU()
        if use_cuda:
            self.cuda()

    def forward(self, z):
        z = self.leakyRelu(self.lin1(z))
        z = self.leakyRelu(self.lin2(z))
        loc = self.lin2_to_loc(z)
        scale = self.lin2_to_scale(z)
        return loc, scale
