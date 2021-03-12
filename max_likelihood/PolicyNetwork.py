import torch as T


class PolicyNetwork(T.nn.Module):
    def __init__(self, input_dim=11, hidden_1_dim=10, hidden_2_dim=10, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.hid1 = T.nn.Linear(input_dim, hidden_1_dim)
        self.hid2 = T.nn.Linear(hidden_1_dim, hidden_2_dim)
        self.oupt = T.nn.Linear(hidden_2_dim, output_dim)
        self.softmax = T.nn.Softmax(dim=2)
        self.leaky_relu = T.nn.LeakyReLU()

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = self.leaky_relu(self.hid1(x))
        z = self.leaky_relu(self.hid2(z))
        z = self.softmax(self.oupt(z))
        return z
