import pyro.distributions as dist
import torch
from torch.distributions import constraints
from pyro.distributions.torch_distribution import (
    TorchDistributionMixin,
)


class CategoricalVals(dist.TorchDistribution):
    arg_constraints = {"probs": constraints.simplex}

    def __init__(self, vals, probs):
        self.probs = probs
        self.vals = vals
        self.categorical = dist.Categorical(probs)
        super(CategoricalVals, self).__init__(
            self.categorical.batch_shape, self.categorical.event_shape
        )

    def sample(self, sample_shape=torch.Size()):
        return self.vals[self.categorical.sample(sample_shape)]

    def log_prob(self, value):
        idx = (self.vals == value).nonzero()
        return self.categorical.log_prob(idx)