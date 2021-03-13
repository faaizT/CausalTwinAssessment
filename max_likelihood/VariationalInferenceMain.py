import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from max_likelihood.HelperNetworks import PolicyNetwork
from max_likelihood.ObservationalDataset import ObservationalDataset
from max_likelihood.SimulatorModel import SimulatorModel
from torch.utils.data.sampler import SubsetRandomSampler


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


if __name__ == "__main__":
    e = ObservationalDataset("/Users/faaiz/exportdir/observational-data-500-0.csv")
    pyro.clear_param_store()
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
