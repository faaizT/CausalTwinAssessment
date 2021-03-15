import argparse
import datetime
import logging
import ntpath
import pandas as pd
import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from max_likelihood.utils.HelperNetworks import PolicyNetwork
from max_likelihood.utils.ObservationalDataset import ObservationalDataset
from max_likelihood.SimulatorModel import SimulatorModel
from torch.utils.data.sampler import SubsetRandomSampler


def train(svi, train_loader, epoch, loss_dict, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    i = 0
    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        x_reversed = torch.flip(x, [1])
        epoch_loss += svi.step(x.float(), x_reversed.float())
        if i % 50 == 0:
            logging.info("[epoch %03d] [minibatch %06d]  epoch_loss: %.4f" % (epoch, i, epoch_loss))
            loss_dict['Epochs'].append(epoch)
            loss_dict['Mini-batch'].append(i)
            loss_dict['Training Loss'].append(epoch_loss)
            pd.DataFrame(data=loss_dict).to_csv(args.exportdir + f"/train-loss-minibatches-{time}.csv")
        i += 1
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
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to observational data")
    parser.add_argument("epochs", help="number of epochs to train for", type=int)
    parser.add_argument("exportdir", help="path to output directory")
    args = parser.parse_args()

    time = datetime.datetime.now().time()
    obs_data_file_name = ntpath.basename(args.path)
    log_file_name = f'{args.exportdir}/model_{time}.log'
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()]
                        )
    observational_dataset = ObservationalDataset(args.path)
    pyro.clear_param_store()
    validation_split = 0.25
    shuffle_dataset = True
    dataset_size = len(observational_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    policy = PolicyNetwork(11, 4)
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16,
                                                    sampler=valid_sampler)
    adam_params = {"lr": 0.001}
    optimizer = ClippedAdam(adam_params)
    simulator_model = SimulatorModel()
    svi = SVI(simulator_model.model, simulator_model.guide, optimizer, Trace_ELBO())

    NUM_EPOCHS = args.epochs
    train_elbo = {'Epochs': [], 'Training Loss': []}
    train_minibatch_elbo = {'Epochs': [], 'Mini-batch': [], 'Training Loss': []}
    test_elbo = {'Epochs': [], 'Test Loss': []}
    TEST_FREQUENCY = 10
    SAVE_FREQUENCY = 10
    # training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, epoch, train_minibatch_elbo, use_cuda=False)
        train_elbo['Epochs'].append(epoch)
        train_elbo['Training Loss'].append(-total_epoch_loss_train)
        logging.info("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        if epoch % TEST_FREQUENCY == 0:
            # report test diagnostics
            total_epoch_loss_test = evaluate(svi, validation_loader, use_cuda=False)
            test_elbo['Epochs'].append(epoch)
            test_elbo['Test Loss'].append(-total_epoch_loss_test)
            logging.info("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
        if epoch > 0 and epoch % SAVE_FREQUENCY == 0:
            logging.info("saving model and optimiser states to %s..." % args.exportdir)
            pd.DataFrame(data=train_elbo).to_csv(args.exportdir+f"/train-loss-{time}.csv")
            pd.DataFrame(data=test_elbo).to_csv(args.exportdir+f"/test-loss-{time}.csv")
            torch.save(simulator_model.state_dict(), args.exportdir+f"/model-state-{time}")
            optimizer.save(args.exportdir+f"/optimiser-state-{time}")
            logging.info("done saving model and optimizer checkpoints to disk.")