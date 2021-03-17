import argparse
import logging
import os
import threading

import pandas as pd
import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from max_likelihood.utils.ObservationalDataset import ObservationalDataset
from torch.utils.data.sampler import SubsetRandomSampler

from simple_model.SimpleModel import cols, SimpleModel


def write_to_file(file_name, x, y, loss):
  with open(file_name, 'a', 1) as f:
      f.write(str(x) + ',' + str(y) + ',' + str(loss) + os.linesep)


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
        epoch_loss += svi.step(x.float())
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
        test_loss += svi.evaluate_loss(x.float())
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def main(path, epochs, exportdir, lr, increment_factor, output_file):
    if increment_factor is not None:
        simulator_model = SimpleModel(increment_factor=tuple(increment_factor))
    else:
        simulator_model = SimpleModel()
    log_file_name = f'{exportdir}/model_{simulator_model.increment_factor}.log'
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()])
    observational_dataset = ObservationalDataset(path, columns=cols)
    pyro.clear_param_store()
    validation_split = 0.25
    shuffle_dataset = True
    dataset_size = len(observational_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16,
                                                    sampler=valid_sampler)
    adam_params = {"lr": lr}
    optimizer = ClippedAdam(adam_params)
    svi = SVI(simulator_model.model, simulator_model.guide, optimizer, Trace_ELBO())

    NUM_EPOCHS = epochs
    train_elbo = {'Epochs': [], 'Training Loss': []}
    test_elbo = {'Epochs': [], 'Test Loss': []}
    TEST_FREQUENCY = 10
    SAVE_FREQUENCY = 2
    # training loop
    consecutive_loss_increments = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss_train = train(svi, train_loader, use_cuda=False)
        train_elbo['Epochs'].append(epoch)
        train_elbo['Training Loss'].append(-epoch_loss_train)
        logging.info("[epoch %03d]  average training loss: %.4f" % (epoch, epoch_loss_train))
        if (epoch+1) % TEST_FREQUENCY == 0:
            # report test diagnostics
            epoch_loss_test = evaluate(svi, validation_loader, use_cuda=False)
            test_elbo['Epochs'].append(epoch)
            test_elbo['Test Loss'].append(-epoch_loss_test)
            logging.info("[epoch %03d] average test loss: %.4f" % (epoch, epoch_loss_test))
            if len(test_elbo['Test Loss']) > 1 and test_elbo['Test Loss'][-1] > test_elbo['Test Loss'][-2]:
                consecutive_loss_increments += 1
            else:
                consecutive_loss_increments = 0
        if (epoch+1) % SAVE_FREQUENCY == 0:
            logging.info("saving model and optimiser states to %s..." % exportdir)
            pd.DataFrame(data=train_elbo).to_csv(exportdir+f"/train-loss-{simulator_model.increment_factor}.csv")
            pd.DataFrame(data=test_elbo).to_csv(exportdir+f"/test-loss-{simulator_model.increment_factor}.csv")
            torch.save(simulator_model.state_dict(), exportdir+f"/model-state-{simulator_model.increment_factor}")
            optimizer.save(exportdir+f"/optimiser-state-{simulator_model.increment_factor}")
            logging.info("done saving model and optimizer checkpoints to disk.")
        if consecutive_loss_increments >= 3:
            break
    logging.info("saving model and optimiser states to %s..." % exportdir)
    pd.DataFrame(data=train_elbo).to_csv(exportdir + f"/train-loss-{simulator_model.increment_factor}.csv")
    pd.DataFrame(data=test_elbo).to_csv(exportdir + f"/test-loss-{simulator_model.increment_factor}.csv")
    torch.save(simulator_model.state_dict(), exportdir + f"/model-state-{simulator_model.increment_factor}")
    optimizer.save(exportdir + f"/optimiser-state-{simulator_model.increment_factor}")
    logging.info("done saving model and optimizer checkpoints to disk.")
    write_to_file(output_file, simulator_model.increment_factor[0], simulator_model.increment_factor[1],
                  test_elbo['Test Loss'][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to observational data")
    parser.add_argument("epochs", help="maximum number of epochs to train for", type=int)
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("output_file", help="Output file to contain final test loss results")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--increment_factor", nargs='+', help="factor by which simulator increments s_t values",
                        type=int, default=None)
    args = parser.parse_args()
    if not os.path.exists(args.output_file):
        with open(args.output_file, "w") as f:
            f.write('x,y,test loss' + os.linesep)
    main(args.path, args.epochs, args.exportdir, args.lr, args.increment_factor, args.output_file)