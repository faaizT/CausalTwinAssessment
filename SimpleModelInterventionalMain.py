import argparse
import logging
import os
import re

import pandas as pd
import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from max_likelihood.utils.ObservationalDataset import ObservationalDataset
from torch.utils.data.sampler import SubsetRandomSampler

from simple_model import Policy
from simple_model.GenerateSimpleModelData import S_t, physicians_policy
from simple_model.SimpleModel import SimpleModel
from simple_model.SimpleModelInterventionalDist import SimpleModelInterventionalDist, cols


def evaluate(model: SimpleModelInterventionalDist, test_loader, use_cuda=False):
    loss_fn = Trace_ELBO().differentiable_loss
    # initialize loss accumulator
    losses = pd.DataFrame()
    test_loss = 0.
    # compute the loss over the entire test set
    with torch.no_grad():
        for x in test_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            # compute ELBO estimate and accumulate loss
            idx = x[0, 0, cols.index('id')].item()
            action = x[0, 0, cols.index('A_t')].item()
            loss = loss_fn(model.model, model.guide, x.float()).numpy()
            losses = losses.append({'id': idx, 'loss': loss, 'action': action}, ignore_index=True)
            test_loss += loss
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test, losses


def main(args):
    simulator_model = SimpleModelInterventionalDist(learned_initial_state=args.model_dir, increment_factor=args.increment_factor,
                                  phy_pol=False, rho=args.rho)
    x, y = simulator_model.increment_factor.numpy()
    exportdir = args.exportdir
    log_file_name = f'{exportdir}/model_{x}-{y}.log'
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()])
    logging.info(f"X: {x}. Y: {y}")
    observational_dataset = ObservationalDataset(args.path, columns=cols)
    pyro.clear_param_store()
    validation_split = 0.20
    test_split = 0.20
    shuffle_dataset = True
    dataset_size = len(observational_dataset)
    indices = list(range(dataset_size))
    split_val = int(np.floor(validation_split * dataset_size))
    split_test = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split_val+split_test:], indices[:split_val], indices[split_val:split_val+split_test]
    # Creating PT data samplers and loaders:
    test_sampler = SubsetRandomSampler(test_indices)

    test_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=1, sampler=test_sampler)
    optimizer = torch.optim.Adam(simulator_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(args.beta1, args.beta2))
    epoch_loss_val, losses = evaluate(simulator_model, test_loader, use_cuda=False)
    logging.info("average validation loss: %.4f" % epoch_loss_val)
    losses.to_csv(f"{exportdir}/losses-{x}-{y}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to model dir")
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--model_dir", help="path to model dir")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", help="weight decay (L2 penalty)", type=float, default=2.0)
    parser.add_argument('--beta1', type=float, default=0.96)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument("--rho", help="Correlation coefficient of S_0 components", type=float, default=None)
    parser.add_argument("--increment_factor", nargs='+', help="factor by which simulator increments s_t values",
                        type=int, default=None)
    args = parser.parse_args()
    main(args)
