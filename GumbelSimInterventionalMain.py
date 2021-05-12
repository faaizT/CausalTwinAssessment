import re
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import pyro
from pyroapi import distributions as dist
from pyroapi import handlers, infer, optim, pyro, pyro_backend
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.util import ignore_jit_warnings
import pyro.contrib.examples.polyphonic_data_loader as poly
from gumbel_max_sim.GumbelMaxInterventionalDist import GumbelMaxInterventionalDist
from gumbel_max_sim.utils.ObservationalDataset import ObservationalDataset
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from pyro.optim import ClippedAdam
from pyro import poutine

cols = [
    "hr_state",
    "sysbp_state",
    "percoxyg_state",
    "antibiotic_state",
    "vaso_state",
    "vent_state",
    "id"
]


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    losses = pd.DataFrame()
    test_loss = 0.0
    # compute the loss over the entire test set
    for mini_batch, actions, lengths in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            mini_batch = mini_batch.cuda()
            actions = actions.cuda()
            lengths = lengths.cuda()
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(torch.arange(mini_batch.size(0)), mini_batch, lengths, cuda=use_cuda)
        # compute ELBO estimate and accumulate loss
        idx = mini_batch[0, 0, cols.index("id")].item()
        action = actions[0, 0, 0].item()
        loss = svi.evaluate_loss(
            mini_batch.float(), actions.float(), mini_batch_mask.float(), mini_batch_seq_lengths, mini_batch_reversed.float()
        )
        test_loss += loss
        losses = losses.append({'id': idx, 'loss': loss, 'action': action}, ignore_index=True)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test, losses


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gumbel_model = GumbelMaxInterventionalDist(model_path=args.model_dir, simulator_name=args.simulator, use_cuda=use_cuda)
    gumbel_model.to(device)
    exportdir = args.exportdir
    log_file_name = f"{exportdir}/gumbel_max_model.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )
    logging.info(f'Using simulator {args.simulator}')
    observational_dataset = ObservationalDataset(
        args.path, xt_columns=cols, action_columns=["A_t"]
    )
    pyro.clear_param_store()
    validation_split = 0.20
    test_split = 0.20
    shuffle_dataset = True
    dataset_size = len(observational_dataset)
    indices = list(range(dataset_size))
    # split_val = int(np.floor(validation_split * dataset_size))
    # split_test = int(np.floor(test_split * dataset_size))
    # if shuffle_dataset:
    #     np.random.shuffle(indices)
    # train_indices, val_indices, test_indices = (
    #     indices[split_val + split_test :],
    #     indices[:split_val],
    #     indices[split_val : split_val + split_test],
    # )
    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(indices)

    # train_loader = torch.utils.data.DataLoader(
    #     observational_dataset, batch_size=16, sampler=train_sampler
    # )
    # validation_loader = torch.utils.data.DataLoader(
    #     observational_dataset, batch_size=16, sampler=valid_sampler
    # )
    test_loader = torch.utils.data.DataLoader(
        observational_dataset, batch_size=1, sampler=test_sampler
    )
    adam_params = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lrd": args.lrd,
        "clip_norm": args.clip_norm,
        "betas": (args.beta1, args.beta2),
    }
    optimizer = ClippedAdam(adam_params)
    elbo = TraceEnum_ELBO(max_plate_nesting=2, strict_enumeration_warning=True)
    guide = AutoDelta(poutine.block(gumbel_model.model))
    svi = SVI(gumbel_model.model, guide, optimizer, elbo)
    evaluation_loss, losses = evaluate(svi, test_loader, use_cuda=use_cuda)
    losses.to_csv(f"{exportdir}/losses-{args.simulator}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to observational data")
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--model_dir", help="path to model dir")
    parser.add_argument("--simulator", help="name of simulator to run", type=str, default='real')
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument(
        "--weight_decay", help="weight decay (L2 penalty)", type=float, default=0.0
    )
    parser.add_argument(
        "--lrd", help="learning rate decay", type=float, default=0.99996
    )
    parser.add_argument("--clip-norm", help="Clip norm", type=float, default=10.0)
    parser.add_argument("--beta1", type=float, default=0.96)
    parser.add_argument("--beta2", type=float, default=0.999)
    args = parser.parse_args()
    main(args)