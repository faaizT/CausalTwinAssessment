import argparse
import logging
import numpy as np
import pandas as pd
import torch
import pyro
from gumbel_max_sim.GumbelMaxModel import GumbelMaxModel, cols
from max_likelihood.utils.ObservationalDataset import ObservationalDataset
from pyro.infer import SVI, Trace_ELBO
from torch.utils.data.sampler import SubsetRandomSampler
from pyro.optim import ClippedAdam


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.0
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x in train_loader:
        # if on GPU put mini-batch into CUDA memory
        reversed_x = torch.flip(x, [1])
        if use_cuda:
            x = x.cuda()
            reversed_x = reversed_x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x.float(), reversed_x.float())
    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.0
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        reversed_x = torch.flip(x, [1])
        if use_cuda:
            x = x.cuda()
            reversed_x = reversed_x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x.float(), reversed_x.float())
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def main(args):
    gumbel_model = GumbelMaxModel()
    exportdir = args.exportdir
    log_file_name = f"{exportdir}/gumbel_max_model.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )
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
    train_indices, val_indices, test_indices = (
        indices[split_val + split_test :],
        indices[:split_val],
        indices[split_val : split_val + split_test],
    )
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        observational_dataset, batch_size=16, sampler=train_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        observational_dataset, batch_size=16, sampler=valid_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        observational_dataset, batch_size=16, sampler=test_sampler
    )
    adam_params = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lrd": args.lrd,
        "clip_norm": args.clip_norm,
        "betas": (args.beta1, args.beta2),
    }
    optimizer = ClippedAdam(adam_params)
    svi = SVI(gumbel_model.model, gumbel_model.guide, optimizer, Trace_ELBO())
    NUM_EPOCHS = args.epochs
    train_loss = {"Epochs": [], "Training Loss": []}
    validation_loss = {"Epochs": [], "Test Loss": []}
    SAVE_N_TEST_FREQUENCY = 4
    # training loop
    i = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss_train = train(svi, train_loader, use_cuda=False)
        train_loss["Epochs"].append(epoch)
        train_loss["Training Loss"].append(epoch_loss_train)
        logging.info(
            "[epoch %03d]  average training loss: %.4f" % (epoch, epoch_loss_train)
        )
        if (epoch + 1) % SAVE_N_TEST_FREQUENCY == 0:
            # report test diagnostics
            epoch_loss_val = evaluate(svi, validation_loader, use_cuda=False)
            validation_loss["Epochs"].append(epoch)
            validation_loss["Test Loss"].append(epoch_loss_val)
            logging.info(
                "[epoch %03d] average validation loss: %.4f" % (epoch, epoch_loss_val)
            )
            # if args.learn_initial_state:
            #     save_params(gumbel_model, epoch)
            #     pd.DataFrame(data=params).to_csv(
            #         exportdir + f"/initial-state-params-{x}-{y}.csv"
            #     )
            # pd.DataFrame(data=train_loss).to_csv(exportdir + f"/train-loss-{x}-{y}.csv")
            # pd.DataFrame(data=validation_loss).to_csv(
            #     exportdir + f"/validation-loss-{x}-{y}.csv"
            # )
            # save_states(simulator_model, exportdir, iter_num=i)
            # i += 1
    pd.DataFrame(data=train_loss).to_csv(exportdir + f"/train-loss.csv")
    pd.DataFrame(data=validation_loss).to_csv(exportdir + f"/validation-loss.csv")
    epoch_loss_test = evaluate(svi, validation_loader, use_cuda=False)
    logging.info("last epoch error: %.4f" % epoch_loss_test)
    # min_val, idx = min(
    #     (val, idx) for (idx, val) in enumerate(validation_loss["Test Loss"])
    # )
    # logging.info(f"Index chosen: {idx}")
    # simulator_model.load_state_dict(
    #     torch.load(exportdir + f"/model-state-{x}-{y}-{idx}")
    # )
    # simulator_model.eval()
    # epoch_loss_test = evaluate(simulator_model, test_loader, use_cuda=False)
    # logging.info("Chosen epoch error: %.4f" % epoch_loss_test)
    # save_states(simulator_model, exportdir, save_final=True)
    # if args.delete_states:
    #     delete_redundant_states(exportdir, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to observational data")
    parser.add_argument(
        "epochs", help="maximum number of epochs to train for", type=int, default=100
    )
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
    parser.add_argument(
        "--weight_decay", help="weight decay (L2 penalty)", type=float, default=2.0
    )
    parser.add_argument(
        "--lrd", help="learning rate decay", type=float, default=0.99996
    )
    parser.add_argument("--clip-norm", help="Clip norm", type=float, default=10.0)
    parser.add_argument("--beta1", type=float, default=0.96)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--delete_states",
        help="delete redundant states from exportdir",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    main(args)