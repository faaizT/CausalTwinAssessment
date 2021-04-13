import re
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import wandb
import pyro
import pyro.distributions as dist
import pyro.contrib.examples.polyphonic_data_loader as poly
from gumbel_max_sim.GumbelMaxModel import GumbelMaxModel
from gumbel_max_sim.utils.ObservationalDataset import ObservationalDataset, cols
from pyro.infer import SVI, Trace_ELBO
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from pyro.optim import ClippedAdam


def log_initial_state(model, epoch, device="cpu"):
    softmax = torch.nn.Softmax(dim=0)
    with torch.no_grad():
        diab_data = [[0, softmax(model.s0_diab_logits).numpy()[0]],
                     [1, softmax(model.s0_diab_logits).numpy()[1]]]
        diab_table = wandb.Table(data=diab_data, columns=["s0_diab_idx", "probability"])
        hr_no_diab = softmax(model.s0_hr(torch.FloatTensor([0]).to(device))).numpy()
        hr_data_no_diab = [[0, hr_no_diab[0]],[1, hr_no_diab[1]], [2, hr_no_diab[2]]]
        hr_no_diab_table = wandb.Table(data=hr_data_no_diab, columns=["s0_hr_no_diab", "probability"])
        hr_diab = softmax(model.s0_hr(torch.FloatTensor([1]).to(device))).numpy()
        hr_data_diab = [[0, hr_diab[0]],[1, hr_diab[1]], [2, hr_diab[2]]]
        hr_diab_table = wandb.Table(data=hr_data_diab, columns=["s0_hr_diab", "probability"])
        sysbp_no_diab = softmax(model.s0_sysbp(torch.FloatTensor([0]).to(device))).numpy()
        sysbp_data_no_diab = [[0, sysbp_no_diab[0]],[1, sysbp_no_diab[1]], [2, sysbp_no_diab[2]]]
        sysbp_no_diab_table = wandb.Table(data=sysbp_data_no_diab, columns=["s0_sysbp_no_diab", "probability"])
        sysbp_diab = softmax(model.s0_sysbp(torch.FloatTensor([1]).to(device))).numpy()
        sysbp_data_diab = [[0, sysbp_diab[0]],[1, sysbp_diab[1]], [2, sysbp_diab[2]]]
        sysbp_diab_table = wandb.Table(data=sysbp_data_diab, columns=["s0_sysbp_diab", "probability"])
        glucose_no_diab = softmax(model.s0_glucose(torch.FloatTensor([0]).to(device))).numpy()
        glucose_data_no_diab = [[0, glucose_no_diab[0]],
                                [1, glucose_no_diab[1]],
                                [2, glucose_no_diab[2]],
                                [3, glucose_no_diab[3]],
                                [4, glucose_no_diab[4]]]
        glucose_no_diab_table = wandb.Table(data=glucose_data_no_diab, columns=["s0_glucose_no_diab", "probability"])
        glucose_diab = softmax(model.s0_glucose(torch.FloatTensor([1]).to(device))).numpy()
        glucose_data_diab = [[0, glucose_diab[0]],
                             [1, glucose_diab[1]],
                             [2, glucose_diab[2]],
                             [3, glucose_diab[3]],
                             [4, glucose_diab[4]]]
        glucose_diab_table = wandb.Table(data=glucose_data_diab, columns=["s0_glucose_diab", "probability"])
        percoxyg_no_diab = softmax(model.s0_percoxyg(torch.FloatTensor([0]).to(device))).numpy()
        percoxyg_data_no_diab = [[0, percoxyg_no_diab[0]],[1, percoxyg_no_diab[1]]]
        percoxyg_no_diab_table = wandb.Table(data=percoxyg_data_no_diab, columns=["s0_percoxyg_no_diab", "probability"])
        percoxyg_diab = softmax(model.s0_percoxyg(torch.FloatTensor([1]).to(device))).numpy()
        percoxyg_data_diab = [[0, percoxyg_diab[0]],[1, percoxyg_diab[1]]]
        percoxyg_diab_table = wandb.Table(data=percoxyg_data_diab, columns=["s0_percoxyg_diab", "probability"])
    wandb.log({'epoch': epoch,
               'diab_table': diab_table,
               'hr_no_diab_table': hr_no_diab_table,
               'hr_diab_table': hr_diab_table,
               'sysbp_no_diab_table': sysbp_no_diab_table,
               'sysbp_diab_table': sysbp_diab_table,
               'glucose_no_diab_table': glucose_no_diab_table,
               'glucose_diab_table': glucose_diab_table,
               'percoxyg_no_diab_table': percoxyg_no_diab_table,
               'percoxyg_diab_table': percoxyg_diab_table})


def delete_redundant_states(dir):
    pattern = f"(model|optimiser)-state-[0-9]+"
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def save_states(model, exportdir, iter_num=None, save_final=False):
    logging.info("saving model and optimiser states to %s..." % exportdir)
    if save_final:
        torch.save(model.state_dict(), exportdir + f"/model-state-final")
    else:
        torch.save(model.state_dict(), exportdir + f"/model-state-{iter_num}")
    logging.info("done saving model checkpoints to disk.")


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.0
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for mini_batch, actions, lengths in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            mini_batch = mini_batch.cuda()
            actions = actions.cuda()
            lengths = lengths.cuda()
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(torch.arange(mini_batch.size(0)), mini_batch, lengths, cuda=use_cuda)
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(
            mini_batch.float(), actions.float(), mini_batch_mask.float(), mini_batch_seq_lengths, mini_batch_reversed.float()
        )
    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
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
        test_loss += svi.evaluate_loss(
            mini_batch.float(), actions.float(), mini_batch_mask.float(), mini_batch_seq_lengths, mini_batch_reversed.float()
        )
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gumbel_model = GumbelMaxModel(use_cuda=use_cuda, tanh_activation=args.tanh_act)
    gumbel_model.to(device)
    exportdir = args.exportdir
    log_file_name = f"{exportdir}/gumbel_max_model.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )
    observational_dataset = ObservationalDataset(
        args.path, xt_columns=cols, action_columns=["A_t"]
    )
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
        epoch_loss_train = train(svi, train_loader, use_cuda=use_cuda)
        train_loss["Epochs"].append(epoch)
        train_loss["Training Loss"].append(epoch_loss_train)
        wandb.log({"epoch": epoch, "Training Loss": epoch_loss_train})
        logging.info(
            "[epoch %03d]  average training loss: %.4f" % (epoch, epoch_loss_train)
        )
        if (epoch + 1) % SAVE_N_TEST_FREQUENCY == 0:
            # report test diagnostics
            epoch_loss_val = evaluate(svi, validation_loader, use_cuda=use_cuda)
            validation_loss["Epochs"].append(epoch)
            validation_loss["Test Loss"].append(epoch_loss_val)
            wandb.log({"epoch": epoch, "Test Loss": epoch_loss_val})
            logging.info(
                "[epoch %03d] average validation loss: %.4f" % (epoch, epoch_loss_val)
            )
            pd.DataFrame(data=train_loss).to_csv(exportdir + f"/train-loss.csv")
            pd.DataFrame(data=validation_loss).to_csv(
                exportdir + f"/validation-loss.csv"
            )
            save_states(gumbel_model, exportdir, iter_num=i)
            log_initial_state(gumbel_model, epoch, device)
            i += 1
    pd.DataFrame(data=train_loss).to_csv(exportdir + f"/train-loss.csv")
    pd.DataFrame(data=validation_loss).to_csv(exportdir + f"/validation-loss.csv")
    epoch_loss_test = evaluate(svi, validation_loader, use_cuda=use_cuda)
    logging.info("last epoch error: %.4f" % epoch_loss_test)
    min_val, idx = min(
        (val, idx) for (idx, val) in enumerate(validation_loss["Test Loss"])
    )
    logging.info(f"Index chosen: {idx}")
    gumbel_model.load_state_dict(torch.load(exportdir + f"/model-state-{idx}"))
    gumbel_model.eval()
    epoch_loss_test = evaluate(svi, test_loader, use_cuda=use_cuda)
    logging.info("Chosen epoch error: %.4f" % epoch_loss_test)
    save_states(gumbel_model, exportdir, save_final=True)
    log_initial_state(gumbel_model, epoch, device)
    if args.delete_states:
        delete_redundant_states(exportdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to observational data")
    parser.add_argument(
        "epochs", help="maximum number of epochs to train for", type=int, default=100
    )
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--run_name", help="wandb run name", type=str, required=True)
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
    parser.add_argument(
        "--delete_states",
        help="delete redundant states from exportdir",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--tanh_act",
        help="use tanh activation in combiner",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    wandb.init(project="SimulatorValidation", name=args.run_name)
    wandb.config.lr = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.lrd = args.lrd
    wandb.config.clip_norm = args.clip_norm
    wandb.config.betas = (args.beta1, args.beta2)
    main(args)