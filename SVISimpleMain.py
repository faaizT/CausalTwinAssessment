import argparse
import logging
import os
import re

import pandas as pd
import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro.distributions as dist
from max_likelihood.utils.ObservationalDataset import ObservationalDataset
from torch.utils.data.sampler import SubsetRandomSampler

from simple_model import Policy
from simple_model.GenerateSimpleModelData import S_t, physicians_policy
from simple_model.SimpleModel import cols, SimpleModel

params = {'Epochs': [],
          'mu_model_1': [], 'mu_model_2': [], 'sigma_model_11': [],
          'sigma_model_12': [], 'sigma_model_21': [], 'sigma_model_22': []}


def save_params(epoch):
    global params
    params['Epochs'].append(epoch)
    params['mu_model_1'].append(pyro.param("mu_model")[0].item())
    params['mu_model_2'].append(pyro.param("mu_model")[1].item())
    params['sigma_model_11'].append(pyro.param("sigma_model")[0][0].item())
    params['sigma_model_12'].append(pyro.param("sigma_model")[0][1].item())
    params['sigma_model_21'].append(pyro.param("sigma_model")[1][0].item())
    params['sigma_model_22'].append(pyro.param("sigma_model")[1][1].item())


def write_to_file(file_name, x, y, loss):
    with open(file_name, 'a', 1) as f:
        f.write(str(x) + ',' + str(y) + ',' + str(loss) + os.linesep)


def delete_redundant_states(dir, x, y):
    pattern = f"(model|optimiser)-state-{x}-{y}-[0-9]+"
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def get_policy_accuracy(model: SimpleModel):
    df = pd.DataFrame()
    for i in range(10000):
        state = S_t()
        s0_1, s0_2 = state.s_t[0], state.s_t[1]
        xt, ut = state.get_xt(), state.get_ut()
        action = physicians_policy(xt, ut)
        df = df.append({'s0_1': s0_1, 's0_2': s0_2, 'x0': xt, 'a0': action}, ignore_index=True)
    model.eval()
    acc = 0
    with torch.no_grad():
        for index, row in df.iterrows():
            s0_1, s0_2 = row['s0_1'], row['s0_2']
            action, xt = row['a0'], row['x0']
            pred_action = dist.Categorical(
                logits=model.policy(torch.tensor((s0_1, s0_2)).float())).sample().numpy().item(0)
            acc += (pred_action == action)
    model.train()
    return acc / len(df) * 100


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


def save_states(model, optimizer, exportdir, iter_num=None, save_final=False):
    logging.info("saving model and optimiser states to %s..." % exportdir)
    x, y = model.increment_factor.numpy()
    if save_final:
        torch.save(model.state_dict(), exportdir + f"/model-state-{x}-{y}-final")
        optimizer.save(exportdir + f"/optimiser-state-{x}-{y}-final")
    else:
        torch.save(model.state_dict(), exportdir + f"/model-state-{x}-{y}-{iter_num}")
        optimizer.save(exportdir + f"/optimiser-state-{x}-{y}-{iter_num}")
    logging.info("done saving model and optimizer checkpoints to disk.")


def main(args):
    simulator_model = SimpleModel(learn_initial_state=args.learn_initial_state, increment_factor=args.increment_factor,
                                  phy_pol=args.phy_pol, rho=args.rho)
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
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16, sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(observational_dataset, batch_size=16, sampler=test_sampler)
    adam_params = {"lr": args.lr, "weight_decay": args.weight_decay, "lrd": args.lrd}
    optimizer = ClippedAdam(adam_params)
    svi = SVI(simulator_model.model, simulator_model.guide, optimizer, Trace_ELBO())

    NUM_EPOCHS = args.epochs
    train_loss = {'Epochs': [], 'Training Loss': []}
    validation_loss = {'Epochs': [], 'Test Loss': []}
    SAVE_N_TEST_FREQUENCY = 4
    # training loop
    i = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss_train = train(svi, train_loader, use_cuda=False)
        train_loss['Epochs'].append(epoch)
        train_loss['Training Loss'].append(epoch_loss_train)
        logging.info("[epoch %03d]  average training loss: %.4f" % (epoch, epoch_loss_train))
        if (epoch+1) % SAVE_N_TEST_FREQUENCY == 0:
            # report test diagnostics
            epoch_loss_val = evaluate(svi, validation_loader, use_cuda=False)
            validation_loss['Epochs'].append(epoch)
            validation_loss['Test Loss'].append(epoch_loss_val)
            logging.info("[epoch %03d] average validation loss: %.4f" % (epoch, epoch_loss_val))
            if args.learn_initial_state:
                save_params(epoch)
                pd.DataFrame(data=params).to_csv(exportdir + f"/initial-state-params-{x}-{y}.csv")
            pd.DataFrame(data=train_loss).to_csv(exportdir+f"/train-loss-{x}-{y}.csv")
            pd.DataFrame(data=validation_loss).to_csv(exportdir+f"/validation-loss-{x}-{y}.csv")
            save_states(simulator_model,optimizer, exportdir, iter_num=i)
            i += 1
    pd.DataFrame(data=train_loss).to_csv(exportdir + f"/train-loss-{x}-{y}.csv")
    pd.DataFrame(data=validation_loss).to_csv(exportdir + f"/validation-loss-{x}-{y}.csv")
    epoch_loss_test = evaluate(svi, test_loader, use_cuda=False)
    logging.info("last epoch error: %.4f" % epoch_loss_test)
    min_val, idx = min((val, idx) for (idx, val) in enumerate(validation_loss['Test Loss']))
    logging.info(f"Index chosen: {idx}")
    simulator_model.load_state_dict(torch.load(exportdir + f"/model-state-{x}-{y}-{idx}"))
    optimizer.load(exportdir + f"/optimiser-state-{x}-{y}-{idx}")
    simulator_model.eval()
    svi = SVI(simulator_model.model, simulator_model.guide, optimizer, Trace_ELBO())
    epoch_loss_test = evaluate(svi, test_loader, use_cuda=False)
    logging.info("Chosen epoch error: %.4f" % epoch_loss_test)
    save_states(simulator_model, optimizer, exportdir, save_final=True)
    write_to_file(args.output_file, x, y, epoch_loss_test)
    policy_acc = get_policy_accuracy(model=simulator_model)
    write_to_file(args.accuracy_file, x, y, policy_acc)
    if args.delete_states:
        delete_redundant_states(exportdir, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to observational data")
    parser.add_argument("epochs", help="maximum number of epochs to train for", type=int)
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("output_file", help="Output file to contain final test loss results")
    parser.add_argument("accuracy_file", help="Output file to contain policy accuracy")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", help="weight decay (L2 penalty)", type=float, default=2.0)
    parser.add_argument("--rho", help="Correlation coefficient of S_0 components", type=float, default=None)
    parser.add_argument("--increment_factor", nargs='+', help="factor by which simulator increments s_t values",
                        type=int, default=None)
    parser.add_argument("--delete_states", help="delete redundant states from exportdir", type=bool, default=False)
    parser.add_argument("--phy_pol", help="use physicians' policy in model", type=bool, default=False)
    parser.add_argument("--learn_initial_state", help="learn initial state distribution", type=bool, default=False)
    parser.add_argument('--lrd', help="learning rate decay", type=float, default=0.99996)
    args = parser.parse_args()
    if not os.path.exists(args.output_file):
        with open(args.output_file, "w") as f:
            f.write('x,y,test loss' + os.linesep)
    if not os.path.exists(args.accuracy_file) and not args.phy_pol:
        with open(args.accuracy_file, "w") as f:
            f.write('x,y,policy accuracy' + os.linesep)
    main(args)
