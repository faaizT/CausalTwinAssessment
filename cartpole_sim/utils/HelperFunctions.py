import numpy as np
import time
import math
import random
import torch
from torch.autograd import Variable
from cartpole_sim.utils.Networks import *
from cartpole_sim.utils.PinballLoss import PinballLoss
import torch.utils.data as data_utils
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import pandas as pd
import logging
import wandb
import os

# x_columns = ['Cart Position', 'Cart Velocity (abs)', 'Pole Angle', 'Pole Angular Velocity (abs)']
x_columns = ["Pole Angular Velocity", "Pole Angle"]

outcome = "Pole Angle"


def log_policy_accuracy(policy, Xtest, Ytest, epoch, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = policy(Xtest)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += Xtest.size(0)
        correct += (predicted == Ytest).sum().item()

    wandb.log({"epoch": epoch, f"acc-{model}": 100 * correct / total})


def normalise(df, data):
    return (df[x_columns] - data[x_columns].mean()) / data[x_columns].std()


def train_policies(data, obs_data_train, obs_bootstrap, models_dir, model):
    training_ids = (
        obs_bootstrap[["episode", "t"]].apply(lambda x: (x[0], x[1]), axis=1).unique()
    )
    include_in_testing = (
        ~obs_data_train[["episode", "t"]]
        .apply(lambda x: (x[0], x[1]), axis=1)
        .isin(training_ids)
    )
    test_data = obs_data_train[include_in_testing]
    X = torch.FloatTensor(normalise(obs_bootstrap, data).values)
    Xtest = torch.FloatTensor(normalise(test_data, data).values)
    Y = torch.tensor(obs_bootstrap["A"].values).to(torch.long)
    Ytest = torch.tensor(test_data["A"].values).to(torch.long)

    train = data_utils.TensorDataset(X, Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(Xtest, Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    loss_func = torch.nn.CrossEntropyLoss()
    policy = PolicyNetwork(input_dim=len(x_columns), output_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in tqdm(range(100)):
        for data, label in trainloader:
            prediction = policy(data)  # input x and predict based on x
            loss = loss_func(prediction, label)  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_loss = 0
                for test_data, test_label in testloader:
                    test_loss += loss_func(policy(test_data), test_label)
                test_loss /= len(testloader)
                wandb.log({"epoch": epoch, f"Policy - mdl {model}": test_loss})
            log_policy_accuracy(policy, Xtest, Ytest, epoch, model)
    torch.save(policy.state_dict(), f"{models_dir}/policy_{model}")


def train_yobs(data, obs_data_train, obs_bootstrap, models_dir, model):
    training_ids = (
        obs_bootstrap[["episode", "t"]].apply(lambda x: (x[0], x[1]), axis=1).unique()
    )
    include_in_testing = (
        ~obs_data_train[["episode", "t"]]
        .apply(lambda x: (x[0], x[1]), axis=1)
        .isin(training_ids)
    )
    test_data = obs_data_train[include_in_testing]
    X = torch.FloatTensor(normalise(obs_bootstrap, data).values)
    Xtest = torch.FloatTensor(normalise(test_data, data).values)
    A = torch.FloatTensor(obs_bootstrap["A"].values).to(torch.long)
    Atest = torch.FloatTensor(test_data["A"].values).to(torch.long)
    Y = torch.FloatTensor(
        (obs_bootstrap[f"{outcome}_t1"] - data[outcome].mean()).values
        / data[outcome].std()
    ).unsqueeze(dim=1)
    Ytest = torch.FloatTensor(
        (test_data[f"{outcome}_t1"] - data[outcome].mean()).values / data[outcome].std()
    ).unsqueeze(dim=1)

    train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(torch.column_stack((Xtest, Atest)), Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    loss_func = torch.nn.MSELoss()
    obs_net = Net(n_feature=len(x_columns) + 1, n_hidden=4, n_output=1)
    optimizer = torch.optim.SGD(obs_net.parameters(), lr=0.01)

    for epoch in tqdm(range(100)):
        for X, Y in trainloader:
            prediction = obs_net(X)
            loss = loss_func(prediction, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_loss = 0
            for Xtest, Ytest in testloader:
                test_loss += loss_func(obs_net(Xtest), Ytest)
            test_loss = test_loss / len(testloader)
            wandb.log({"epoch": epoch, f"yobs - {model}": test_loss})

    torch.save(obs_net.state_dict(), f"{models_dir}/yobs_{model}")


def train_yminmax(
    data, quantile_data, quantile_data_bootstrap, quantile, models_dir, model
):
    training_ids = (
        quantile_data_bootstrap[["episode", "t"]]
        .apply(lambda x: (x[0], x[1]), axis=1)
        .unique()
    )
    include_in_testing = (
        ~quantile_data[["episode", "t"]]
        .apply(lambda x: (x[0], x[1]), axis=1)
        .isin(training_ids)
    )
    test_data = quantile_data[include_in_testing]
    X = torch.FloatTensor(normalise(quantile_data_bootstrap, data).values)
    Xtest = torch.FloatTensor(normalise(test_data, data).values)
    A = torch.FloatTensor(quantile_data_bootstrap["A"].values).to(torch.long)
    Atest = torch.FloatTensor(test_data["A"].values).to(torch.long)
    Y = torch.FloatTensor(
        (quantile_data_bootstrap[f"{outcome}_t1"] - data[outcome].mean()).values
        / data[outcome].std()
    ).unsqueeze(dim=1)
    Ytest = torch.FloatTensor(
        (test_data[f"{outcome}_t1"] - data[outcome].mean()).values / data[outcome].std()
    ).unsqueeze(dim=1)

    train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(torch.column_stack((Xtest, Atest)), Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    loss_func = PinballLoss(quantile=quantile, reduction="mean")
    ymax_net = Net(n_feature=len(x_columns) + 1, n_hidden=4, n_output=1)
    optimizer = torch.optim.SGD(ymax_net.parameters(), lr=0.005)

    for epoch in tqdm(range(100)):
        for X, Y in trainloader:
            prediction = ymax_net(X)
            loss = loss_func(prediction, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_loss = 0
            for Xtest, Ytest in testloader:
                test_loss += loss_func(ymax_net(Xtest), Ytest)
            test_loss = test_loss / len(testloader)
            wandb.log({"epoch": epoch, f"ymax - {model}": test_loss})
    torch.save(ymax_net.state_dict(), f"{models_dir}/ymax_{quantile}_{model}")

    loss_func = PinballLoss(quantile=1 - quantile, reduction="mean")
    ymin_net = Net(n_feature=len(x_columns) + 1, n_hidden=4, n_output=1)
    optimizer = torch.optim.SGD(ymin_net.parameters(), lr=0.005)

    for epoch in tqdm(range(100)):
        for X, Y in trainloader:
            prediction = ymin_net(X)
            loss = loss_func(prediction, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_loss = 0
            for Xtest, Ytest in testloader:
                test_loss += loss_func(ymin_net(Xtest), Ytest)
            test_loss = test_loss / len(testloader)
            wandb.log({"epoch": epoch, f"ymin - {model}": test_loss})
    torch.save(ymin_net.state_dict(), f"{models_dir}/ymin_{quantile}_{model}")


def train_fake_ysim(data, sim_data, sim_bootstrap, models_dir, model, noise_std):
    training_ids = (
        sim_bootstrap[["episode", "t"]].apply(lambda x: (x[0], x[1]), axis=1).unique()
    )
    include_in_testing = (
        ~sim_data[["episode", "t"]]
        .apply(lambda x: (x[0], x[1]), axis=1)
        .isin(training_ids)
    )
    test_data = sim_data.loc[include_in_testing]
    X = torch.FloatTensor(normalise(sim_bootstrap, data).values)
    Xtest = torch.FloatTensor(normalise(test_data, data).values)
    A = torch.FloatTensor(sim_bootstrap["A"].values).to(torch.long)
    Atest = torch.FloatTensor(test_data["A"].values).to(torch.long)
    obs_net = Net(n_feature=len(x_columns) + 1, n_hidden=4, n_output=1)
    obs_net.load_state_dict(torch.load(f"{models_dir}/yobs_0"))
    obs_net.eval()
    Y = obs_net(torch.column_stack((X, A))).squeeze().detach() + torch.normal(
        mean=torch.zeros(X.shape[0]), std=noise_std
    )
    Ytest = obs_net(
        torch.column_stack((Xtest, Atest))
    ).squeeze().detach() + torch.normal(mean=torch.zeros(Xtest.shape[0]), std=noise_std)
    # Clipping in the correct range
    Y = torch.max(Y, torch.tensor(-0.418))
    Y = torch.min(Y, torch.tensor(0.418)).unsqueeze(dim=1)
    Ytest = torch.max(Ytest, torch.tensor(-0.418))
    Ytest = torch.min(Ytest, torch.tensor(0.418)).unsqueeze(dim=1)

    train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(torch.column_stack((Xtest, Atest)), Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    loss_func = torch.nn.MSELoss()
    sim_net = Net(n_feature=len(x_columns) + 1, n_hidden=4, n_output=1)
    optimizer = torch.optim.SGD(sim_net.parameters(), lr=0.001)

    for epoch in tqdm(range(100)):
        for X, Y in trainloader:
            prediction = sim_net(X)
            loss = loss_func(prediction, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_loss = 0
            for Xtest, Ytest in testloader:
                test_loss += loss_func(sim_net(Xtest), Ytest)
            test_loss = test_loss / len(testloader)
            if epoch % 10 == 0:
                wandb.log({"epoch": epoch, f"ysim fake - {noise_std}": test_loss})
    torch.save(
        sim_net.state_dict(), f"{models_dir}/ysim_gaussian_fake_{model}_{noise_std}"
    )


def train_ysim(data, sim_data, models_dir, model, false_sim=False, perturbation=0):
    test_ids = np.random.rand(len(sim_data)) > 0.8
    train_data, test_data = sim_data[~test_ids], sim_data[test_ids]
    X = torch.FloatTensor(normalise(train_data, data).values)
    Xtest = torch.FloatTensor(normalise(test_data, data).values)
    A = torch.FloatTensor(train_data["A"].values).to(torch.long)
    Atest = torch.FloatTensor(test_data["A"].values).to(torch.long)
    if false_sim:
        epsilon = perturbation * (sim_bootstrap["A"] == 1.0) + perturbation * (
            sim_bootstrap["A"] == 0
        )
        epsilon_test = perturbation * (test_data["A"] == 1.0) + perturbation * (
            test_data["A"] == 0
        )
    else:
        epsilon = 0
        epsilon_test = 0
    Y = torch.FloatTensor(
        (train_data[f"{outcome}_t1"] + epsilon - data[outcome].mean()).values
        / data[outcome].std()
    ).unsqueeze(dim=1)
    Ytest = torch.FloatTensor(
        (test_data[f"{outcome}_t1"] + epsilon_test - data[outcome].mean()).values
        / data[outcome].std()
    ).unsqueeze(dim=1)

    train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(torch.column_stack((Xtest, Atest)), Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    loss_func = torch.nn.MSELoss()
    sim_net = Net(n_feature=len(x_columns) + 1, n_hidden=6, n_output=1)
    optimizer = torch.optim.SGD(sim_net.parameters(), lr=0.001)

    for epoch in tqdm(range(100)):
        for X, Y in trainloader:
            prediction = sim_net(X)
            loss = loss_func(prediction, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_loss = 0
            for Xtest, Ytest in testloader:
                test_loss += loss_func(sim_net(Xtest), Ytest)
            test_loss = test_loss / len(testloader)
            if false_sim:
                wandb.log({"epoch": epoch, f"ysim false - {model}": test_loss})
            else:
                wandb.log({"epoch": epoch, f"ysim - {model}": test_loss})

    if false_sim:
        torch.save(
            sim_net.state_dict(), f"{models_dir}/ysim_false2_{model}_{perturbation}"
        )
    else:
        torch.save(sim_net.state_dict(), f"{models_dir}/ysim_{model}")


def load_and_preprocess_data(args):
    logging.info("Preprocessing data")
    df = pd.read_csv(f"{args.files_dir}/Cartpole-v1-obs-data-iid.csv")
    sim_data = pd.read_csv(f"{args.files_dir}/Cartpole-v1-sim-data-iid.csv")
    quantile_data = pd.read_csv(f"{args.files_dir}/Cartpole-v1-sim-data-iid.csv")
    obs_data_train = pd.read_csv(f"{args.files_dir}/Cartpole-v1-obs-data-iid-train.csv")
    return df, sim_data, obs_data_train, quantile_data


def transform_data(filename):
    rename_cols_t1 = {k: f"{k}_t1" for k in x_columns}
    data = pd.read_csv(filename)
    combined_data = pd.DataFrame()
    for episode in data["episode"].unique():
        data_t0 = (
            data.loc[data["episode"] == episode].copy().iloc[:-1].reset_index(drop=True)
        )
        data_t1 = (
            data.loc[data["episode"] == episode]
            .copy()
            .iloc[1:][x_columns]
            .rename(columns=rename_cols_t1)
            .reset_index(drop=True)
        )
        df_episode = pd.merge(data_t0, data_t1, left_index=True, right_index=True)
        combined_data = combined_data.append(df_episode, ignore_index=True)
    combined_data.to_csv(f"{filename.replace('.csv', '')}-combined.csv", index=False)


def combine_data(args):
    for i in tqdm(range(500)):
        file_path = f"/data/localhost/not-backed-up/taufiq/Cartpole/Cartpole-v1-obs-data-run{i}-ep100.csv"
        transform_data(file_path)


def preprocess_iid_sim_data(args):
    data = pd.DataFrame()
    for i in range(500):
        file_path = f"/data/localhost/not-backed-up/taufiq/Cartpole/simulator_data/Cartpole-v1-sim-data-run{i}-ep100-combined.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data = data.append(df.groupby("episode").sample(), ignore_index=True)
    data.to_csv(
        "/data/localhost/taufiq/export-dir/Cartpole-v1-sim-data-iid.csv", index=False
    )


def preprocess_iid_obs_data(args):
    data = pd.DataFrame()
    for i in tqdm(range(500)):
        file_path = f"/data/localhost/not-backed-up/taufiq/Cartpole/obs-data-epsilon-0.3/Cartpole-v1-obs-data-run{i}-ep100.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            data = data.append(df.groupby("episode").sample(), ignore_index=True)
    data.to_csv(
        "/data/localhost/taufiq/export-dir/Cartpole-v1-obs-data-iid-eps0.3.csv",
        index=False,
    )
    p = np.random.rand(len(data))
    data[p < 0.8].to_csv(
        "/data/localhost/taufiq/export-dir/Cartpole-v1-obs-data-iid-eps0.3-train.csv",
        index=False,
    )
    data[p >= 0.8].to_csv(
        "/data/localhost/taufiq/export-dir/Cartpole-v1-obs-data-iid-eps0.3-test.csv",
        index=False,
    )
