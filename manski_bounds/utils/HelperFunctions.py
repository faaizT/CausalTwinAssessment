import argparse
import logging
import pandas as pd
import numpy as np
import os
import glob
import re
import torch
import torch.utils.data as data_utils
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from manski_bounds.utils.Networks import *
from manski_bounds.utils.Losses import *
from tqdm import tqdm
import wandb


column_mappings = {
    'Albumin - BloodConcentration (mg/L)': 'Albumin',
    'ArterialCarbonDioxidePressure (mmHg)': 'paCO2',
    'ArterialOxygenPressure (mmHg)': 'paO2',
    'Bicarbonate - BloodConcentration (mg/L)': 'HCO3',
    'BloodPH (None)': 'Arterial_pH',
    'BloodUreaNitrogenConcentration (mg/dL)': 'BUN',
    'Calcium - BloodConcentration (mg/L)': 'Calcium',
    'Chloride - BloodConcentration (mg/L)': 'Chloride',
    'Creatinine - BloodConcentration (mg/L)': 'Creatinine',
    'DiastolicArterialPressure (mmHg)': 'DiaBP',
    'Glucose - BloodConcentration (mg/L)': 'Glucose',
    'Lactate - BloodConcentration (mg/L)': 'Arterial_lactate',
    'MeanArterialPressure (mmHg)': 'MeanBP',
    'Potassium - BloodConcentration (mg/L)': 'Potassium',
    'RespirationRate (1/min)': 'RR',
    'SaturationAndFractionOfInspiredOxygenRatio (None)': 'PaO2_FiO2',
    'SkinTemperature (degC)': 'Temp_C',
    'Sodium - BloodConcentration (mg/L)': 'Sodium',
    'SystolicArterialPressure (mmHg)': 'SysBP',
    'WhiteBloodCellCount (ct/uL)': 'WBC_count',
    'HeartRate (1/min)': 'HR'
}

cols = ['paCO2', 'paO2', 'HCO3', 'Arterial_pH', 'Calcium', 'Chloride', 'DiaBP', 'Glucose', 'MeanBP', 'Potassium', 'RR', 'Temp_C', 'Sodium', 'SysBP', 'HR']


def log_policy_accuracy(policy, Xtestmimic, Ytest, epoch, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        outputs = policy(Xtestmimic)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += Xtestmimic.size(0)
        correct += (predicted == Ytest).sum().item()

    wandb.log({'epoch': epoch, f'acc-{model}': 100 * correct / total})

def train_ysim_deprecated(MIMIC_data_combined, MIMICtable, pulse_data_combined, pulseraw, actionbloc, models_dir, nr_reps, col_name):
    icuuniqueids = pulse_data_combined['icustay_id'].unique()
    for model in tqdm(range(nr_reps)):
        grp = np.floor(5*np.random.rand(len(icuuniqueids))+1)
        crossval = 1
        trainidx = icuuniqueids[grp != crossval]
        testidx = icuuniqueids[grp == crossval]
        X = torch.FloatTensor(pulseraw.loc[pulse_data_combined['icustay_id'].isin(trainidx)].values)
        Xtestmimic = torch.FloatTensor(pulseraw[pulse_data_combined['icustay_id'].isin(testidx)].values)
        A = (torch.tensor(actionbloc.loc[MIMIC_data_combined['icustay_id'].isin(trainidx), 'action_bloc'].values).to(torch.long)-1)/24
        Atest = (torch.tensor(actionbloc.loc[MIMIC_data_combined['icustay_id'].isin(testidx), 'action_bloc'].values).to(torch.long)-1)/24
        ptid = pulse_data_combined.loc[pulse_data_combined['icustay_id'].isin(trainidx), 'icustay_id']
        ptidtestmimic = pulse_data_combined.loc[pulse_data_combined['icustay_id'].isin(testidx), 'icustay_id']
        Y = torch.FloatTensor(pulse_data_combined.loc[pulse_data_combined['icustay_id'].isin(trainidx), f'{col_name}_t1'].values).unsqueeze(dim=1)
        Ytest = torch.FloatTensor(pulse_data_combined.loc[pulse_data_combined['icustay_id'].isin(testidx), f'{col_name}_t1'].values).unsqueeze(dim=1)
        Y = (Y - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
        Ytest = (Ytest - MIMICtable[col_name].mean())/MIMICtable[col_name].std()

        train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
        trainloader = torch.utils.data.DataLoader(train, batch_size=32)
        test = data_utils.TensorDataset(torch.column_stack((Xtestmimic, Atest)), Ytest)
        testloader = torch.utils.data.DataLoader(test, batch_size=32)

        net = Net(n_feature=4, n_hidden=10, n_output=1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.1)
        loss_func = torch.nn.MSELoss()

        for epoch in tqdm(range(200)):
            for X, Y in trainloader:
                prediction = net(X)     # input x and predict based on x

                loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            if (epoch+1) % 10 == 0:
                with torch.no_grad():
                    test_loss = 0
                    for Xtest, Ytest in testloader:
                        test_loss += loss_func(net(Xtest), Ytest)
                    test_loss = test_loss/len(testloader)
                    wandb.log({'epoch': epoch, f'TL ysim - mdl {model}': test_loss})
        
        torch.save(net.state_dict(), f'{models_dir}/ysim_{model}')


def train_ysim(MIMICtable, pulse_data_t0_tr, pulse_data_t1_tr, pulse_data_t0, pulse_data_t1, pulseraw, models_dir, col_name, model):
    pulseraw_tr = pulse_data_t0_tr[['icustay_id']].merge(pulseraw, on='icustay_id')
    pulseraw_test = pulseraw[~pulseraw['icustay_id'].isin(pulse_data_t0_tr['icustay_id'])]
    X = torch.FloatTensor(pulseraw_tr.drop(columns=['icustay_id']).values)
    Xtestmimic = torch.FloatTensor(pulseraw_test.drop(columns=['icustay_id']).values)
    A = torch.tensor(pulse_data_t0_tr['A'].values).to(torch.long)-1
    Atest = torch.tensor(pulseraw_test[['icustay_id']].merge(pulse_data_t0, on=['icustay_id'])['A'].values).to(torch.long)-1

    Y = torch.FloatTensor(pulse_data_t1_tr[col_name].values).unsqueeze(dim=1)
    Ytest = torch.FloatTensor(pulseraw_test[['icustay_id']].merge(pulse_data_t1, on='icustay_id')[col_name].values).unsqueeze(dim=1)
    Y = (Y - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
    Ytest = (Ytest - MIMICtable[col_name].mean())/MIMICtable[col_name].std()

    train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(torch.column_stack((Xtestmimic, Atest)), Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    net = Net(n_feature=X.shape[1] + 1, n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.1)
    loss_func = torch.nn.MSELoss()

    for epoch in tqdm(range(200)):
        for X, Y in trainloader:
            prediction = net(X)     # input x and predict based on x

            loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                test_loss = 0
                for Xtest, Ytest in testloader:
                    test_loss += loss_func(net(Xtest), Ytest)
                test_loss = test_loss/len(testloader)
                wandb.log({'epoch': epoch, f'ysim {col_name} - {model}': test_loss})
    
    torch.save(net.state_dict(), f'{models_dir}/ysim_{col_name}_{model}')


def train_yminmax(MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMICtable_filtered_t0_tr, MIMICtable_filtered_t1_tr, MIMICraw, MIMICtable, models_dir, col_name, model):
    MIMICraw_tr = MIMICtable_filtered_t0_tr[['icustay_id']].merge(MIMICraw, on='icustay_id')
    MIMICraw_test = MIMICraw[~MIMICraw['icustay_id'].isin(MIMICtable_filtered_t0_tr['icustay_id'])]
    X = torch.FloatTensor(MIMICraw_tr.drop(columns=['icustay_id']).values)
    Xtestmimic = torch.FloatTensor(MIMICraw_test.drop(columns=['icustay_id']).values)
    Y = torch.FloatTensor(MIMICtable_filtered_t1_tr[col_name].values).unsqueeze(dim=1)
    Ytest = torch.FloatTensor(MIMICraw_test[['icustay_id']].merge(MIMICtable_filtered_t1, on='icustay_id')[col_name].values).unsqueeze(dim=1)

    Y = (Y - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
    Ytest = (Ytest - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
    train = data_utils.TensorDataset(X, Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(Xtestmimic, Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    quantile_net = QuantileNet(n_feature=X.shape[1], n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(quantile_net.parameters(), lr=0.001)
    loss_func = PinballLoss(quantile=0.001, reduction='mean')

    for epoch in tqdm(range(200)):
        for X, Y in trainloader:
            prediction = quantile_net(X)     # input x and predict based on x

            loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        if (epoch+1)%10 == 0:
            with torch.no_grad():
                test_loss = 0
                for Xtest, Ytest in testloader:
                    test_loss += loss_func(quantile_net(Xtest), Ytest)
                test_loss = test_loss/len(testloader)
                wandb.log({'epoch': epoch, f'ymin {col_name} - {model}': test_loss})

    torch.save(quantile_net.state_dict(), f'{models_dir}/ymin_{col_name}_{model}')

    quantile_net = QuantileNet(n_feature=X.shape[1], n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(quantile_net.parameters(), lr=0.001)
    loss_func = PinballLoss(quantile=0.999, reduction='mean')

    for epoch in tqdm(range(200)):
        for X, Y in trainloader:
            prediction = quantile_net(X)     # input x and predict based on x

            loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        if (epoch+1)%10 == 0:
            with torch.no_grad():
                test_loss = 0
                for Xtest, Ytest in testloader:
                    test_loss += loss_func(quantile_net(Xtest), Ytest)
                test_loss = test_loss/len(testloader)
                wandb.log({'epoch': epoch, f'ymax {col_name} - {model}': test_loss})

    torch.save(quantile_net.state_dict(), f'{models_dir}/ymax_{col_name}_{model}')


def train_yminmax_deprecated(MIMIC_data_combined, MIMICtable_filtered_t0, MIMICraw, MIMICtable, actionbloc, models_dir, nr_reps, col_name):
    icuuniqueids = MIMIC_data_combined['icustay_id'].unique()
    for model in tqdm(range(nr_reps)):
        grp = np.floor(5*np.random.rand(len(icuuniqueids))+1)
        crossval = 1
        trainidx = icuuniqueids[grp != crossval]
        testidx = icuuniqueids[grp == crossval]
        X = torch.FloatTensor(MIMICraw.loc[MIMICtable_filtered_t0['icustay_id'].isin(trainidx)].values)
        Xtestmimic = torch.FloatTensor(MIMICraw[MIMICtable_filtered_t0['icustay_id'].isin(testidx)].values)
        ptid = MIMICtable_filtered_t0.loc[MIMICtable_filtered_t0['icustay_id'].isin(trainidx), 'icustay_id']
        ptidtestmimic = MIMICtable_filtered_t0.loc[MIMICtable_filtered_t0['icustay_id'].isin(testidx), 'icustay_id']
        Y = torch.FloatTensor(MIMIC_data_combined.loc[MIMICtable_filtered_t0['icustay_id'].isin(trainidx), f'{col_name}_t1'].values).unsqueeze(dim=1)
        Ytest = torch.FloatTensor(MIMIC_data_combined.loc[MIMICtable_filtered_t0['icustay_id'].isin(testidx), f'{col_name}_t1'].values).unsqueeze(dim=1)
        
        Y = (Y - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
        Ytest = (Ytest - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
        train = data_utils.TensorDataset(X, Y)
        trainloader = torch.utils.data.DataLoader(train, batch_size=32)
        test = data_utils.TensorDataset(Xtestmimic, Ytest)
        testloader = torch.utils.data.DataLoader(test, batch_size=32)
        quantile_net = QuantileNet(n_feature=3, n_hidden=10, n_output=1)
        optimizer = torch.optim.SGD(quantile_net.parameters(), lr=0.001)
        loss_func = PinballLoss(quantile=0.001, reduction='mean')

        for epoch in tqdm(range(200)):
            for X, Y in trainloader:
                prediction = quantile_net(X)     # input x and predict based on x

                loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            if (epoch+1)%10 == 0:
                with torch.no_grad():
                    test_loss = 0
                    for Xtest, Ytest in testloader:
                        test_loss += loss_func(quantile_net(Xtest), Ytest)
                    test_loss = test_loss/len(testloader)
                    wandb.log({'epoch': epoch, f'TL ymin - mdl {model}': test_loss})
        
        torch.save(quantile_net.state_dict(), f'{models_dir}/ymin_{model}')

        quantile_net = QuantileNet(n_feature=3, n_hidden=10, n_output=1)
        optimizer = torch.optim.SGD(quantile_net.parameters(), lr=0.001)
        loss_func = PinballLoss(quantile=0.999, reduction='mean')

        for epoch in tqdm(range(200)):
            for X, Y in trainloader:
                prediction = quantile_net(X)     # input x and predict based on x

                loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            if (epoch+1)%10 == 0:
                with torch.no_grad():
                    test_loss = 0
                    for Xtest, Ytest in testloader:
                        test_loss += loss_func(quantile_net(Xtest), Ytest)
                    test_loss = test_loss/len(testloader)
                    wandb.log({'epoch': epoch, f'TL ymax - mdl {model}': test_loss})

        torch.save(quantile_net.state_dict(), f'{models_dir}/ymax_{model}')


def train_yobs(MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMICtable_filtered_t0_tr, MIMICtable_filtered_t1_tr, MIMICraw, MIMICtable, models_dir, col_name, model):
    MIMICraw_tr = MIMICtable_filtered_t0_tr[['icustay_id']].merge(MIMICraw, on='icustay_id')    
    MIMICraw_test = MIMICraw[~MIMICraw['icustay_id'].isin(MIMICtable_filtered_t0_tr['icustay_id'])]
    X = torch.FloatTensor(MIMICraw_tr.drop(columns=['icustay_id']).values)
    Xtestmimic = torch.FloatTensor(MIMICraw_test.drop(columns=['icustay_id']).values)
    A = torch.tensor(MIMICtable_filtered_t0_tr['A'].values).to(torch.long)-1
    Atest = torch.tensor(MIMICraw_test[['icustay_id']].merge(MIMICtable_filtered_t0, on=['icustay_id'])['A'].values).to(torch.long)-1
    Y = torch.FloatTensor(MIMICtable_filtered_t1_tr[col_name].values).unsqueeze(dim=1)
    Ytest = torch.FloatTensor(MIMICraw_test[['icustay_id']].merge(MIMICtable_filtered_t1, on='icustay_id')[col_name].values).unsqueeze(dim=1)
    
    Y = (Y - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
    Ytest = (Ytest - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
    train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(torch.column_stack((Xtestmimic, Atest)), Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)
    
    net = Net(n_feature=X.shape[1]+1, n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.1)
    loss_func = torch.nn.MSELoss()    
    
    for epoch in tqdm(range(200)):
        for X, Y in trainloader:
            prediction = net(X)     # input x and predict based on x

            loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        if (epoch+1)%10 == 0:
            with torch.no_grad():
                test_loss = 0
                for Xtest, Ytest in testloader:
                    test_loss += loss_func(net(Xtest), Ytest)
                test_loss = test_loss/len(testloader)
                wandb.log({'epoch': epoch, f'yobs {col_name} - {model}': test_loss})
    
    torch.save(net.state_dict(), f'{models_dir}/yobs_{col_name}_{model}')


def train_yobs_deprecated(MIMIC_data_combined, MIMICtable_filtered_t0, MIMICraw, MIMICtable, actionbloc, models_dir, nr_reps, col_name):
    icuuniqueids = MIMIC_data_combined['icustay_id'].unique()
    for model in tqdm(range(nr_reps)):
        grp = np.floor(5*np.random.rand(len(icuuniqueids))+1)
        crossval = 1
        trainidx = icuuniqueids[grp != crossval]
        testidx = icuuniqueids[grp == crossval]
        X = torch.FloatTensor(MIMICraw.loc[MIMICtable_filtered_t0['icustay_id'].isin(trainidx)].values)
        A = (torch.tensor(actionbloc.loc[MIMIC_data_combined['icustay_id'].isin(trainidx), 'action_bloc'].values).to(torch.long)-1)/24
        Xtestmimic = torch.FloatTensor(MIMICraw[MIMICtable_filtered_t0['icustay_id'].isin(testidx)].values)
        Atest = (torch.tensor(actionbloc.loc[MIMIC_data_combined['icustay_id'].isin(testidx), 'action_bloc'].values).to(torch.long)-1)/24
        ptid = MIMICtable_filtered_t0.loc[MIMICtable_filtered_t0['icustay_id'].isin(trainidx), 'icustay_id']
        ptidtestmimic = MIMICtable_filtered_t0.loc[MIMICtable_filtered_t0['icustay_id'].isin(testidx), 'icustay_id']
        Y = torch.FloatTensor(MIMIC_data_combined.loc[MIMICtable_filtered_t0['icustay_id'].isin(trainidx), f'{col_name}_t1'].values).unsqueeze(dim=1)
        Ytest = torch.FloatTensor(MIMIC_data_combined.loc[MIMICtable_filtered_t0['icustay_id'].isin(testidx), f'{col_name}_t1'].values).unsqueeze(dim=1)
        Y = (Y - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
        Ytest = (Ytest - MIMICtable[col_name].mean())/MIMICtable[col_name].std()
        train = data_utils.TensorDataset(torch.column_stack((X, A)), Y)
        trainloader = torch.utils.data.DataLoader(train, batch_size=32)
        test = data_utils.TensorDataset(torch.column_stack((Xtestmimic, Atest)), Ytest)
        testloader = torch.utils.data.DataLoader(test, batch_size=32)
        
        net = Net(n_feature=4, n_hidden=10, n_output=1)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.1)
        loss_func = torch.nn.MSELoss()
        
        for epoch in tqdm(range(200)):
            for X, Y in trainloader:
                prediction = net(X)     # input x and predict based on x

                loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            if (epoch+1)%10 == 0:
                with torch.no_grad():
                    test_loss = 0
                    for Xtest, Ytest in testloader:
                        test_loss += loss_func(net(Xtest), Ytest)
                    test_loss = test_loss/len(testloader)
                    wandb.log({'epoch': epoch, f'TL yobs - mdl {model}': test_loss})
        
        torch.save(net.state_dict(), f'{models_dir}/yobs_{model}')


def train_policies(MIMICtable_filtered_t0_tr, MIMICtable_filtered_t0, MIMICraw, models_dir, model):
    MIMICraw_tr = MIMICtable_filtered_t0_tr[['icustay_id']].merge(MIMICraw, on='icustay_id')
    MIMICraw_test = MIMICraw[~MIMICraw['icustay_id'].isin(MIMICtable_filtered_t0_tr['icustay_id'])]
    X = torch.FloatTensor(MIMICraw_tr.drop(columns=['icustay_id']).values)
    Xtestmimic = torch.FloatTensor(MIMICraw_test.drop(columns=['icustay_id']).values)
    Y = torch.tensor(MIMICtable_filtered_t0_tr['A'].values).to(torch.long)-1
    Ytest = torch.tensor(MIMICraw_test[['icustay_id']].merge(MIMICtable_filtered_t0, on=['icustay_id'])['A'].values).to(torch.long)-1
    # Ytest = torch.tensor(MIMICtable_filtered_t0.loc[~MIMICtable_filtered_t0['icustay_id'].isin(MIMICtable_filtered_t0_tr['icustay_id']), 'A'].values).to(torch.long)-1
    train = data_utils.TensorDataset(X, Y)
    trainloader = torch.utils.data.DataLoader(train, batch_size=32)
    test = data_utils.TensorDataset(Xtestmimic, Ytest)
    testloader = torch.utils.data.DataLoader(test, batch_size=32)

    loss_func = torch.nn.CrossEntropyLoss()
    policy = PolicyNetwork(input_dim=X.shape[1], output_dim=25)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)        
    for epoch in range(300):
        for data, label in trainloader:
            prediction = policy(data)     # input x and predict based on x
            loss = loss_func(prediction, label)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                test_loss = 0
                for test_data, test_label in testloader:
                    test_loss += loss_func(policy(test_data), test_label)
                wandb.log({'epoch': epoch, f'TL pol - mdl {model}': test_loss})
            log_policy_accuracy(policy, Xtestmimic, Ytest, epoch, model)
    torch.save(policy.state_dict(), f'{models_dir}/policy_{model}')


def train_policies_deprecated(MIMIC_data_combined, MIMICraw, actionbloc, models_dir, nr_reps):
    icuuniqueids = MIMIC_data_combined['icustay_id'].unique()
    for model in tqdm(range(nr_reps)):
        grp = np.floor(5*np.random.rand(len(icuuniqueids))+1)
        crossval = 1
        trainidx = icuuniqueids[grp != crossval]
        testidx = icuuniqueids[grp == crossval]
        X = torch.FloatTensor(MIMICraw.loc[MIMIC_data_combined['icustay_id'].isin(trainidx)].values)
        Xtestmimic = torch.FloatTensor(MIMICraw[MIMIC_data_combined['icustay_id'].isin(testidx)].values)
        ptid = MIMIC_data_combined.loc[MIMIC_data_combined['icustay_id'].isin(trainidx), 'icustay_id']
        ptidtestmimic = MIMIC_data_combined.loc[MIMIC_data_combined['icustay_id'].isin(testidx), 'icustay_id']
        Y = torch.tensor(actionbloc.loc[MIMIC_data_combined['icustay_id'].isin(trainidx), 'action_bloc'].values).to(torch.long)-1
        Ytest = torch.tensor(actionbloc.loc[MIMIC_data_combined['icustay_id'].isin(testidx), 'action_bloc'].values).to(torch.long)-1
        train = data_utils.TensorDataset(X, Y)
        trainloader = torch.utils.data.DataLoader(train, batch_size=32)
        test = data_utils.TensorDataset(Xtestmimic, Ytest)
        testloader = torch.utils.data.DataLoader(test, batch_size=32)
        
        loss_func = torch.nn.CrossEntropyLoss()
        policy = PolicyNetwork(input_dim=3, output_dim=25)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
        
        for epoch in range(100):
            for data, label in trainloader:
                prediction = policy(data)     # input x and predict based on x
                loss = loss_func(prediction, label)     # must be (1. nn output, 2. target)
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    test_loss = 0
                    for test_data, test_label in testloader:
                        test_loss += loss_func(policy(test_data), test_label)
                    wandb.log({'epoch': epoch, f'TL pol - mdl {model}': test_loss})
                log_policy_accuracy(policy, Xtestmimic, Ytest, epoch, model)
        torch.save(policy.state_dict(), f'{models_dir}/policy_{model}')


def preprocess_data(args):
    logging.info("Preprocessing data")
    extension = 'final_.csv'
    all_filenames = [i for i in glob.glob((args.sim_path + '/*{}').format(extension))]
    pulse_data = pd.concat([pd.read_csv(f) for f in all_filenames ])
    pulse_data['icustay_id'] = pulse_data['id'].astype(int)
    pulse_data = pulse_data.reset_index()
    pulse_rename = {}

    for k, v in column_mappings.items():
        pulse_rename.update({k: f"{v}"})

    pulse_data = pulse_data.rename(columns=pulse_rename)

    MIMICtable = pd.read_csv(f"{args.obs_path}/MIMIC-1hourly-length-5-train.csv")
    MIMICtable['icustay_id'] = MIMICtable['icustay_id'].astype(int)
    MIMICtable = create_action_bins(MIMICtable, args.nra)
    
    MIMICtable_filtered_t0 = MIMICtable[MIMICtable['bloc']==1].reset_index()
    MIMICtable_filtered_t1 = MIMICtable[MIMICtable['bloc']==2][[
        'gender', 'age', 'Weight_kg',
        'icustay_id', 'RR', 'HR', 'SysBP', 'MeanBP', 'DiaBP',
        'SpO2', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride',
        'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca',
        'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'Hb', 'WBC_count',
        'Platelets_count', 'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',
        'Arterial_BE', 'HCO3', 'Arterial_lactate', 'A']].reset_index()

    x_columns = ['gender', 'age', 'Weight_kg'] + cols
    MIMICtable_filtered_t1 = MIMICtable_filtered_t1[MIMICtable_filtered_t1[cols].min(axis=1)>0].reset_index()
    MIMICtable_filtered_t0 = MIMICtable_filtered_t0[MIMICtable_filtered_t0['icustay_id'].isin(MIMICtable_filtered_t1['icustay_id'])].reset_index()

    pulse_data_t0 = pulse_data[pulse_data['index']==1].reset_index(drop=True)
    pulse_data_t1 = pulse_data[pulse_data['index']==2].reset_index(drop=True)
    pulse_data_t2 = pulse_data[pulse_data['index']==3].reset_index(drop=True)
    pulse_data_t3 = pulse_data[pulse_data['index']==4].reset_index(drop=True)

    pulse_data_t0 = MIMICtable[MIMICtable['bloc']==2][['gender', 'age', 'Weight_kg', 'icustay_id', 'A']].merge(pulse_data_t0, on=['icustay_id'])
    pulse_data_t1 = MIMICtable[MIMICtable['bloc']==3][['gender', 'age', 'Weight_kg', 'icustay_id', 'A']].merge(pulse_data_t1, on=['icustay_id'])
    pulse_data_t2 = MIMICtable[MIMICtable['bloc']==4][['gender', 'age', 'Weight_kg', 'icustay_id', 'A']].merge(pulse_data_t2, on=['icustay_id'])
    pulse_data_t3 = MIMICtable[MIMICtable['bloc']==5][['gender', 'age', 'Weight_kg', 'icustay_id', 'A']].merge(pulse_data_t3, on=['icustay_id'])

    pulse_data_t0 = pd.concat([pulse_data_t0, pulse_data_t1, pulse_data_t2])
    pulse_data_t1 = pd.concat([pulse_data_t1, pulse_data_t2, pulse_data_t3])

    if not pulse_data_t0['icustay_id'].equals(pulse_data_t1['icustay_id']):
        raise RuntimeError("Pulse data not contain same icustay_id's at t0 and t1")

    pulse_data_t0['icustay_id'] = np.arange(len(pulse_data_t0))
    pulse_data_t1['icustay_id'] = np.arange(len(pulse_data_t1))

    logging.info('Processing raw data')
    # all 47 columns of interest
    colbin = ['gender']
    colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',\
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',\
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',\
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']
    collog=['SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','output_total','output_1hourly']


    MIMICraw = MIMICtable_filtered_t0[['icustay_id'] + x_columns].copy()

    for col in MIMICraw:
        if col in colbin:
            MIMICraw[col] = MIMICraw[col] - 0.5
        elif col in colnorm:
            cmu = MIMICtable_filtered_t0[col].mean()
            csigma = MIMICtable_filtered_t0[col].std()
            MIMICraw[col] = (MIMICraw[col] - cmu)/csigma
        elif col in collog:
            log_values = np.log(0.1 + MIMICtable_filtered_t0[col])
            dmu = log_values.mean()
            dsigma = log_values.std()
            MIMICraw[col] = (log_values - dmu)/dsigma    

    pulseraw = pulse_data_t0[['icustay_id'] + x_columns].copy()

    for col in pulseraw:
        if col in colbin:
            pulseraw[col] = pulseraw[col] - 0.5
        elif col in colnorm:
            cmu = pulseraw[col].mean()
            csigma = pulseraw[col].std()
            pulseraw[col] = (pulseraw[col] - cmu)/csigma
        elif col in collog:
            log_values = np.log(0.1 + pulseraw[col])
            dmu = log_values.mean()
            dsigma = log_values.std()
            pulseraw[col] = (log_values - dmu)/dsigma
    logging.info('Raw data processed')
    logging.info("Preprocessing done")
    return MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMICtable, MIMICraw, pulseraw, pulse_data_t0, pulse_data_t1


def preprocess_data_deprecated(args):
    logging.info("Preprocessing data")
    extension = 'final_.csv'
    col_name = args.col_name
    all_filenames = [i for i in glob.glob((args.sim_path + '/*{}').format(extension))]
    pulse_data = pd.concat([pd.read_csv(f) for f in all_filenames ])
    pulse_data['icustay_id'] = pulse_data['id'].astype(int)
    pulse_data = pulse_data.reset_index()
    pulse_rename = {}

    for k, v in column_mappings.items():
        pulse_rename.update({k: f"{v}"})

    pulse_data = pulse_data.rename(columns=pulse_rename)

    MIMICtable = pd.read_csv(f"{args.obs_path}/MIMIC-1hourly-length-5-train.csv")
    MIMICtable['icustay_id'] = MIMICtable['icustay_id'].astype(int)
    
    MIMICtable_filtered_t0 = MIMICtable[MIMICtable['bloc']==1].reset_index()
    MIMICtable_filtered_t1 = MIMICtable[MIMICtable['bloc']==2][[
        'icustay_id', 'RR', 'HR', 'SysBP', 'MeanBP', 'DiaBP',
        'SpO2', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride',
        'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca',
        'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'Hb', 'WBC_count',
        'Platelets_count', 'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',
        'Arterial_BE', 'HCO3', 'Arterial_lactate']].reset_index()

    MIMICtable_filtered_t0 = MIMICtable_filtered_t0.rename(columns={f'{col_name}':f'{col_name}_t0'})
    MIMICtable_filtered_t1 = MIMICtable_filtered_t1[MIMICtable_filtered_t1[col_name]>0].reset_index()
    MIMICtable_filtered_t1 = MIMICtable_filtered_t1.rename(columns={f'{col_name}':f'{col_name}_t1'})
    MIMICtable_filtered_t0 = MIMICtable_filtered_t0[MIMICtable_filtered_t0['icustay_id'].isin(MIMICtable_filtered_t1['icustay_id'])].reset_index()
    MIMIC_data_combined = MIMICtable_filtered_t0[['gender', 'age', 'icustay_id', f'{col_name}_t0']].merge(MIMICtable_filtered_t1[['icustay_id', f'{col_name}_t1']], on=['icustay_id'])
    MIMIC_data_combined.head()

    pulse_data_t0 = pulse_data[pulse_data['index']==1].reset_index(drop=True)
    pulse_data_t1 = pulse_data[pulse_data['index']==2].reset_index(drop=True)
    pulse_data_t0 = pulse_data_t0.rename(columns={f'{col_name}':f'{col_name}_t0'})
    pulse_data_t1 = pulse_data_t1.rename(columns={f'{col_name}':f'{col_name}_t1'})
    pulse_data_combined = pulse_data_t0[['icustay_id', f'{col_name}_t0']].merge(pulse_data_t1[['icustay_id', f'{col_name}_t1']], on=['icustay_id'])
    pulse_data_combined = MIMICtable_filtered_t0[['gender', 'age', 'icustay_id']].merge(pulse_data_combined[['icustay_id', f'{col_name}_t0', f'{col_name}_t1']], on=['icustay_id'])

    logging.info('Processing raw data')
    colbin = ['gender']
    colnorm=['age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',\
        'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',\
        'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',\
        'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance']

    MIMICraw = MIMIC_data_combined[['gender', 'age', f'{col_name}_t0']].copy()

    for col in MIMICraw:
        if col in colbin:
            MIMICraw[col] = MIMICraw[col] - 0.5
        else:
            cmu = MIMICraw[col].mean()
            csigma = MIMICraw[col].std()
            MIMICraw[col] = (MIMICraw[col] - cmu)/csigma
    pulseraw = pulse_data_combined[['gender', 'age', f'{col_name}_t0']].copy()

    for col in pulseraw:
        if col in colbin:
            pulseraw[col] = pulseraw[col] - 0.5
        else:
            cmu = pulseraw[col].mean()
            csigma = pulseraw[col].std()
            pulseraw[col] = (pulseraw[col] - cmu)/csigma
    logging.info('Raw data processed')
    logging.info("Preprocessing done")
    return MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMIC_data_combined, MIMICtable, pulse_data_combined, MIMICraw, pulseraw


def create_action_bins_deprecated(MIMICtable_filtered_t0, nra):
    logging.info('Creating action bins')
    nact = nra**2
    input_1hourly_nonzero = MIMICtable_filtered_t0.loc[MIMICtable_filtered_t0['input_1hourly']>0, 'input_1hourly']
    iol_ranked = rankdata(input_1hourly_nonzero)/len(input_1hourly_nonzero) # excludes zero fluid (will be action 1)
    iof = np.floor((iol_ranked + 0.2499999999)*4) # converts iv volume in 4 actions
    io = np.ones(len(MIMICtable_filtered_t0)) # array of ones, by default
    io[MIMICtable_filtered_t0['input_1hourly']>0] = iof + 1 # where more than zero fluid given: save actual action
    vc = MIMICtable_filtered_t0['max_dose_vaso'].copy()
    vc_nonzero = MIMICtable_filtered_t0.loc[MIMICtable_filtered_t0['max_dose_vaso']!=0, 'max_dose_vaso']
    vc_ranked = rankdata(vc_nonzero)/len(vc_nonzero)
    vcf = np.floor((vc_ranked + 0.2499999999)*4) # converts to 4 bins
    vcf[vcf==0] = 1
    vc[vc!=0] = vcf + 1
    vc[vc==0] = 1
    # median dose of drug in all bins
    ma1 = [MIMICtable_filtered_t0.loc[io==1, 'input_1hourly'].median(), MIMICtable_filtered_t0.loc[io==2, 'input_1hourly'].median(), MIMICtable_filtered_t0.loc[io==3, 'input_1hourly'].median(), MIMICtable_filtered_t0.loc[io==4, 'input_1hourly'].median(), MIMICtable_filtered_t0.loc[io==5, 'input_1hourly'].median()]
    ma2 = [MIMICtable_filtered_t0.loc[vc==1, 'max_dose_vaso'].median(), MIMICtable_filtered_t0.loc[vc==2, 'max_dose_vaso'].median(), MIMICtable_filtered_t0.loc[vc==3, 'max_dose_vaso'].median(), MIMICtable_filtered_t0.loc[vc==4, 'max_dose_vaso'].median(), MIMICtable_filtered_t0.loc[vc==5, 'max_dose_vaso'].median()]
    med = pd.DataFrame(data={'IV':io, 'VC': vc})
    med = med.astype({'IV': 'int32', 'VC': 'int32'})
    uniqueValues = med.drop_duplicates().reset_index(drop=True)
    uniqueValueDoses = pd.DataFrame()
    for index, row in uniqueValues.iterrows():
        uniqueValueDoses.at[index, 'IV'], uniqueValueDoses.at[index, 'VC'] = ma1[row['IV']-1], ma2[row['VC']-1]

    actionbloc = pd.DataFrame()
    for index, row in med.iterrows():
        actionbloc.at[index, 'action_bloc'] = uniqueValues.loc[(uniqueValues['IV'] == row['IV']) & (uniqueValues['VC'] == row['VC'])].index.values[0]+1
    actionbloc['icustay_id'] = MIMICtable_filtered_t0['icustay_id']
    actionbloc = actionbloc.astype({'action_bloc':'int32'})
    logging.info('Action bins created')
    return actionbloc


def create_action_bins(MIMICtable, nra):
    logging.info('Creating action bins')
    nact = nra**2
    input_1hourly_nonzero = MIMICtable.loc[MIMICtable['input_1hourly']>0, 'input_1hourly']
    iol_ranked = rankdata(input_1hourly_nonzero)/len(input_1hourly_nonzero) # excludes zero fluid (will be action 1)
    iof = np.floor((iol_ranked + 0.2499999999)*4) # converts iv volume in 4 actions
    io = np.ones(len(MIMICtable)) # array of ones, by default
    io[MIMICtable['input_1hourly']>0] = iof + 1 # where more than zero fluid given: save actual action
    vc = MIMICtable['max_dose_vaso'].copy()
    vc_nonzero = MIMICtable.loc[MIMICtable['max_dose_vaso']!=0, 'max_dose_vaso']
    vc_ranked = rankdata(vc_nonzero)/len(vc_nonzero)
    vcf = np.floor((vc_ranked + 0.2499999999)*4) # converts to 4 bins
    vcf[vcf==0] = 1
    vc[vc!=0] = vcf + 1
    vc[vc==0] = 1
    # median dose of drug in all bins
    ma1 = [MIMICtable.loc[io==1, 'input_1hourly'].median(), MIMICtable.loc[io==2, 'input_1hourly'].median(), MIMICtable.loc[io==3, 'input_1hourly'].median(), MIMICtable.loc[io==4, 'input_1hourly'].median(), MIMICtable.loc[io==5, 'input_1hourly'].median()]
    ma2 = [MIMICtable.loc[vc==1, 'max_dose_vaso'].median(), MIMICtable.loc[vc==2, 'max_dose_vaso'].median(), MIMICtable.loc[vc==3, 'max_dose_vaso'].median(), MIMICtable.loc[vc==4, 'max_dose_vaso'].median(), MIMICtable.loc[vc==5, 'max_dose_vaso'].median()]
    med = pd.DataFrame(data={'IV':io, 'VC': vc})
    med = med.astype({'IV': 'int32', 'VC': 'int32'})
    uniqueValues = med.drop_duplicates().reset_index(drop=True)
    uniqueValueDoses = pd.DataFrame()
    for index, row in uniqueValues.iterrows():
        uniqueValueDoses.at[index, 'IV'], uniqueValueDoses.at[index, 'VC'] = ma1[row['IV']-1], ma2[row['VC']-1]

    actionbloc = pd.DataFrame()
    for index, row in med.iterrows():
        actionbloc.at[index, 'action_bloc'] = uniqueValues.loc[(uniqueValues['IV'] == row['IV']) & (uniqueValues['VC'] == row['VC'])].index.values[0]+1
    actionbloc = actionbloc.astype({'action_bloc':'int32'})

    logging.info('Action bins created')
    MIMICtable['A'] = actionbloc['action_bloc']
    return MIMICtable