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
from manski_bounds.utils.HelperFunctions import *
from sklearn.utils import resample
from tqdm import tqdm
import wandb

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

def main_deprecated(args):
    MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMIC_data_combined, MIMICtable, pulse_data_combined, MIMICraw, pulseraw = preprocess_data(args)
    actionbloc = create_action_bins(MIMICtable_filtered_t0, args.nra)
    train_policies(MIMIC_data_combined, MIMICraw, actionbloc, args.models_dir, args.nr_reps)
    train_yobs(MIMIC_data_combined, MIMICtable_filtered_t0, MIMICraw, MIMICtable, actionbloc, args.models_dir, args.nr_reps, args.col_name)
    train_yminmax(MIMIC_data_combined, MIMICtable_filtered_t0, MIMICraw, MIMICtable, actionbloc, args.models_dir, args.nr_reps, args.col_name)
    train_ysim(MIMIC_data_combined, MIMICtable, pulse_data_combined, pulseraw, actionbloc, args.models_dir, args.nr_reps, args.col_name)


def main(args):
    MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMICtable, MIMICraw, pulseraw, pulse_data_t0, pulse_data_t1 = preprocess_data(args)
    test_size = int(len(MIMICtable_filtered_t0)/5)
    train_size = len(MIMICtable_filtered_t0) - test_size
    MIMICtable_filtered_t0_tr = resample(MIMICtable_filtered_t0, n_samples=train_size)
    MIMICtable_filtered_t1_tr = MIMICtable_filtered_t0_tr[['icustay_id']].merge(MIMICtable_filtered_t1, on='icustay_id')
    actionbloc = create_action_bins(MIMICtable_filtered_t0, args.nra)
    train_policies(MIMICtable_filtered_t0_tr, MIMICraw, actionbloc, args.models_dir, model=0)
    train_yobs(MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMICtable_filtered_t0_tr, MIMICtable_filtered_t1_tr, MIMICraw, MIMICtable, actionbloc, args.models_dir, args.col_name, model=0)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--col_name", help="Column name to run hypothesis tests for", type=str, required=True)
    parser.add_argument("--sim_name", help="Simulator name (pulse/biogears)", type=str, default="pulse")
    parser.add_argument("--obs_path", help="path to observational data directory", default="/data/ziz/taufiq/export-dir")
    parser.add_argument("--sim_path", help="path to sim data directory", default="/data/ziz/taufiq/pulse-data-5-step")
    parser.add_argument("--models_dir", help="Directory to save trained models", required=True)
    parser.add_argument("--nr_reps", help="Number of models to be trained", default=2, type=int)
    parser.add_argument("--nra", help="Number of action bins", default=5, type=int)
    args = parser.parse_args()

    wandb.init(project="Manski-Regression", name=f"{args.sim_name}-{args.col_name}-dryrun")

    main(args)
