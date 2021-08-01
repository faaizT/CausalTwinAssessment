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
from tqdm import tqdm
import wandb

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

def main(args):
    MIMICtable_filtered_t0, MIMICtable_filtered_t1, MIMIC_data_combined, pulse_data_combined, MIMICraw, pulseraw = preprocess_data(args)
    actionbloc = create_action_bins(MIMICtable_filtered_t0, args.nra)
    train_policies(MIMIC_data_combined, MIMICraw, actionbloc, args.models_dir, args.nr_reps)
    train_yobs(MIMIC_data_combined, MIMICtable_filtered_t0, MIMICraw, actionbloc, args.models_dir, args.nr_reps, args.col_name)
    train_yminmax(MIMIC_data_combined, MIMICtable_filtered_t0, MIMICraw, actionbloc, args.models_dir, args.nr_reps, args.col_name)
    train_ysim(MIMIC_data_combined, pulse_data_combined, pulseraw, actionbloc, args.models_dir, args.nr_reps, args.col_name)


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

    wandb.init(project="Manski-Regression", name=f"{args.sim_name}-{args.col_name}")

    main(args)

