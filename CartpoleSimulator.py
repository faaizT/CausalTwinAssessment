import argparse
import logging
import pandas as pd
import numpy as np
import os
import glob
import re
import torch
import torch.utils.data as data_utils
from sklearn.utils import resample
from cartpole_sim.utils.HelperFunctions import *
from tqdm import tqdm
import wandb
from pathlib import Path
from utils import str2bool


def main(args):
    # Make results dir if does not exist
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    df_partial, sim_data, obs_data_train = load_and_preprocess_data(args)
    obs_test_size = int(len(obs_data_train) / 5)
    obs_train_size = len(obs_data_train) - obs_test_size

    obs_bootstrap = resample(obs_data_train, n_samples=obs_train_size)

    train_policies(
        df_partial, obs_data_train, obs_bootstrap, args.models_dir, args.model, args.use_bootstrap
    )
    train_yobs(df_partial, obs_data_train, obs_bootstrap, args.models_dir, args.model, args.use_bootstrap)
    for quantile in args.quantiles:
        train_yminmax(
            df_partial,
            obs_data_train,
            obs_bootstrap,
            quantile,
            args.models_dir,
            args.model,
            args.use_bootstrap,
        )
    train_ysim(
        df_partial,
        sim_data,
        args.models_dir,
        args.model,
        false_sim=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files_dir",
        help="path to data directory",
        default="/data/ziz/taufiq/export-dir",
    )
    parser.add_argument(
        "--models_dir",
        help="Directory to save trained models",
        type=str,
        default="/data/ziz/not-backed-up/taufiq/Cartpole/dry-run",
    )
    parser.add_argument("--model", help="Model number", type=int, default=0)
    parser.add_argument(
        "--quantiles",
        help="Worst case quantiles",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.975, 0.99, 0.999],
    )
    parser.add_argument(
        "--perturbation",
        help="Perturbation of False Simulator",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--noise_std",
        help="Std of noise added to fake simulator",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--n_obs",
        help="Number of observational data for estimating Manski's bounds",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--use_bootstrap",
        help="Use bootstrap resampling when training the quantities in bounds",
        type=str2bool,
        default="True",
    )

    args = parser.parse_args()

    wandb.init(project="Carpole-Simulator", name=f"Model-{args.model}")

    main(args)