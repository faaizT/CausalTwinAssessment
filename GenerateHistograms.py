import argparse
import logging
import pandas as pd
import numpy as np
import os
import glob
import re
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from bareinboim_bounds.Utils import *
from utils import str2bool
import matplotlib.pyplot as plt


def get_hoeffding_bounds(index, row, alpha=0.05):
    ylo, yup = row['y_lo'], row['y_up']
    nobs, nsim = row['n_obs'], row['n_sim']
    delta_obs = (yup - ylo)*np.sqrt(1/(2*nobs)*np.log(4/alpha))
    delta_sim = (yup - ylo)*np.sqrt(1/(2*nsim)*np.log(4/alpha))
    ylb_lb = np.mean(row['Y_lb_mean']) - delta_obs
    ylb_ub = np.mean(row['Y_lb_mean']) + delta_obs
    yub_lb = np.mean(row['Y_ub_mean']) - delta_obs
    yub_ub = np.mean(row['Y_ub_mean']) + delta_obs
    ysim_lb = np.mean(row['ysim_values']) - delta_sim
    ysim_ub = np.mean(row['ysim_values']) + delta_sim
    return [ylb_lb , ylb_ub], [ysim_lb, ysim_ub], [yub_lb, yub_ub]


def generate_histograms_bootstrapping(index, row, results_directory):
    fig, axs = plt.subplots(1, 2, figsize=(18,6), sharex=True)
    a = axs[1].hist(row['Y_lb_mean'], label='$Q_{lo}$', density=True, alpha=0.4, bins='auto')
    a = axs[1].hist(row['Y_ub_mean'], label='$Q_{up}$', density=True, alpha=0.4, bins='auto')
    a = axs[1].hist(row['Sim_exp_y'], label='$Q^{twin}$', density=True, alpha=0.4, bins='auto')
    
    
    axs[0].hist(row['yobs_values'], label='$Y$ values', density=True, alpha=0.4, bins='auto')
    axs[0].hist(row['ysim_values'], label='$Y^{twin}$ values', density=True, alpha=0.4, bins='auto')
    axs[0].axvline(row['y_lo'], linestyle='--', color='black', label='$y_{lo}$ & $y_{up}$')
    axs[0].axvline(x=row['y_up'], linestyle='--', color='black')
    p_lb, p_ub = row['p_lb'], row['p_ub']
    plt.suptitle(f'p_lb = {p_lb} | p_ub = {p_ub} | n_sim = {row["n_sim"]} | n_obs = {row["n_obs"]}', fontsize=14)
    rejected = (p_lb < 0.05/4) or (p_ub < 0.05/4)
    figtitle = f"{row['col']}_hyp_{index}"
    
    ylb_interval, ysim_interval, yub_interval = get_hoeffding_bounds(index, row)
    max_ylim = axs[1].get_ylim()[1]
    axs[1].fill_betweenx([0, max_ylim], ylb_interval[0], ylb_interval[1], color='g', alpha=0.1, label='Hoeffding CIs')
    axs[1].fill_betweenx([0,max_ylim], ysim_interval[0], ysim_interval[1], color='g', alpha=0.1,)
    axs[1].fill_betweenx([0,max_ylim], yub_interval[0], yub_interval[1], color='g', alpha=0.1,)
    axs[1].axvline(row['y_lo'], linestyle='--', color='black', label='$y_{lo}$ & $y_{up}$')
    axs[1].axvline(x=row['y_up'], linestyle='--', color='black')

    axs[1].legend(fontsize=14, )
    axs[0].legend(fontsize=14, )    
    os.makedirs(f"{results_directory}/images/{row['col']}_rej{rejected}", exist_ok=True)
    plt.savefig(f"{results_directory}/images/{row['col']}_rej{rejected}/{figtitle}")    
    plt.close()


def main(args):
    data = {'columns':[], 'total_hypotheses':[], 'rejected_hypotheses': [], 'percentage_of_rejected_hyp': []}
    p_values = pd.read_csv(f"{args.results_dir}/p_values_{args.col_name}_hoeffFalse.csv",  converters={'actions': eval, 'x_t': eval, 'Y_lb_mean': eval, 'Y_ub_mean': eval, 'Sim_exp_y': eval, 'yobs_values': eval, 'ysim_values': eval})
    p_values = p_values.loc[p_values['t']>0]
    p_values['col'] = args.col_name
    os.makedirs(f"{args.image_export_dir}/images/{args.col_name}_rejTrue", exist_ok=True)
    os.makedirs(f"{args.image_export_dir}/images/{args.col_name}_rejFalse", exist_ok=True)
    for index, row in p_values.iterrows():
        generate_histograms_bootstrapping(index, row, args.image_export_dir)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--col_name",
        help="Column name to run hypothesis tests for",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--results_dir", 
        help="Location of saved results data", 
        default="/data/ziz/not-backed-up/taufiq/HypothesisTesting/cts_outcome_2bins_yminmax_8020quantiles_d0_splitting/perc",
    )
    parser.add_argument(
        "--image_export_dir", 
        help="Location to save images", 
        default="/data/ziz/not-backed-up/taufiq/HypothesisTesting/cts_outcome_2bins_yminmax_8020quantiles_d0_splitting/histograms",
    )
    args = parser.parse_args()
    main(args)
