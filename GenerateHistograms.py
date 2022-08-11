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
import seaborn as sns
from statsmodels.stats.multitest import multipletests


column_names_dict = {
    'Albumin': 'Albumin Blood Concentration',
    'paCO2': 'Arterial $CO_2$ Pressure',
    'paO2': 'Arterial $O_2$ Pressure',
    'HCO3': 'Bicarbonate Blood Concentration',
    'Arterial_pH': 'Arterial pH',
    'Arterial_lactate': 'Lactate Blood Concentration',
    'Calcium': 'Calcium Blood Concentration',
    'Chloride': 'Chloride Blood Concentration',
    'Creatinine': 'Creatinine Blood Concentration',
    'DiaBP': 'Diastolic Arterial Pressure',
    'SysBP': 'Systolic Arterial Pressure',
    'Glucose': 'Glucose Blood Concentration',
    'MeanBP': 'Mean Arterial Pressure',
    'Potassium': 'Potassium Blood Concentration',
    'RR': 'Respiration Rate',
    'Temp_C': 'Skin Temperature',
    'Sodium': 'Sodium Blood Concentration',
    'WBC_count': 'White Blood Cell Count',
    'HR': 'Heart Rate'
}


column_names_unit = {
    'Albumin': 'Albumin Blood Concentration (mg/L)',
    'paCO2': 'Arterial $CO_2$ Pressure (mmHg)',
    'paO2': 'Arterial $O_2$ Pressure (mmHg)',
    'HCO3': 'Bicarbonate Blood Concentration (mg/L)',
    'Arterial_pH': 'Arterial pH',
    'Arterial_lactate': 'Lactate Blood Concentration (mg/L)',
    'Calcium': 'Calcium Blood Concentration (mg/L)',
    'Chloride': 'Chloride Blood Concentration (mg/L)',
    'Creatinine': 'Creatinine Blood Concentration (mg/L)',
    'DiaBP': 'Diastolic Arterial Pressure (mmHg)',
    'SysBP': 'Systolic Arterial Pressure (mmHg)',
    'Glucose': 'Glucose Blood Concentration (mg/L)',
    'MeanBP': 'Mean Arterial Pressure (mmHg)',
    'Potassium': 'Potassium Blood Concentration (mg/L)',
    'RR': 'Respiration Rate (1/min)',
    'Temp_C': 'Skin Temperature (C)',
    'Sodium': 'Sodium Blood Concentration (mg/L)',
    'WBC_count': 'White Blood Cell Count (ct/uL)',
    'HR': 'Heart Rate (1/min)'
}



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


def generate_histograms_bootstrapping(index, row, results_directory, total_hypotheses):
    fig, axs = plt.subplots(1, 2, figsize=(18,8))
    plt.style.use('fivethirtyeight')
    a = axs[1].hist(row['Y_lb_mean'], label='$Q_{lo}$', density=True, alpha=0.4, bins='auto', color='blue')
    a = axs[1].hist(row['Y_ub_mean'], label='$Q_{up}$', density=True, alpha=0.4, bins='auto', color=sns.color_palette("mako", 10)[5])
    a = axs[1].hist(row['Sim_exp_y'], label='$Q^{twin}$', density=True, alpha=0.4, bins='auto', color='red')
    
    
    axs[0].hist(row['yobs_values'], label='$Y$ values', density=True, alpha=0.4, bins='auto', color='blue')
    axs[0].hist(row['ysim_values'], label='$Y^{twin}$ values', density=True, alpha=0.4, bins='auto', color='red')
    # axs[0].axvline(row['y_lo'], linestyle='--', color='black', label='$y_{lo}$ & $y_{up}$')
    # axs[0].axvline(x=row['y_up'], linestyle='--', color='black')
    p_lb, p_ub = row['p_lb'], row['p_ub']
    plt.suptitle(f'p_lb = {p_lb} | p_ub = {p_ub} | n_sim = {row["n_sim"]} | n_obs = {row["n_obs"]}', fontsize=14)
    rejected = (row['rejected_holms_lb']) or (row['rejected_holms_ub'])
    figtitle = f"{row['col']}_hyp_{index}"
    
    # ylb_interval, ysim_interval, yub_interval = get_hoeffding_bounds(index, row, alpha=0.05/total_hypotheses)
    # max_ylim = axs[1].get_ylim()[1]
    # axs[1].fill_betweenx([0, max_ylim], ylb_interval[0], ylb_interval[1], color='g', alpha=0.1, label='Hoeffding CIs')
    # axs[1].fill_betweenx([0,max_ylim], ysim_interval[0], ysim_interval[1], color='g', alpha=0.1,)
    # axs[1].fill_betweenx([0,max_ylim], yub_interval[0], yub_interval[1], color='g', alpha=0.1,)
    axs[1].axvline(row['y_lo'], linestyle='--', color='black', label='$y_{lo}$ & $y_{up}$')
    axs[1].axvline(x=row['y_up'], linestyle='--', color='black')
    axs[0].set_xlabel(column_names_unit[row['col']], fontsize=14)
    axs[1].set_xlabel(column_names_unit[row['col']], fontsize=14)
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[0].grid(False)
    axs[1].grid(False)

    min_ylim = axs[1].get_ylim()[0]
    max_ylim = axs[1].get_ylim()[1]
    min_xlim = axs[1].get_xlim()[0]
    max_xlim = axs[1].get_xlim()[1]
    axs[1].axhline(min_ylim, color='black')
    axs[1].axvline(min_xlim, color='black')
    axs[1].axhline(max_ylim, color='black')
    axs[1].axvline(max_xlim, color='black')
    
    
    min_ylim = axs[0].get_ylim()[0]
    max_ylim = axs[0].get_ylim()[1]
    min_xlim = axs[0].get_xlim()[0]
    max_xlim = axs[0].get_xlim()[1]
    axs[0].axhline(min_ylim, color='black')
    axs[0].axvline(min_xlim, color='black')
    axs[0].axhline(max_ylim, color='black')
    axs[0].axvline(max_xlim, color='black')

    axs[1].legend(fontsize=14, )
    axs[0].legend(fontsize=14, )    
    plt.savefig(f"{results_directory}/histograms/{row['col']}_rej{rejected}/{figtitle}")    
    plt.close()


def generate_longitudinal_plots(index, row, p_values, results_directory):
    a = row['actions']
    age = row['age']
    gender = row['gender']
    x_t = row['x_t']
    
    y_lb = []
    y_ub = []
    y_twin = []
    y_twin_lb = []
    y_twin_ub = []
    y_obs = []
    y_obs_lb = []
    y_obs_ub = []
    y_obs_std = []
    p_lb_vals = []
    p_ub_vals = []
    
    for t in range(1, 5):
        p_values_filtered = p_values[(p_values['gender'] == gender) & (p_values['age'] == age) & (p_values['actions'] == a[:t]) & (p_values['x_t'] == x_t[:t])  & (p_values['t'] == t)]
        p_lb_vals.append(p_values_filtered['p_lb'].values[0])
        p_ub_vals.append(p_values_filtered['p_ub'].values[0])
        y_lb.append(np.quantile(p_values_filtered['Y_lb_mean'].values[0], 0.05/4))
        y_ub.append(np.quantile(p_values_filtered['Y_ub_mean'].values[0], 1 - 0.05/4))
        y_twin.append(np.mean(p_values_filtered['Sim_exp_y'].values[0]))
        y_twin_lb.append(np.quantile(p_values_filtered['Sim_exp_y'].values[0], 0.05/4))
        y_twin_ub.append(np.quantile(p_values_filtered['Sim_exp_y'].values[0], 1 - 0.05/4))
        y_obs.append(np.mean(p_values_filtered['Exp_y'].values[0]))
        y_obs_std.append(np.std(p_values_filtered['yobs_values'].values[0])/len(p_values_filtered['yobs_values'].values[0]))
        y_obs_lb.append(np.quantile(p_values_filtered['Exp_y'].values[0], 0.05/4))
        y_obs_ub.append(np.quantile(p_values_filtered['Exp_y'].values[0], 1 - 0.05/4))
    
    x = [1, 2, 3, 4]
    plt.style.use('fivethirtyeight')
    fig, axis = plt.subplots(1, 1, figsize=(18,8))
    axis.plot(x, y_obs, color = 'purple', label="$\hat{Q}^{obs}$")
    axis.fill_between(x, y_obs_lb, y_obs_ub, label="$\hat{Q}^{obs}$ uncertainty", color='purple', alpha = 0.2)
    axis.plot(x, y_twin, color = sns.color_palette("mako", 10)[5], label="$\hat{Q}^{twin}$")
    axis.fill_between(x, y_twin_lb, y_twin_ub, label="$\hat{Q}^{twin}$ uncertainty", color=sns.color_palette("mako", 10)[5], alpha = 0.2)
    axis.fill_between(x, y_lb, y_ub, color='r', alpha=0.1, label='$[\hat{Q}_{lo}, \hat{Q}_{up}]$ interval')
    axis.set_xlabel('Time (hr)', fontsize=13)
    axis.set_ylabel(column_names_unit[row['col']], fontsize=13)
    p_lb, p_ub = row['p_lb'], row['p_ub']
    min_p_value = np.min((np.min(p_lb_vals), np.min(p_ub_vals)))
    rejected = (row['rejected_holms_lb']) or (row['rejected_holms_ub'])

    plt.suptitle(f'min $p$-value = {min_p_value} | $n_{{sim}}$ = {row["n_sim"]} | $n_{{obs}}$ = {row["n_obs"]}', fontsize=16)
    plt.legend()

    figtitle = f"{row['col']}_hyp_{index}"
    plt.savefig(f"{results_directory}/longitudinal/{row['col']}_rej{rejected}/{figtitle}")  
    plt.close()
    


def main(args):

    p_vals = pd.DataFrame()
    for col in column_names_dict:
        if os.path.exists(f"{args.results_dir}/p_values_{args.col_name}_hoeffFalse.csv"):
            p_values = pd.read_csv(f"{args.results_dir}/p_values_{args.col_name}_hoeffFalse.csv", converters={'actions': eval, 'x_t': eval, 'Y_lb_mean': eval, 'Y_ub_mean': eval, 'Sim_exp_y': eval, 'yobs_values': eval, 'ysim_values': eval})
            if len(p_values)>0:
                p_values = p_values.loc[p_values['t'] > 0]
                p_values['col'] = col
                p_values['p_lb'] = p_values['p_lb'].clip(0,1)
                p_values['p_ub'] = p_values['p_ub'].clip(0,1)
                p_vals = pd.concat([p_vals, p_values], axis=0)
    
    p_vals['rejected_bonf_lb'] = p_vals['p_lb']<0.05/4/len(p_vals)
    p_vals['rejected_bonf_ub'] = p_vals['p_ub']<0.05/4/len(p_vals)
    p_vals['rejected_holms_lb'] = multipletests(p_vals['p_lb'], alpha=0.05/4, method='holm')[0]
    p_vals['rejected_holms_ub'] = multipletests(p_vals['p_ub'], alpha=0.05/4, method='holm')[0]

    data = {'columns':[], 'total_hypotheses':[], 'rejected_hypotheses': [], 'percentage_of_rejected_hyp': []}
    p_values = p_vals[p_vals['col']==args.col_name]
    p_values['Exp_y'] = p_values['Exp_y'].apply(lambda val: eval(val) if 'nan' not in val else [])
    p_values = p_values[p_values['Exp_y'].apply(lambda x: len(x)) > 0]
    p_values['col'] = args.col_name
    os.makedirs(f"{args.image_export_dir}/histograms/{args.col_name}_rejTrue", exist_ok=True)
    os.makedirs(f"{args.image_export_dir}/histograms/{args.col_name}_rejFalse", exist_ok=True)
    os.makedirs(f"{args.image_export_dir}/longitudinal/{args.col_name}_rejTrue", exist_ok=True)
    os.makedirs(f"{args.image_export_dir}/longitudinal/{args.col_name}_rejFalse", exist_ok=True)
    for index, row in p_values.iterrows():
        generate_histograms_bootstrapping(index, row, args.image_export_dir, total_hypotheses=len(p_vals))

    p_values_complete_trajecs = p_values[(p_values['t']==4)]
    for index, row in p_values_complete_trajecs.iterrows():
        generate_longitudinal_plots(index, row, p_values, args.image_export_dir)
        


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
        default="/data/ziz/not-backed-up/taufiq/HypothesisTesting/cts_outcome_2bins_yminmax_8020quantiles_d0_splitting_t5/perc",
    )
    parser.add_argument(
        "--image_export_dir", 
        help="Location to save images", 
        default="/data/ziz/not-backed-up/taufiq/HypothesisTesting/cts_outcome_2bins_yminmax_8020quantiles_d0_splitting_t5/images",
    )
    args = parser.parse_args()
    main(args)
