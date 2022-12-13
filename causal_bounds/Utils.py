import pandas as pd
import numpy as np
import logging
from ast import literal_eval
from sklearn.utils import resample
import scipy.stats as st
from scipy.stats import rankdata
from sklearn.cluster import KMeans
import scipy

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_trajec_actions_df(trajectories):
    trajec_actions = trajectories.loc[trajectories['t']==0].reset_index(drop=True)
    trajec_actions_t1 = trajectories.loc[trajectories['t']==1].reset_index(drop=True)
    trajec_actions_t2 = trajectories.loc[trajectories['t']==2].reset_index(drop=True)
    trajec_actions_t3 = trajectories.loc[trajectories['t']==3].reset_index(drop=True)
    trajec_actions_t1.rename(columns={'x_t':'x_1', 'A_t':'A_1'}, inplace=True)
    trajec_actions_t2.rename(columns={'x_t':'x_2', 'A_t':'A_2'}, inplace=True)
    trajec_actions_t3.rename(columns={'x_t':'x_3', 'A_t':'A_3'}, inplace=True)
    trajec_actions = pd.merge(left=trajec_actions, right=trajec_actions_t1[['x_1', 'A_1']], left_index=True, right_index=True)
    trajec_actions = pd.merge(left=trajec_actions, right=trajec_actions_t2[['x_2', 'A_2']], left_index=True, right_index=True)
    trajec_actions = pd.merge(left=trajec_actions, right=trajec_actions_t3[['x_3', 'A_3']], left_index=True, right_index=True)
    trajec_actions.loc[:,'x_t'] = trajec_actions.loc[:,['x_t', 'x_1', 'x_2', 'x_3']].apply(lambda x: list(x), axis=1)
    trajec_actions.loc[:,'actions'] = trajec_actions.loc[:,['A_t', 'A_1', 'A_2', 'A_3']].apply(lambda x: list(x), axis=1)
    trajec_actions.drop(columns=['A_t', 'x_1', 'A_1', 'x_2', 'A_2', 'x_3', 'A_3'], inplace=True)
    return trajec_actions


def find_elements(series, element):
    return series.apply(lambda x: literal_eval(str(x)) == element)


def find_elements_starting_with(series, element):
    return series.apply(lambda x: literal_eval(str(x))[:len(element)] == element)


def find_elements_containing(series, element):
    return series.apply(lambda x: literal_eval(str(element)) in literal_eval(str(x)))

def xn_in_bn(actions_of_interest, action_taken, x_trajec_of_interest, x_trajec, t):
    for i in range(min(t, len(actions_of_interest))):
        if actions_of_interest[i] != action_taken[i]:
            break
    return x_trajec_of_interest[:i+1] == x_trajec[:i+1]

def filter_dataset_to_get_d0(x_trajec_of_interest, gender, age, obs_data, actions_of_interest, t):
    obs_data_filtered = obs_data.loc[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age)]
    datapoint_in_d0 = obs_data_filtered.apply(lambda x: xn_in_bn(actions_of_interest, x['actions'], x_trajec_of_interest, x['x_t'], t), axis=1)
    obs_data_filtered = obs_data_filtered.loc[datapoint_in_d0]
    return obs_data_filtered


def bootstrap_distribution_causal_bounds(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=4):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+1], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+1], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(lambda x: tuple(x[:i]))
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(lambda x: tuple(x[:i]))
    df = pd.DataFrame()
    real_filtered_before_pruning = filter_dataset_to_get_d0(x_trajec, gender, age, obs_data, action, i)
    withheld_real_data = real_filtered_before_pruning.sample(frac=0.05, replace=False, random_state=1)
    real_filtered_before_pruning = real_filtered_before_pruning.drop(withheld_real_data.index)
    max_y, min_y = round(withheld_real_data[col].quantile(0.80), 2), round(withheld_real_data[col].quantile(0.20), 2)
    if max_y - min_y < 1e-6:
        max_y = min_y + 1
    sim_filtered_before_pruning = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec[:i])) & (sim['actions_tuple'] == tuple(action[:i]))].copy()
    sim_filtered_before_pruning = sim_filtered_before_pruning.groupby('icustay_id').first().reset_index()
    sim_filtered = sim_filtered_before_pruning.copy()
    real_filtered = real_filtered_before_pruning.copy()
    sim_filtered[col] = sim_filtered[col].clip(min_y, max_y)
    real_filtered[col] = real_filtered[col].clip(min_y, max_y)
    real_filtered.loc[:, 'Y_lb'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + min_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    real_filtered.loc[:, 'Y_ub'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + max_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    yobs_values = list(real_filtered_before_pruning.loc[real_filtered_before_pruning['actions_tuple'] == tuple(action[:i]), col].values)
    ysim_values = list(sim_filtered_before_pruning[col].values)
    if len(real_filtered) > 10 and len(sim_filtered) > 10:
        exp_y_all_data = real_filtered.loc[real_filtered['actions_tuple'] == tuple(action[:i]), col].mean()
        exp_y_sim_all_data = sim_filtered[col].mean()
        ub_all_data = real_filtered['Y_ub'].mean()
        lb_all_data = real_filtered['Y_lb'].mean()
        for j in range(n_iter):
            obs_resampled = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = obs_resampled.loc[obs_resampled['actions_tuple'] == tuple(action[:i]), col].mean()
            ub = obs_resampled['Y_ub'].mean()
            lb = obs_resampled['Y_lb'].mean()
            sim_resampled = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_resampled[col].mean()
            row_append = pd.DataFrame.from_dict({'Exp_y': [exp_y], 'UB': [ub], 'LB': [lb], 'Sim_exp_y_all_data': [exp_y_sim_all_data], 'Sim_exp_y': [exp_y_sim], 'max_y':[max_y], 'min_y': [min_y], 'LB_all_data': [lb_all_data], 'UB_all_data': [ub_all_data],})
            df = pd.concat([df, row_append], ignore_index=True)
        return df, len(real_filtered), len(sim_filtered_before_pruning), len(sim_filtered), yobs_values, ysim_values
    return None, len(real_filtered), len(sim_filtered_before_pruning), len(sim_filtered), yobs_values, ysim_values


def compute_p_values(df, reverse_percentile):
    p_lb, p_ub = 1, 1
    for alpha in np.linspace(1e-6, 1.0, 500):
        if reverse_percentile:
            sim_lb = 2*df['Sim_exp_y_all_data'].mean() - df['Sim_exp_y'].quantile(1-alpha)
            sim_ub = 2*df['Sim_exp_y_all_data'].mean() - df['Sim_exp_y'].quantile(alpha)
            lb = 2*df['LB_all_data'].mean() - df['LB'].quantile(1 - alpha)
            ub = 2*df['UB_all_data'].mean() - df['UB'].quantile(alpha)
        else:
            sim_lb = df['Sim_exp_y'].quantile(alpha)
            sim_ub = df['Sim_exp_y'].quantile(1 - alpha)
            lb = df['LB'].quantile(alpha)
            ub = df['UB'].quantile(1 - alpha)
        if sim_ub < lb and p_lb == 1:
            p_lb = alpha
        if sim_lb > ub and p_ub == 1:
            p_ub = alpha
    return round(p_lb, 6), round(p_ub, 6)


def rejected_hypotheses_bootstrap_percentile(col, trajec_actions, sim_trajec_actions, obs_trajecs_actions_held_back, sim_data, MIMICtable, reverse_percentile):
    logging.info("Using bootstrapping on Qtwin as well")
    T = 5
    p_values = pd.DataFrame()
    pruned_hypotheses = pd.DataFrame()
    for t in range(1, 5):
        state_actions = obs_trajecs_actions_held_back[['gender', 'age', 'actions', 'x_t']].copy()
        state_actions.loc[:,'a'] = state_actions['actions'].apply(lambda x: tuple(x[:t]))
        state_actions.loc[:,'s'] = state_actions['x_t'].apply(lambda x: tuple(x[:t]))
        state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 5).drop_duplicates(['gender', 'age', 'a', 's'])
        total_hypotheses = len(state_actions)
        counter = 0
        logging.info(f"Using Reverse Percentile Bootstrap: {reverse_percentile}")
        for index, row in state_actions.iterrows():
            counter += 1
            logging.info(f"On hypothesis {counter}/{total_hypotheses} for t={t}")
            p = 1
            M_lower_quantile, M_upper_quantile = [], []
            df, n_obs, n_sim_before_pruning, n_sim, yobs_values, ysim_values = bootstrap_distribution_causal_bounds(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=t)
            if df is not None:
                p_lb, p_ub = compute_p_values(df, reverse_percentile)
                row_append = pd.DataFrame.from_dict({'gender': [row['gender']], 'age': [row['age']], 'actions': [row['a']], 'x_t': [row['s']], 'p_lb': [p_lb], 'p_ub': [p_ub], 'Y_lb_mean': [list(df['LB'])], 'Y_ub_mean': [list(df['UB'])], 'Sim_exp_y': [list(df['Sim_exp_y'])], 'Exp_y': [list(df['Exp_y'])], 'y_lo': [df['min_y'].mean()], 'y_up':[df['max_y'].mean()], 'yobs_values':[yobs_values], 'ysim_values':[ysim_values], 't': [t], 'n_obs': [n_obs], 'n_sim_before_pruning': [n_sim_before_pruning], 'n_sim': [n_sim]})
                p_values = pd.concat([p_values, row_append], ignore_index=True)
            else:
                row_append = pd.DataFrame.from_dict({'gender': [row['gender']], 'age': [row['age']], 'actions': [row['a']], 'x_t': [row['s']], 't': [t], 'n_obs': [n_obs], 'n_sim_before_pruning': [n_sim_before_pruning], 'n_sim': [n_sim]})
                pruned_hypotheses = pd.concat([pruned_hypotheses, row_append], ignore_index=True)
    if len(p_values) > 0:
        rej_hyps = p_values[(p_values['p_lb']<0.05/4) | (p_values['p_ub']<0.05/4)].copy()
    else:
        rej_hyps = pd.DataFrame()
    return len(rej_hyps), p_values, rej_hyps, len(p_values), pruned_hypotheses


def causal_bounds_hoeffdings_p_values(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=4):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+1], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+1], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(lambda x: tuple(x[:i]))
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(lambda x: tuple(x[:i]))
    df = pd.DataFrame()
    real_filtered = filter_dataset_to_get_d0(x_trajec, gender, age, obs_data, action, i)
    withheld_real_data = real_filtered.sample(frac=0.05, replace=False, random_state=1)
    real_filtered = real_filtered.drop(withheld_real_data.index)
    max_y, min_y = round(withheld_real_data[col].quantile(0.80), 2), round(withheld_real_data[col].quantile(0.20), 2)
    if max_y - min_y < 1e-6:
        max_y = min_y + 1
    sim_filtered_before_pruning = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec[:i])) & (sim['actions_tuple'] == tuple(action[:i]))].copy()
    sim_filtered_before_pruning = sim_filtered_before_pruning.groupby('icustay_id').first().reset_index()
    sim_filtered = sim_filtered_before_pruning
    sim_filtered[col] = sim_filtered[col].clip(min_y, max_y)
    real_filtered[col] = real_filtered[col].clip(min_y, max_y)
    real_filtered.loc[:, 'Y_lb'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + min_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    real_filtered.loc[:, 'Y_ub'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + max_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    if len(real_filtered) > 10 and len(sim_filtered) > 10:
        exp_y_sim = sim_filtered[col].clip(min_y, max_y).mean()
        ub_all_data = real_filtered['Y_ub'].mean()
        lb_all_data = real_filtered['Y_lb'].mean()
        p_lb = 1*(real_filtered['Y_lb'].mean() <= exp_y_sim) + 4*np.exp(-2 * (1/(1/np.sqrt(len(real_filtered)) + 1/np.sqrt(len(sim_filtered))) * (lb_all_data - exp_y_sim)/(max_y - min_y))**2)*(real_filtered['Y_lb'].mean() > exp_y_sim)
        p_ub = 1*(real_filtered['Y_ub'].mean() >= exp_y_sim) + 4*np.exp(-2 * (1/(1/np.sqrt(len(real_filtered)) + 1/np.sqrt(len(sim_filtered))) * (exp_y_sim - ub_all_data)/(max_y - min_y))**2)*(real_filtered['Y_ub'].mean() < exp_y_sim)
        return p_lb, p_ub, real_filtered['Y_lb'].mean(), real_filtered['Y_ub'].mean(), real_filtered[col].mean(), exp_y_sim, len(real_filtered), (real_filtered['actions_tuple'] == tuple(action[:i])).sum(), min_y, max_y, len(sim_filtered_before_pruning), sim_filtered_before_pruning[col].between(min_y, max_y).sum()
    return None, None, None, None, None, None, len(real_filtered), (real_filtered['actions_tuple'] == tuple(action[:i])).sum(), min_y, max_y, len(sim_filtered_before_pruning), sim_filtered_before_pruning[col].between(min_y, max_y).sum()


def rejected_hypotheses_hoeffdings(col, trajec_actions, sim_trajec_actions, obs_trajecs_actions_held_back, sim_data, MIMICtable):
    T = 5
    p_values = pd.DataFrame()
    pruned_hypotheses = pd.DataFrame()
    for t in range(1, T):
        state_actions = obs_trajecs_actions_held_back[['gender', 'age', 'actions', 'x_t']].copy()
        state_actions.loc[:,'a'] = state_actions['actions'].apply(lambda x: tuple(x[:t]))
        state_actions.loc[:,'s'] = state_actions['x_t'].apply(lambda x: tuple(x[:t]))
        state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 5).drop_duplicates(['gender', 'age', 'a', 's'])
        total_hypotheses = len(state_actions)
        counter = 0
        logging.info(f"Using Hoeffdings")
        for index, row in state_actions.iterrows():
            counter += 1
            logging.info(f"On hypothesis {counter}/{total_hypotheses} for t={t}")
            M_lower_quantile, M_upper_quantile = [], []
            p_lb, p_ub, Y_lb_mean, Y_ub_mean, obs_Y_mean, exp_y_sim, n_d0, n_d0_with_right_actions, y_lo, y_up, n_sim, n_sim_w_pruning = causal_bounds_hoeffdings_p_values(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=t)
            if p_lb is not None:
                row_append = pd.DataFrame.from_dict({'gender': [row['gender']], 'age': [row['age']], 'actions': [row['a']], 'x_t': [row['s']], 'p_lb': [p_lb], 'p_ub': [p_ub], 'Y_lb_mean': [Y_lb_mean], 'Y_ub_mean': [Y_ub_mean], 'Exp_y': [obs_Y_mean], 'Sim_exp_y': [exp_y_sim], 't': [t], 'n_obs': [n_d0], 'n_obs_with_actions_of_interest': [n_d0_with_right_actions], 'y_lo': [y_lo], 'y_up': [y_up], 'n_sim': [n_sim], 'n_sim_w_pruning': [n_sim_w_pruning]})
                p_values = pd.concat([p_values, row_append], ignore_index=True)
            else:
                row_append = pd.DataFrame.from_dict({'gender': [row['gender']], 'age': [row['age']], 'actions': [row['a']], 'x_t': [row['s']], 't': [t], 'n_obs': [n_d0], 'n_obs_with_actions_of_interest': [n_d0_with_right_actions], 'y_lo': [y_lo], 'y_up': [y_up], 'n_sim': [n_sim], 'n_sim_w_pruning': [n_sim_w_pruning]})
                pruned_hypotheses = pd.concat([pruned_hypotheses, row_append], ignore_index=True)
    if len(p_values) > 0:
        rej_hyps = p_values[(p_values['p_lb']<0.05) | (p_values['p_ub']<0.05)].copy()
    else:
        rej_hyps = pd.DataFrame()
    return len(rej_hyps), p_values, rej_hyps, len(p_values), pruned_hypotheses


x_columns = ['Weight_kg', 'paCO2', 'paO2', 'HCO3', 'Arterial_pH', 'Calcium', 'Chloride', 'DiaBP', 'Glucose', 'MeanBP', 'Potassium', 'RR', 'Temp_C', 'Sodium', 'SysBP', 'HR']

def get_col_bins(MIMIC_data_used, MIMIC_data_held_back, sim_data, use_kmeans, col_bins_num, column):
    if use_kmeans:
        kmeans = KMeans(n_clusters=col_bins_num, random_state=0).fit((MIMIC_data_held_back[x_columns]-MIMIC_data_held_back[x_columns].mean())/MIMIC_data_held_back[x_columns].std())
        col_bins_obs_held_back = kmeans.labels_
        col_bins_obs = kmeans.predict((MIMIC_data_used[x_columns] - MIMIC_data_held_back[x_columns].mean())/MIMIC_data_held_back[x_columns].std())
        col_bins_sim = kmeans.predict((sim_data[x_columns] - MIMIC_data_held_back[x_columns].mean())/MIMIC_data_held_back[x_columns].std())
    else:
        bins, labels = [0], [0]
        for i in range(1, col_bins_num):
            bins.append(MIMIC_data_held_back[column].quantile(i/col_bins_num))
            labels.append(i)
        bins.append(max(MIMIC_data_used[column].max(), sim_data[column].max()))
        col_bins_obs_held_back = pd.cut(MIMIC_data_held_back[column], bins=bins, labels=labels).astype(float)
        col_bins_obs = pd.cut(MIMIC_data_used[column], bins=bins, labels=labels).astype(float)
        col_bins_sim = pd.cut(sim_data[column], bins=bins, labels=labels).astype(float)
    return col_bins_obs_held_back, col_bins_obs, col_bins_sim


def do_hypothesis_testing(column, MIMICtable, MIMIC_data_held_back, MIMIC_data_used, sim_data, col_bins_num, hyp_test_dir, use_kmeans, reverse_percentile, hoeffdings_test):
    logging.info("Using held-out data to determine which hypotheses to run")
    icustay_ids_held_back = MIMIC_data_held_back['icustay_id'].unique()

    logging.info("doing hypothesis testing")
    col_bins_obs_held_back, col_bins_obs, col_bins_sim = get_col_bins(MIMIC_data_used, MIMIC_data_held_back, sim_data, use_kmeans, col_bins_num, column)
    logging.info("Extracted column bins for Sim and Obs data")

    trajectories = pd.DataFrame()
    trajectories.loc[:,'t'] = np.arange(len(MIMIC_data_used))%5
    trajectories.loc[:,'icustay_id'] = MIMIC_data_used['icustay_id']
    trajectories.loc[:,'gender'] = MIMIC_data_used['gender']
    trajectories.loc[:,'age'] = MIMIC_data_used['age']
    trajectories.loc[:,'x_t'] = col_bins_obs
    trajectories.loc[:,'A_t'] = MIMIC_data_used['A']
    trajectories = trajectories[trajectories['t']!=4]
    
    obs_trajecs_held_back = pd.DataFrame()
    obs_trajecs_held_back.loc[:,'t'] = np.arange(len(MIMIC_data_held_back))%5
    obs_trajecs_held_back.loc[:,'icustay_id'] = MIMIC_data_held_back['icustay_id']
    obs_trajecs_held_back.loc[:,'gender'] = MIMIC_data_held_back['gender']
    obs_trajecs_held_back.loc[:,'age'] = MIMIC_data_held_back['age']
    obs_trajecs_held_back.loc[:,'x_t'] = col_bins_obs_held_back
    obs_trajecs_held_back.loc[:,'A_t'] = MIMIC_data_held_back['A']
    obs_trajecs_held_back = obs_trajecs_held_back[obs_trajecs_held_back['t']!=4]

    sim_trajecs = pd.DataFrame()
    sim_trajecs.loc[:,'t'] = np.arange(len(sim_data))%5
    sim_trajecs.loc[:,'icustay_id'] = sim_data['icustay_id']
    sim_trajecs.loc[:,'gender'] = sim_data['gender']
    sim_trajecs.loc[:,'age'] = sim_data['age']
    sim_trajecs.loc[:,'x_t'] = col_bins_sim
    sim_trajecs.loc[:,'A_t'] = sim_data['A']
    sim_trajecs = sim_trajecs[sim_trajecs['t']!=4]
    
    trajec_actions = get_trajec_actions_df(trajectories)
    sim_trajec_actions = get_trajec_actions_df(sim_trajecs)
    obs_trajecs_actions_held_back = get_trajec_actions_df(obs_trajecs_held_back)
    logging.info(f'trajec_actions length: {len(trajec_actions)}')
    logging.info(f'trajec_actions held back length: {len(obs_trajecs_actions_held_back)}')
    
    logging.info("Simulator and Observational data ready")
    if use_kmeans:
        trajec_actions.to_csv(f'{hyp_test_dir}/trajec_actions.csv', index=False)
        sim_trajec_actions.to_csv(f'{hyp_test_dir}/sim_trajec_actions.csv', index=False)
        obs_trajecs_actions_held_back.to_csv(f'{hyp_test_dir}/obs_trajecs_actions_held_back.csv', index=False)
    else:
        trajec_actions.to_csv(f'{hyp_test_dir}/trajec_actions_{column}.csv', index=False)
        sim_trajec_actions.to_csv(f'{hyp_test_dir}/sim_trajec_actions_{column}.csv', index=False)
        obs_trajecs_actions_held_back.to_csv(f'{hyp_test_dir}/obs_trajecs_actions_held_back_{column}.csv', index=False)

    if hoeffdings_test:
        logging.info("Using hoeffdings for hypothesis testing")
        num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses = rejected_hypotheses_hoeffdings(column, trajec_actions, sim_trajec_actions, obs_trajecs_actions_held_back, sim_data, MIMICtable)
    else:
        num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses = rejected_hypotheses_bootstrap_percentile(column, trajec_actions, sim_trajec_actions, obs_trajecs_actions_held_back, sim_data, MIMICtable, reverse_percentile)
    return num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses, trajec_actions, sim_trajec_actions
