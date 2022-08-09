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
    for i in range(min(t+1, len(actions_of_interest))):
        if actions_of_interest[i] != action_taken[i]:
            break
    return x_trajec_of_interest[:i+1] == x_trajec[:i+1]

def filter_dataset_to_get_d0(x_trajec_of_interest, gender, age, obs_data, actions_of_interest, t):
    obs_data_filtered = obs_data.loc[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age)]
    datapoint_in_d0 = obs_data_filtered.apply(lambda x: xn_in_bn(actions_of_interest, x['actions'], x_trajec_of_interest, x['x_t'], t), axis=1)
    obs_data_filtered = obs_data_filtered.loc[datapoint_in_d0]
    return obs_data_filtered

def compute_probs_deprecated(trajec_actions, column):
    for index, row in trajec_actions.iterrows():
        prob_a_den = 0
        gamma = []
        for i in range(len(row['actions'])):
            prob_a_num = (find_elements(trajec_actions['gender'],row['gender']) & find_elements(trajec_actions['age'],row['age']) & find_elements_starting_with(trajec_actions[column],row[column][:i+1]) & find_elements_starting_with(trajec_actions['actions'],row['actions'][:i+1])).sum()
            if i == 0:
                prob_a_den += (find_elements(trajec_actions['gender'],row['gender']) & find_elements(trajec_actions['age'],row['age']) & find_elements_starting_with(trajec_actions[column],row[column][:i+1])).sum()
            else:
                prob_a_den += ((find_elements_starting_with(trajec_actions[column],row[column][:i+1])) & (find_elements(trajec_actions['gender'],row['gender']))&(find_elements(trajec_actions['age'],row['age']))&(find_elements_starting_with(trajec_actions['actions'],row['actions'][:i]))).sum() -\
                ((find_elements_starting_with(trajec_actions[column],row[column][:i])) & (find_elements(trajec_actions['gender'],row['gender']))&(find_elements(trajec_actions['age'],row['age']))&(find_elements_starting_with(trajec_actions['actions'],row['actions'][:i]))).sum()
            gamma.append(prob_a_num/prob_a_den)
            trajec_actions.at[index, f'prob_a_{i}'] = prob_a_num/prob_a_den
        trajec_actions.at[index, 'prob_a'] = prob_a_num/prob_a_den
    return trajec_actions


def compute_probs(trajec_actions, x_trajec, gender, age, actions):
    prob_a_den = 0
    gamma = []
    df = pd.DataFrame()
    df[[f'x_0',f'x_1',f'x_2',f'x_3']] = pd.DataFrame(trajec_actions['x_t'].tolist(), index= trajec_actions.index)
    df[[f'a_0',f'a_1',f'a_2',f'a_3']] = pd.DataFrame(trajec_actions['actions'].tolist(), index= trajec_actions.index)
    trajecs_filtered = (trajec_actions['gender'] == gender) & (trajec_actions['age'] == age)
    num_term = trajecs_filtered
    den_term_1 = trajecs_filtered
    den_term_2 = trajecs_filtered
    for i in range(len(actions)):
        num_term = num_term & (df[f'x_{i}'] == x_trajec[i]) & (df[f'a_{i}'] == actions[i])
        prob_a_num = num_term.sum()
        if i == 0:
            den_term_1 = den_term_1 & (df[f'x_{i}'] == x_trajec[i])
            prob_a_den += den_term_1.sum()
        else:
            den_term_1 = den_term_1 & (df[f'x_{i}'] == x_trajec[i]) & (df[f'a_{i-1}'] == actions[i-1])
            den_term_2 = den_term_2 & (df[f'x_{i-1}'] == x_trajec[i-1]) & (df[f'a_{i-1}'] == actions[i-1])
            prob_a_den += den_term_1.sum() - den_term_2.sum()
        if prob_a_num == 0:
            gamma.append(0)
        else:
            gamma.append(prob_a_num/prob_a_den)
    return gamma


def bootstrap_distribution(col, gender, age, action, column_v, trajec_actions, sim_trajec_actions, n_iter=100):
    df = pd.DataFrame()
    max_y = trajec_actions.loc[find_elements(trajec_actions['gender'], gender) & find_elements(trajec_actions['age'], age) & find_elements_containing(trajec_actions[col], max(column_v)), f'{col}_raw'].max()
    min_y = trajec_actions.loc[find_elements(trajec_actions['gender'], gender) & find_elements(trajec_actions['age'], age) & find_elements_containing(trajec_actions[col], min(column_v)), f'{col}_raw'].min()
    sim_filtered = sim_trajec_actions[find_elements_starting_with(sim_trajec_actions[col], column_v) & find_elements(sim_trajec_actions['gender'], gender) & find_elements(sim_trajec_actions['age'], age) & find_elements_starting_with(sim_trajec_actions['actions'], action)].copy()
    real_filtered = trajec_actions[find_elements(trajec_actions[col], column_v) & find_elements(trajec_actions['gender'], gender) & find_elements(sim_trajec_actions['age'], age) & find_elements_starting_with(trajec_actions['actions'], action)].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for i in range(n_iter):
            real_train = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = real_train[f'{col}_raw'].mean()
            prob = real_train['prob_a'].max()
            sim_train = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_train[f'{col}_raw'].mean()
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y}, ignore_index=True)
        return df
    return None


def bootstrap_distribution_deprecated(col, gender, age, action, column_v, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim_data = sim_data.rename(columns={col: f'{col}_raw'})
    MIMICtable = MIMICtable.rename(columns={col: f'{col}_raw'})
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', col]].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', col, f'prob_a_{i}']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    df = pd.DataFrame()
    max_y = obs_data.loc[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age) & find_elements_containing(obs_data[col], max(column_v)), f'{col}_raw'].max()
    min_y = obs_data.loc[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age) & find_elements_containing(obs_data[col], min(column_v)), f'{col}_raw'].min()
    sim_filtered = sim[find_elements(sim['gender'], gender) & find_elements(sim['age'], age) & find_elements_starting_with(sim[col], column_v[:i+1]) & find_elements_starting_with(sim['actions'], action[:i+1])].copy()
    real_filtered = obs_data[find_elements(obs_data['gender'], gender) & find_elements(obs_data['age'], age) & find_elements_starting_with(obs_data[col], column_v[:i+1]) & find_elements_starting_with(obs_data['actions'], action[:i+1])].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for j in range(n_iter):
            real_train = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = real_train[f'{col}_raw'].mean()
            prob = real_train[f'prob_a_{i}'].max()
            sim_train = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_train[f'{col}_raw'].mean()
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y}, ignore_index=True)
        return df
    return None


class Statistic():
    def __init__(self, obs_data_filtered, sim_data_filtered, col, gender, age, action, x_trajec, i):
        self.obs_data_filtered = obs_data_filtered
        self.col = col
        self.sim_data_filtered = sim_data_filtered
        self.gender = gender
        self.age = age
        self.action = action
        self.x_trajec = x_trajec
        self.i = i

    def __call__(self, values):
        obs_resampled = self.obs_data_filtered.loc[values]
        probs = compute_probs(obs_resampled, self.x_trajec, self.gender, self.age, self.action)
        prob = probs[self.i]
        exp_y = obs_resampled[self.col].mean()
        exp_y_sim = self.sim_data_filtered[self.col].mean()
        R_statistic = (exp_y_sim - prob*exp_y)#/(1-prob)
        return R_statistic

def bootstrap_scipy_wrapper(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(tuple)
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(tuple)
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(tuple)
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(tuple)
    df = pd.DataFrame()
    max_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[-1]) == x_trajec[-1]), col].max()
    min_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[-1]) == x_trajec[-1]), col].min()
    sim_filtered = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec)) & (sim['actions_tuple'] == tuple(action))].copy()
    real_filtered = obs_data[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data[f'x_tuple']==tuple(x_trajec)) & (obs_data['actions_tuple'] == tuple(action))].copy()
    r_statistic = Statistic(real_filtered, sim_filtered, col, gender, age, action, x_trajec, i)
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for alpha in np.linspace(0.05, 0.95, 19):
            ci = scipy.stats.bootstrap((real_filtered.index,), r_statistic, vectorized=False, n_resamples=n_iter, confidence_level=alpha)
    return None

def bootstrap_distribution_percentile(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(tuple)
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(tuple)
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(tuple)
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(tuple)
    df = pd.DataFrame()
    max_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[-1]) == x_trajec[-1]), col].max()
    min_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[-1]) == x_trajec[-1]), col].min()
    sim_filtered = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec)) & (sim['actions_tuple'] == tuple(action))].copy()
    real_filtered = obs_data[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data[f'x_tuple']==tuple(x_trajec))].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        probs_all_data = compute_probs(real_filtered, x_trajec, gender, age, action)
        prob_all_data = probs_all_data[i]
        exp_y_all_data = real_filtered.loc[real_filtered['actions_tuple'] == tuple(action), col].mean()
        exp_y_sim = sim_filtered[col].mean()
        for j in range(n_iter):
            obs_resampled = resample(real_filtered, n_samples=len(real_filtered))
            probs = compute_probs(obs_resampled, x_trajec, gender, age, action)
            prob = probs[i]
            exp_y = obs_resampled.loc[obs_resampled['actions_tuple'] == tuple(action), col].mean()
            row_append = pd.DataFrame.from_dict({'Exp_y': [exp_y], 'UB': [prob*exp_y + (1-prob)*max_y], 'LB': [prob*exp_y + (1-prob)*min_y], 'Sim_exp_y': [exp_y_sim], 'max_y':[max_y], 'min_y': [min_y], 'LB_all_data': [prob_all_data*exp_y_all_data + (1-prob_all_data)*min_y], 'UB_all_data': [prob_all_data*exp_y_all_data + (1-prob_all_data)*max_y], 'M': [(exp_y_sim - prob*exp_y)/(1-prob)]})
            df = pd.concat([df, row_append], ignore_index=True)
        return df
    return None


def bootstrap_distribution_causal_bounds(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(lambda x: tuple(x[:i]))
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(lambda x: tuple(x[:i]))
    df = pd.DataFrame()
    max_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[i]) == x_trajec[i]), col].max()
    min_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[i]) == x_trajec[i]), col].min()
    sim_filtered = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec[:i])) & (sim['actions_tuple'] == tuple(action[:i]))].copy()
    real_filtered = filter_dataset_to_get_d0(x_trajec, gender, age, obs_data, action, i)
    real_filtered.loc[:, 'Y_lb'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + min_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    real_filtered.loc[:, 'Y_ub'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + max_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        exp_y_all_data = real_filtered.loc[real_filtered['actions_tuple'] == tuple(action[:i]), col].mean()
        exp_y_sim = sim_filtered[col].mean()
        ub_all_data = real_filtered['Y_ub'].mean()
        lb_all_data = real_filtered['Y_lb'].mean()
        for j in range(n_iter):
            obs_resampled = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = obs_resampled.loc[obs_resampled['actions_tuple'] == tuple(action[:i]), col].mean()
            ub = obs_resampled['Y_ub'].mean()
            lb = obs_resampled['Y_lb'].mean()
            row_append = pd.DataFrame.from_dict({'Exp_y': [exp_y], 'UB': [ub], 'LB': [lb], 'Sim_exp_y': [exp_y_sim], 'max_y':[max_y], 'min_y': [min_y], 'LB_all_data': [lb_all_data], 'UB_all_data': [ub_all_data],})
            df = pd.concat([df, row_append], ignore_index=True)
        return df
    return None


def bootstrap_distribution_causal_bounds_with_qtwin(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, discretised_outcome, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(lambda x: tuple(x[:i]))
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(lambda x: tuple(x[:i]))
    df = pd.DataFrame()
    real_filtered = filter_dataset_to_get_d0(x_trajec, gender, age, obs_data, action, i)
    withheld_real_data = real_filtered.sample(frac=0.05, replace=False, random_state=1)
    real_filtered = real_filtered.drop(withheld_real_data.index)
    if discretised_outcome:
        max_y, min_y = 1, 0
    else:
        max_y, min_y = round(withheld_real_data[col].quantile(0.80), 2), round(withheld_real_data[col].quantile(0.20), 2)
    if max_y - min_y < 1e-6:
        max_y = min_y + 1
    sim_filtered_before_pruning = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec[:i])) & (sim['actions_tuple'] == tuple(action[:i]))].copy()
    sim_filtered = sim_filtered_before_pruning
    sim_filtered[col] = sim_filtered[col].clip(min_y, max_y)
    real_filtered[col] = real_filtered[col].clip(min_y, max_y)
    real_filtered.loc[:, 'Y_lb'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + min_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    real_filtered.loc[:, 'Y_ub'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + max_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    yobs_values = list(real_filtered.loc[real_filtered['actions_tuple'] == tuple(action[:i]), col].values)
    ysim_values = list(sim_filtered[col].values)
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



def bootstrap_distribution_(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(tuple)
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(tuple)
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(tuple)
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(tuple)
    df = pd.DataFrame()
    max_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[-1]) == x_trajec[-1]), col].max()
    min_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[-1]) == x_trajec[-1]), col].min()
    sim_filtered = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec)) & (sim['actions_tuple'] == tuple(action))].copy()
    real_filtered = obs_data[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data[f'x_tuple']==tuple(x_trajec)) & (obs_data['actions_tuple'] == tuple(action))].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for j in range(n_iter):
            obs_resampled = resample(obs_data, n_samples=len(obs_data))
            probs = compute_probs(obs_resampled, x_trajec, gender, age, action)
            prob = probs[i]
            real_train = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = real_train[col].mean()
            sim_train = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_train[col].mean()
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y, 'M': (exp_y_sim - prob*exp_y)/(1-prob)}, ignore_index=True)
        return df
    return None


def rejected_hypotheses_bootstrap(col, trajec_actions, sim_trajec_actions):
    logging.info("calculating rejected hypotheses")
    state_actions = trajec_actions[['gender', 'age', 'actions', col]].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions[col].apply(tuple)
    state_actions = state_actions.drop_duplicates(['gender', 'age', 'a', 's'])
    logging.info(f"Total hypotheses to be checked: {len(state_actions)}")
    p_values = pd.DataFrame()
    counter = 0
    for index, row in state_actions.iterrows():
        counter += 1
        logging.info(f"On hypothesis {counter}/{len(state_actions)}")
        df = bootstrap_distribution(col, row['gender'], row['age'], row['actions'], row[col], trajec_actions, sim_trajec_actions)
        if df is not None:
            sigma_ub = (df['UB']-df['Sim_exp_y']).var()
            exp_ub = (df['UB']-df['Sim_exp_y']).mean()
            p_ub = st.norm.cdf(exp_ub/np.sqrt(sigma_ub))
            sigma_lb = (df['Sim_exp_y']-df['LB']).var()
            exp_lb = (df['Sim_exp_y']-df['LB']).mean()
            p_lb = st.norm.cdf(exp_lb/np.sqrt(sigma_lb))
            p_values = p_values.append({'gender': row['gender'], 'age': row['age'], 'actions': row['actions'], col: row[col], 'p_lb': p_lb, 'p_ub': p_ub}, ignore_index=True)
    total_hypotheses = len(p_values)
    rej_hyps = p_values[(p_values['p_lb']<0.05/total_hypotheses) ^ (p_values['p_ub']<0.05/total_hypotheses)].copy()
    for index, row in rej_hyps.iterrows():
        rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions[col], row[col])).sum()
        rej_hyps.loc[index, 'n_sim'] = (find_elements(sim_trajec_actions['gender'], row['gender']) & find_elements(sim_trajec_actions['age'], row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions[col], row[col])).sum()
    return len(rej_hyps), p_values, rej_hyps, total_hypotheses


def compute_p_values(df, reverse_percentile):
    p_lb, p_ub = 1, 1
    for alpha in np.linspace(1e-6, 1.0, 100):
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
    return round(p_lb, 3), round(p_ub, 3)


def rejected_hypotheses_bootstrap_percentile(col, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, reverse_percentile):
    T = 4
    state_actions = trajec_actions[['gender', 'age', 'actions', 'x_t']].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions['x_t'].apply(tuple)
    state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 10).drop_duplicates(['gender', 'age', 'a', 's'])
    total_hypotheses = len(state_actions)
    p_values = pd.DataFrame()
    counter = 0
    logging.info(f"Using Reverse Percentile Bootstrap: {reverse_percentile}")
    for index, row in state_actions.iterrows():
        counter += 1
        logging.info(f"On hypothesis {counter}/{total_hypotheses}")
        p = 1
        M_lower_quantile, M_upper_quantile = [], []
        for t in range(T):
            df = bootstrap_distribution_causal_bounds(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=t)
            if df is not None:
                if reverse_percentile:
                    p_lb = (2*df['LB_all_data']-df['LB'] <= df['Sim_exp_y']).sum()/len(df)
                    p_ub = (2*df['UB_all_data']-df['UB'] >= df['Sim_exp_y']).sum()/len(df)
                else:
                    p_lb = (df['LB'] <= df['Sim_exp_y']).sum()/len(df)
                    p_ub = (df['UB'] >= df['Sim_exp_y']).sum()/len(df)
                row_append = pd.DataFrame.from_dict({'gender': [row['gender']], 'age': [row['age']], 'actions': [row['actions']], 'x_t': [row['x_t']], 'p_lb': [p_lb], 'p_ub': [p_ub], 'Y_lb_mean': [list(df['LB'])], 'Y_ub_mean': [list(df['UB'])], 'Sim_exp_y': [list(df['Sim_exp_y'])], 'Exp_y': [list(df['Exp_y'])], 't': [t]})
                p_values = pd.concat([p_values, row_append], ignore_index=True)
    if len(p_values) > 0:
        rej_hyps = p_values[(p_values['p_lb']<0.05) | (p_values['p_ub']<0.05)].copy()
        for index, row in rej_hyps.iterrows():
            rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions['x_t'], row['x_t'])).sum()
            rej_hyps.loc[index, 'n_sim'] = (find_elements(sim_trajec_actions['gender'], row['gender']) & find_elements(sim_trajec_actions['age'], row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions['x_t'], row['x_t'])).sum()
        for index, row in p_values.iterrows():
            p_values.loc[index, 'n_real'] = ((trajec_actions['gender'] == row['gender']) & (trajec_actions['age'] == row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions['x_t'], row['x_t'])).sum()
            p_values.loc[index, 'n_sim'] = ((sim_trajec_actions['gender'] == row['gender']) & (sim_trajec_actions['age'] == row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions['x_t'], row['x_t'])).sum()
    else:
        rej_hyps = pd.DataFrame()
    return len(rej_hyps), p_values, rej_hyps, len(p_values)


def rejected_hypotheses_bootstrap_percentile_with_qtwin(col, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, reverse_percentile, discretised_outcome):
    logging.info("Using bootstrapping on Qtwin as well")
    T = 4
    p_values = pd.DataFrame()
    pruned_hypotheses = pd.DataFrame()
    for t in range(T):
        state_actions = sim_trajec_actions[['gender', 'age', 'actions', 'x_t']].copy()
        state_actions.loc[:,'a'] = state_actions['actions'].apply(lambda x: tuple(x[:t]))
        state_actions.loc[:,'s'] = state_actions['x_t'].apply(lambda x: tuple(x[:t]))
        state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 10).drop_duplicates(['gender', 'age', 'a', 's'])
        total_hypotheses = len(state_actions)
        counter = 0
        logging.info(f"Using Reverse Percentile Bootstrap: {reverse_percentile}")
        for index, row in state_actions.iterrows():
            counter += 1
            logging.info(f"On hypothesis {counter}/{total_hypotheses} for t={t}")
            p = 1
            M_lower_quantile, M_upper_quantile = [], []
            df, n_obs, n_sim_before_pruning, n_sim, yobs_values, ysim_values = bootstrap_distribution_causal_bounds_with_qtwin(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, discretised_outcome, n_iter=100, i=t)
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


def causal_bounds_hoeffdings_p_values(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, pruning, discretised_outcome, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(lambda x: tuple(x[:i]))
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(lambda x: tuple(x[:i]))
    df = pd.DataFrame()
    real_filtered = filter_dataset_to_get_d0(x_trajec, gender, age, obs_data, action, i)
    withheld_real_data = real_filtered.sample(frac=0.05, replace=False, random_state=1)
    real_filtered = real_filtered.drop(withheld_real_data.index)
    if discretised_outcome:
        max_y, min_y = 1, 0
    else:
        max_y, min_y = round(withheld_real_data[col].quantile(0.80), 2), round(withheld_real_data[col].quantile(0.20), 2)
    if max_y - min_y < 1e-6:
        max_y = min_y + 1
    sim_filtered_before_pruning = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec[:i])) & (sim['actions_tuple'] == tuple(action[:i]))].copy()
    if pruning:
        sim_filtered = sim_filtered_before_pruning[sim_filtered_before_pruning[col].between(min_y, max_y)]
    else:
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


def causal_bounds_hoeffdings_p_values_without_qtwin_est(col, gender, age, action, x_trajec, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', 'x_t']].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    sim.loc[:,f'x_tuple'] = sim['x_t'].apply(lambda x: tuple(x[:i]))
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'x_tuple'] = obs_data['x_t'].apply(lambda x: tuple(x[:i]))
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(lambda x: tuple(x[:i]))
    df = pd.DataFrame()
    max_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[i]) == x_trajec[i]), col].max()
    min_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data['x_t'].apply(lambda x: x[i]) == x_trajec[i]), col].min()
    sim_filtered = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'x_tuple']==tuple(x_trajec[:i])) & (sim['actions_tuple'] == tuple(action[:i]))].copy()
    real_filtered = filter_dataset_to_get_d0(x_trajec, gender, age, obs_data, action, i)
    real_filtered.loc[:, 'Y_lb'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + min_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    real_filtered.loc[:, 'Y_ub'] = real_filtered[col]*(real_filtered['actions_tuple'] == tuple(action[:i])) + max_y*(real_filtered['actions_tuple'] != tuple(action[:i]))
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        exp_y_sim = sim_filtered[col].mean()
        ub_all_data = real_filtered['Y_ub'].mean()
        lb_all_data = real_filtered['Y_lb'].mean()
        p_lb = 1*(real_filtered['Y_lb'].mean() <= exp_y_sim) + 2*np.exp(-2 * len(real_filtered) * ((lb_all_data - exp_y_sim)/(max_y - min_y))**2)*(real_filtered['Y_lb'].mean() > exp_y_sim)
        p_ub = 1*(real_filtered['Y_ub'].mean() >= exp_y_sim) + 2*np.exp(-2 * len(real_filtered) * ((ub_all_data - exp_y_sim)/(max_y - min_y))**2)*(real_filtered['Y_ub'].mean() < exp_y_sim)
        return p_lb, p_ub, real_filtered['Y_lb'].mean(), real_filtered['Y_ub'].mean(), real_filtered[col].mean(), exp_y_sim
    return None, None, None, None, None, None


def rejected_hypotheses_hoeffdings(col, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, pruning, discretised_outcome):
    T = 4
    p_values = pd.DataFrame()
    pruned_hypotheses = pd.DataFrame()
    for t in range(T):
        state_actions = sim_trajec_actions[['gender', 'age', 'actions', 'x_t']].copy()
        state_actions.loc[:,'a'] = state_actions['actions'].apply(lambda x: tuple(x[:t]))
        state_actions.loc[:,'s'] = state_actions['x_t'].apply(lambda x: tuple(x[:t]))
        state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 10).drop_duplicates(['gender', 'age', 'a', 's'])
        total_hypotheses = len(state_actions)
        counter = 0
        logging.info(f"Using Hoeffdings")
        for index, row in state_actions.iterrows():
            counter += 1
            logging.info(f"On hypothesis {counter}/{total_hypotheses} for t={t}")
            M_lower_quantile, M_upper_quantile = [], []
            p_lb, p_ub, Y_lb_mean, Y_ub_mean, obs_Y_mean, exp_y_sim, n_d0, n_d0_with_right_actions, y_lo, y_up, n_sim, n_sim_w_pruning = causal_bounds_hoeffdings_p_values(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, pruning, discretised_outcome, n_iter=100, i=t)
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


def rejected_hypotheses_bootstrap_trajectories(col, trajec_actions, sim_trajec_actions, sim_data, MIMICtable):
    T = 4
    state_actions = trajec_actions[['gender', 'age', 'actions', 'x_t']].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions['x_t'].apply(tuple)
    state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 10).drop_duplicates(['gender', 'age', 'a', 's'])
    total_hypotheses = len(state_actions)
    p_values = pd.DataFrame()
    counter = 0
    for index, row in state_actions.iterrows():
        counter += 1
        logging.info(f"On hypothesis {counter}/{total_hypotheses}")
        p = 1
        M_lower_quantile, M_upper_quantile = [], []
        for t in range(T):
            df = bootstrap_distribution_(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=t)
            if df is not None:
                p = min(p, ((df['LB'] <= df['Sim_exp_y']) & (df['UB'] >= df['Sim_exp_y'])).sum()/len(df))
                M_lower_quantile.append(df['M'].quantile(0.05))
                M_upper_quantile.append(df['M'].quantile(0.95))
        if df is not None:
            p_values = p_values.append({'gender': row['gender'], 'age': row['age'], 'actions': row['actions'], 'x_t': row['x_t'], 'p': p, 'Sim_exp_y': df['Sim_exp_y'].mean(), 'Exp_y': df['Exp_y'].mean(), 'M_0.05_quantiles': M_lower_quantile , 'M_0.95_quantiles': M_upper_quantile}, ignore_index=True)
    if len(p_values) > 0:
        rej_hyps = p_values[(p_values['p']<0.05/total_hypotheses/T)].copy()
        for index, row in rej_hyps.iterrows():
            rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions['x_t'], row['x_t'])).sum()
            rej_hyps.loc[index, 'n_sim'] = (find_elements(sim_trajec_actions['gender'], row['gender']) & find_elements(sim_trajec_actions['age'], row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions['x_t'], row['x_t'])).sum()
        for index, row in p_values.iterrows():
            p_values.loc[index, 'n_real'] = ((trajec_actions['gender'] == row['gender']) & (trajec_actions['age'] == row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions['x_t'], row['x_t'])).sum()
            p_values.loc[index, 'n_sim'] = ((sim_trajec_actions['gender'] == row['gender']) & (sim_trajec_actions['age'] == row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions['x_t'], row['x_t'])).sum()
    else:
        rej_hyps = pd.DataFrame()
    return len(rej_hyps), p_values, rej_hyps, total_hypotheses


def get_col_bin(col_v, col_name, col_bins_num, MIMICtable, col_bins_obs):
    for col_bin in range(1, col_bins_num+1):
        if (col_v <= MIMICtable.loc[col_bins_obs==col_bin, f'{col_name}'].min()):
            return col_bin
        if (col_v <= MIMICtable.loc[col_bins_obs==col_bin, f'{col_name}'].max()) and (col_v >= MIMICtable.loc[col_bins_obs==col_bin, f'{col_name}'].min()):
            return col_bin
    return col_bin

x_columns = ['Weight_kg', 'paCO2', 'paO2', 'HCO3', 'Arterial_pH', 'Calcium', 'Chloride', 'DiaBP', 'Glucose', 'MeanBP', 'Potassium', 'RR', 'Temp_C', 'Sodium', 'SysBP', 'HR']

def get_col_bins(MIMICtable, sim_data, use_kmeans, col_bins_num, column):
    if use_kmeans:
        kmeans = KMeans(n_clusters=col_bins_num, random_state=0).fit((MIMICtable[x_columns]-MIMICtable[x_columns].mean())/MIMICtable[x_columns].std())
        col_bins_obs = kmeans.labels_
        col_bins_sim = kmeans.predict((sim_data[x_columns] - MIMICtable[x_columns].mean())/MIMICtable[x_columns].std())
    else:
        bins, labels = [0], [0]
        for i in range(1, col_bins_num):
            bins.append(MIMICtable[column].quantile(i/col_bins_num))
            labels.append(i)
        bins.append(max(MIMICtable[column].max(), sim_data[column].max()))
        col_bins_obs = pd.cut(MIMICtable[column], bins=bins, labels=labels).astype(float)
        col_bins_sim = pd.cut(sim_data[column], bins=bins, labels=labels).astype(float)
    return col_bins_obs, col_bins_sim

def do_hypothesis_testing(column, MIMICtable, sim_data, col_bins_num, hyp_test_dir, use_kmeans, reverse_percentile, hoeffdings_test, pruning, discretised_outcome):
    logging.info("doing hypothesis testing")
    col_bins_obs, col_bins_sim = get_col_bins(MIMICtable, sim_data, use_kmeans, col_bins_num, column)
    logging.info("Extracted column bins for simulator data")

    trajectories = pd.DataFrame()
    trajectories.loc[:,'t'] = np.arange(len(MIMICtable))%5
    trajectories.loc[:,'icustay_id'] = MIMICtable['icustay_id']
    trajectories.loc[:,'gender'] = MIMICtable['gender']
    trajectories.loc[:,'age'] = MIMICtable['age']
    trajectories.loc[:,'x_t'] = col_bins_obs
    trajectories.loc[:,'A_t'] = MIMICtable['A']
    trajectories = trajectories[trajectories['t']!=4]
    
    sim_trajecs = pd.DataFrame()
    sim_trajecs.loc[:,'t'] = np.arange(len(sim_data))%5
    sim_trajecs.loc[:,'icustay_id'] = sim_data['icustay_id']
    sim_trajecs.loc[:,'x_t'] = col_bins_sim
    sim_trajecs = sim_trajecs[sim_trajecs['t']!=4]
    sim_trajecs = sim_trajecs.merge(trajectories[['t','icustay_id', 'A_t', 'gender', 'age']], left_on=['icustay_id', 't'], right_on=['icustay_id', 't'])
    
    trajec_actions = get_trajec_actions_df(trajectories)
    sim_trajec_actions = get_trajec_actions_df(sim_trajecs)
    logging.info(f'trajec_actions length: {len(trajec_actions)}')
    if discretised_outcome:
        MIMICtable[column] = col_bins_obs
        sim_data[column] = col_bins_sim

    icustayids = MIMICtable['icustay_id'].unique()
    logging.info(f'icustayids length: {len(icustayids)}')
    trajec_actions.loc[:,'icustay_id'] = icustayids
    logging.info("Simulator and Observational data ready")
    if use_kmeans:
        trajec_actions.to_csv(f'{hyp_test_dir}/trajec_actions.csv', index=False)
        sim_trajec_actions.to_csv(f'{hyp_test_dir}/sim_trajec_actions.csv', index=False)
    else:
        trajec_actions.to_csv(f'{hyp_test_dir}/trajec_actions_{column}.csv', index=False)
        sim_trajec_actions.to_csv(f'{hyp_test_dir}/sim_trajec_actions_{column}.csv', index=False)

    if hoeffdings_test:
        logging.info("Using hoeffdings for hypothesis testing")
        num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses = rejected_hypotheses_hoeffdings(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, pruning, discretised_outcome)
    else:
        num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses = rejected_hypotheses_bootstrap_percentile_with_qtwin(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, reverse_percentile, discretised_outcome)
    return num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses, trajec_actions, sim_trajec_actions

def do_hypothesis_testing_saved(column, directory, sim_data, MIMICtable, sofa_bin, use_kmeans, reverse_percentile, hoeffdings_test, pruning, discretised_outcome):
    if use_kmeans:
        trajec_actions = pd.read_csv(f"{directory}/trajec_actions.csv", converters={'actions': eval, 'x_t': eval})
        sim_trajec_actions = pd.read_csv(f"{directory}/sim_trajec_actions.csv", converters={'actions': eval, 'x_t': eval})
    else:
        trajec_actions = pd.read_csv(f"{directory}/trajec_actions_{column}.csv", converters={'actions': eval, 'x_t': eval})
        sim_trajec_actions = pd.read_csv(f"{directory}/sim_trajec_actions_{column}.csv", converters={'actions': eval, 'x_t': eval})
    if sofa_bin == 0:
        logging.info(f'Sofa bin: {sofa_bin}')
        trajec_actions = trajec_actions[trajec_actions['SOFA'] <= 6]
    elif sofa_bin == 1:
        logging.info(f'Sofa bin: {sofa_bin}')
        trajec_actions = trajec_actions[(trajec_actions['SOFA'] <= 12) & (trajec_actions['SOFA'] > 6)]
    elif sofa_bin == 2:
        logging.info(f'Sofa bin: {sofa_bin}')
        trajec_actions = trajec_actions[(trajec_actions['SOFA'] <= 18) & (trajec_actions['SOFA'] > 12)]
    elif sofa_bin == 3:
        logging.info(f'Sofa bin: {sofa_bin}')
        trajec_actions = trajec_actions[(trajec_actions['SOFA'] <= 24) & (trajec_actions['SOFA'] > 18)]
    sim_trajec_actions = sim_trajec_actions[sim_trajec_actions['icustay_id'].isin(trajec_actions['icustay_id'])]
    sim_data = sim_data[sim_data['icustay_id'].isin(trajec_actions['icustay_id'])]
    MIMICtable = MIMICtable[MIMICtable['icustay_id'].isin(trajec_actions['icustay_id'])]
    logging.info(f'Filtered trajectory length: {len(trajec_actions)}')

    if discretised_outcome:
        raise NotImplementedError("discretised outcome not implemented for saved hypotheses")

    if hoeffdings_test:
        logging.info("Using hoeffdings for hypothesis testing")
        num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses = rejected_hypotheses_hoeffdings(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, pruning, discretised_outcome)
    else:
        num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses = rejected_hypotheses_bootstrap_percentile_with_qtwin(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, reverse_percentile, discretised_outcome)
    return num_rej_hyps, p_values, rej_hyps, total_hypotheses, pruned_hypotheses, trajec_actions, sim_trajec_actions