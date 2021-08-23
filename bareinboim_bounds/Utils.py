import pandas as pd
import numpy as np
import logging
from ast import literal_eval
from sklearn.utils import resample
import scipy.stats as st
from scipy.stats import rankdata
from sklearn.cluster import KMeans

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
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y}, ignore_index=True)
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


def rejected_hypotheses_bootstrap_trajectories(col, trajec_actions, sim_trajec_actions, sim_data, MIMICtable):
    T = 4
    state_actions = trajec_actions[['gender', 'age', 'actions', 'x_t']].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions['x_t'].apply(tuple)
    state_actions = state_actions.groupby(by=['gender', 'age', 'a', 's']).filter(lambda x: len(x) >= 25).drop_duplicates(['gender', 'age', 'a', 's'])
    total_hypotheses = len(state_actions)
    p_values = pd.DataFrame()
    counter = 0
    for index, row in state_actions.iterrows():
        counter += 1
        logging.info(f"On hypothesis {counter}/{total_hypotheses}")
        p = 1
        p_ub, p_lb = [], []
        for t in range(T):
            df = bootstrap_distribution_(col, row['gender'], row['age'], row['actions'], row['x_t'], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=t)
            if df is not None:
                p = min(p, ((df['LB'] <= df['Sim_exp_y']) & (df['UB'] >= df['Sim_exp_y'])).sum()/len(df))
        if df is not None:
            p_values = p_values.append({'gender': row['gender'], 'age': row['age'], 'actions': row['actions'], 'x_t': row['x_t'], 'p': p}, ignore_index=True)
    if len(p_values) > 0:
        rej_hyps = p_values[(p_values['p']<0.05/total_hypotheses/T)].copy()
        for index, row in rej_hyps.iterrows():
            rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions['x_t'], row['x_t'])).sum()
            rej_hyps.loc[index, 'n_sim'] = (find_elements(sim_trajec_actions['gender'], row['gender']) & find_elements(sim_trajec_actions['age'], row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions['x_t'], row['x_t'])).sum()
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

def get_col_bins(MIMICtable, sim_data, use_kmeans):
    if use_kmeans:
        kmeans = KMeans(n_clusters=4, random_state=0).fit((MIMICtable[x_columns]-MIMICtable[x_columns].mean())/MIMICtable[x_columns].std())
        col_bins_obs = kmeans.labels_
        col_bins_sim = kmeans.predict((sim_data[x_columns] - MIMICtable[x_columns].mean())/MIMICtable[x_columns].std())
    else:
        col_ranked = rankdata(pd.concat([MIMICtable[column], sim_data[column]]))/len(MIMICtable)
        col_bins = np.floor((col_ranked + 1/(col_bins_num + 0.0000001))*col_bins_num)
        col_bins_obs = col_bins[:len(MIMICtable)]
        col_bins_sim = col_bins[len(MIMICtable):]
    return col_bins_obs, col_bins_sim

def do_hypothesis_testing(column, MIMICtable, sim_data, col_bins_num, hyp_test_dir, use_kmeans):
    logging.info("doing hypothesis testing")
    col_bins_obs, col_bins_sim = get_col_bins(MIMICtable, sim_data, use_kmeans)
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
    icustayids = MIMICtable['icustay_id'].unique()
    logging.info(f'icustayids length: {len(icustayids)}')
    trajec_actions.loc[:,'icustay_id'] = icustayids
    logging.info("Simulator and Observational data ready")
    
    trajec_actions.to_csv(f'{hyp_test_dir}/trajec_actions_{column}.csv', index=False)
    sim_trajec_actions.to_csv(f'{hyp_test_dir}/sim_trajec_actions_{column}.csv', index=False)

    num_rej_hyps, p_values, rej_hyps, total_hypotheses = rejected_hypotheses_bootstrap_trajectories(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable)
    return num_rej_hyps, p_values, rej_hyps, total_hypotheses, trajec_actions, sim_trajec_actions

def do_hypothesis_testing_saved(column, directory, sim_data, MIMICtable, sofa_bin):
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

    num_rej_hyps, p_values, rej_hyps, total_hypotheses = rejected_hypotheses_bootstrap_trajectories(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable)
    return num_rej_hyps, p_values, rej_hyps, total_hypotheses, trajec_actions, sim_trajec_actions