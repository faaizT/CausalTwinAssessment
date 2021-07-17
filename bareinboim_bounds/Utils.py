import pandas as pd
import numpy as np
import logging
from ast import literal_eval
from sklearn.utils import resample
import scipy.stats as st
from scipy.stats import rankdata


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def get_trajec_actions(trajectories, column):
    trajec_actions = pd.DataFrame()
    for index, row in trajectories.iterrows():
        if row['t'] == 0 and index > 0:
            trajec_actions = trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
            age = row['age']
        elif index == 0:
            age = row['age']
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
        else:
            age = row['age']
            actions.append(row['A_t'])
            col_traj.append(row[column])
            gender = row['gender']
    trajec_actions = trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
    return trajec_actions


def get_sim_trajec_actions(sim_trajecs, column):
    sim_trajec_actions = pd.DataFrame()
    for index, row in sim_trajecs.iterrows():
        if row['t'] == 0 and index > 0:
            sim_trajec_actions = sim_trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
            age = row['age']
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
        elif index == 0:
            age = row['age']
            actions = [row['A_t']]
            col_traj = [row[column]]
            gender = row['gender']
        else:
            age = row['age']
            actions.append(row['A_t'])
            col_traj.append(row[column])
            gender = row['gender']
    sim_trajec_actions = sim_trajec_actions.append({'actions': actions, 'gender': gender, 'age': age, column: col_traj}, ignore_index=True)
    sim_trajec_actions['icustay_id'] = sim_trajecs['icustay_id'].unique()
    return sim_trajec_actions


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


def compute_probs(trajec_actions, column, column_v, gender, age, actions):
    prob_a_den = 0
    gamma = []
    df = pd.DataFrame()
    df[[f'{column}_0',f'{column}_1',f'{column}_2',f'{column}_3']] = pd.DataFrame(trajec_actions[column].tolist(), index= trajec_actions.index)
    df[[f'a_0',f'a_1',f'a_2',f'a_3']] = pd.DataFrame(trajec_actions['actions'].tolist(), index= trajec_actions.index)
    trajecs_filtered = (trajec_actions['gender'] == gender) & (trajec_actions['age'] == age)
    num_term = trajecs_filtered
    den_term_1 = trajecs_filtered
    den_term_2 = trajecs_filtered
    for i in range(len(actions)):
        num_term = num_term & (df[f'{column}_{i}'] == column_v[i]) & (df[f'a_{i}'] == actions[i])
        prob_a_num = num_term.sum()
        if i == 0:
            den_term_1 = den_term_1 & (df[f'{column}_{i}'] == column_v[i])
            prob_a_den += den_term_1.sum()
        else:
            den_term_1 = den_term_1 & (df[f'{column}_{i}'] == column_v[i]) & (df[f'a_{i-1}'] == actions[i-1])
            den_term_2 = den_term_2 & (df[f'{column}_{i-1}'] == column_v[i-1]) & (df[f'a_{i-1}'] == actions[i-1])
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


def bootstrap_distribution_(col, gender, age, action, column_v, trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=3):
    sim_data = sim_data.rename(columns={col: f'{col}_raw'})
    MIMICtable = MIMICtable.rename(columns={col: f'{col}_raw'})
    sim = sim_trajec_actions[['actions', 'gender', 'age', 'icustay_id', col]].merge(sim_data[sim_data['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    obs_data = trajec_actions[['actions', 'gender', 'age', 'icustay_id', col]].merge(MIMICtable[MIMICtable['bloc'] == i+2], left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])    
    sim.loc[:,f'{col}_tuple'] = sim[col].apply(tuple)
    sim.loc[:,f'actions_tuple'] = sim['actions'].apply(tuple)
    obs_data.loc[:,f'{col}_tuple'] = obs_data[col].apply(tuple)
    obs_data.loc[:,f'actions_tuple'] = obs_data['actions'].apply(tuple)
    df = pd.DataFrame()
    max_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & find_elements_containing(obs_data[col], max(column_v)), f'{col}_raw'].max()
    min_y = obs_data.loc[(obs_data['gender'] == gender) & (obs_data['age'] == age) & find_elements_containing(obs_data[col], min(column_v)), f'{col}_raw'].min()
    sim_filtered = sim[(sim['gender'] == gender) & (sim['age'] == age) & (sim[f'{col}_tuple']==tuple(column_v)) & (sim['actions_tuple'] == tuple(action))].copy()
    real_filtered = obs_data[(obs_data['gender'] == gender) & (obs_data['age'] == age) & (obs_data[f'{col}_tuple']==tuple(column_v)) & (obs_data['actions_tuple'] == tuple(action))].copy()
    if len(real_filtered) > 1 and len(sim_filtered) > 1:
        for j in range(n_iter):
            obs_resampled = resample(obs_data, n_samples=len(obs_data))
            probs = compute_probs(obs_resampled, col, column_v, gender, age, action)
            prob = probs[i]
            real_train = resample(real_filtered, n_samples=len(real_filtered))
            exp_y = real_train[f'{col}_raw'].mean()
            sim_train = resample(sim_filtered, n_samples=len(sim_filtered))
            exp_y_sim = sim_train[f'{col}_raw'].mean()
            df = df.append({'Exp_y': exp_y, 'UB': prob*exp_y + (1-prob)*max_y, 'LB': prob*exp_y + (1-prob)*min_y, 'Sim_exp_y': exp_y_sim, 'max_y':max_y, 'min_y': min_y}, ignore_index=True)
        return df
    return None


def rejected_hypotheses_bootstrap(col, trajec_actions, sim_trajec_actions):
    logging.info("calculating rejected hypotheses")
    state_actions = trajec_actions[['gender', 'age', 'actions', col]].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions[col].apply(tuple)
    state_actions = state_actions.drop_duplicates(['gender', 'age', 'a', 's'])
    total_hypotheses = len(state_actions)
    logging.info(f"Total hypotheses: {total_hypotheses}")
    p_values = pd.DataFrame()
    counter = 0
    for index, row in state_actions.iterrows():
        counter += 1
        logging.info(f"On hypothesis {counter}/{total_hypotheses}")
        df = bootstrap_distribution(col, row['gender'], row['age'], row['actions'], row[col], trajec_actions, sim_trajec_actions)
        if df is not None:
            sigma_ub = (df['UB']-df['Sim_exp_y']).var()
            exp_ub = (df['UB']-df['Sim_exp_y']).mean()
            p_ub = st.norm.cdf(exp_ub/np.sqrt(sigma_ub))
            sigma_lb = (df['Sim_exp_y']-df['LB']).var()
            exp_lb = (df['Sim_exp_y']-df['LB']).mean()
            p_lb = st.norm.cdf(exp_lb/np.sqrt(sigma_lb))
            p_values = p_values.append({'gender': row['gender'], 'age': row['age'], 'actions': row['actions'], col: row[col], 'p_lb': p_lb, 'p_ub': p_ub}, ignore_index=True)
    rej_hyps = p_values[(p_values['p_lb']<0.05/total_hypotheses) ^ (p_values['p_ub']<0.05/total_hypotheses)].copy()
    for index, row in rej_hyps.iterrows():
        rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions[col], row[col])).sum()
        rej_hyps.loc[index, 'n_sim'] = (find_elements(sim_trajec_actions['gender'], row['gender']) & find_elements(sim_trajec_actions['age'], row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions[col], row[col])).sum()
    return len(rej_hyps), p_values, rej_hyps, total_hypotheses


def rejected_hypotheses_bootstrap_trajectories(col, trajec_actions, sim_trajec_actions, sim_data, MIMICtable):
    T = 4
    state_actions = trajec_actions[['gender', 'age', 'actions', col]].copy()
    state_actions.loc[:,'a'] = state_actions['actions'].apply(tuple)
    state_actions.loc[:,'s'] = state_actions[col].apply(tuple)
    state_actions = state_actions.drop_duplicates(['gender', 'age', 'a', 's'])
    total_hypotheses = len(state_actions)
    p_values = pd.DataFrame()
    counter = 0
    for index, row in state_actions.iterrows():
        counter += 1
        logging.info(f"On hypothesis {counter}/{total_hypotheses}")
        p = 1
        p_ub, p_lb = [], []
        for t in range(T):
            df = bootstrap_distribution_(col, row['gender'], row['age'], row['actions'], row[col], trajec_actions, sim_trajec_actions, sim_data, MIMICtable, n_iter=100, i=t)
            if df is not None:
                # sigma_ub = (df['UB']-df['Sim_exp_y']).var()
                # exp_ub = (df['UB']-df['Sim_exp_y']).mean()
                # p_ub.append(st.norm.cdf(exp_ub/np.sqrt(sigma_ub)))
                # sigma_lb = (df['Sim_exp_y']-df['LB']).var()
                # exp_lb = (df['Sim_exp_y']-df['LB']).mean()
                # p_lb.append(st.norm.cdf(exp_lb/np.sqrt(sigma_lb)))
                # p = min(p, p_ub[-1] + p_lb[-1] - 1)
                p = min(p, ((df['LB'] <= df['Sim_exp_y']) & (df['UB'] >= df['Sim_exp_y'])).sum()/len(df))
        if df is not None:
            p_values = p_values.append({'gender': row['gender'], 'age': row['age'], 'actions': row['actions'], col: row[col], 'p': p}, ignore_index=True)
    if len(p_values) > 0:
        rej_hyps = p_values[(p_values['p']<0.05/total_hypotheses/T)].copy()
        for index, row in rej_hyps.iterrows():
            rej_hyps.loc[index, 'n_real'] = (find_elements(trajec_actions['gender'], row['gender']) & find_elements(trajec_actions['age'], row['age']) & find_elements(trajec_actions['actions'], row['actions']) & find_elements(trajec_actions[col], row[col])).sum()
            rej_hyps.loc[index, 'n_sim'] = (find_elements(sim_trajec_actions['gender'], row['gender']) & find_elements(sim_trajec_actions['age'], row['age']) & find_elements(sim_trajec_actions['actions'], row['actions']) & find_elements(sim_trajec_actions[col], row[col])).sum()
    else:
        rej_hyps = pd.DataFrame()
    return len(rej_hyps), p_values, rej_hyps, total_hypotheses


def do_hypothesis_testing(column, MIMICtable, sim_data, col_bins_num, actionbloc):
    logging.info("doing hypothesis testing")
    sim_data = sim_data.rename(columns={column: f'{column}_raw'})
    
    col_ranked = rankdata(MIMICtable[column])/len(MIMICtable)
    col_bins = np.floor((col_ranked + 1/(col_bins_num + 0.0000001))*col_bins_num)
    median_col = [MIMICtable.loc[col_bins==1, column].median(), MIMICtable.loc[col_bins==2, column].median(), MIMICtable.loc[col_bins==3, column].median(), MIMICtable.loc[col_bins==4, column].median()]
    
    MIMICtable = MIMICtable.rename(columns={column: f'{column}_raw'})
    MIMICtable[column] = col_bins
    sim_data = sim_data.merge(MIMICtable[['age', 'icustay_id', 'bloc', column]], left_on=['icustay_id', 'bloc', 'age'], right_on=['icustay_id', 'bloc', 'age'])
    
    trajectories = pd.DataFrame()
    trajectories['t'] = np.arange(len(MIMICtable))%5
    trajectories['icustay_id'] = MIMICtable['icustay_id']
    trajectories['gender'] = MIMICtable['gender']
    trajectories['age'] = MIMICtable['age']
    trajectories[column] = MIMICtable[column]
    trajectories['A_t'] = actionbloc['action_bloc']
    trajectories = trajectories[trajectories['t']!=4]
    
    sim_trajecs = pd.DataFrame()
    sim_trajecs['t'] = np.arange(len(sim_data))%5
    sim_trajecs['icustay_id'] = sim_data['icustay_id']
    sim_trajecs = sim_trajecs[sim_trajecs['t']!=4]
    sim_trajecs = sim_trajecs.merge(trajectories[['t','icustay_id', 'A_t', 'gender', 'age', column]], left_on=['icustay_id', 't'], right_on=['icustay_id', 't'])
    
    trajec_actions = get_trajec_actions(trajectories, column)
    sim_trajec_actions = get_sim_trajec_actions(sim_trajecs, column)
    
    icustayids = MIMICtable['icustay_id'].unique()
    trajec_actions['icustay_id'] = icustayids
    
    mimic_data_last_time = MIMICtable[MIMICtable['bloc'] == 5].drop(columns=column)
    trajec_actions = trajec_actions.merge(mimic_data_last_time, left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])

    sim_data_last_time = sim_data[sim_data['bloc'] == 5].drop(columns=column)
    sim_trajec_actions = sim_trajec_actions.merge(sim_data_last_time, left_on=['icustay_id', 'gender', 'age'], right_on=['icustay_id', 'gender', 'age'])
    
    num_rej_hyps, p_values, rej_hyps, total_hypotheses = rejected_hypotheses_bootstrap_trajectories(column, trajec_actions, sim_trajec_actions, sim_data, MIMICtable)
    return num_rej_hyps, p_values, rej_hyps, total_hypotheses, trajec_actions, sim_trajec_actions

def do_hypothesis_testing_saved(column, directory, sim_data, MIMICtable, sofa_bin):
    rej_hyps = pd.read_csv(f"{directory}/rej_hyps_{column}.csv", converters={'actions': eval, column: eval})
    trajec_actions = pd.read_csv(f"{directory}/trajec_actions_{column}.csv", converters={'actions': eval, column: eval})
    sim_trajec_actions = pd.read_csv(f"{directory}/sim_trajec_actions_{column}.csv", converters={'actions': eval, column: eval})
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