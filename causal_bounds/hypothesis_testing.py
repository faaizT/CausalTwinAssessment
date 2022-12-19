import pandas as pd
import numpy as np
from sklearn.utils import resample


def xn_in_bn(actions_of_interest, action_taken, x_trajec, xt_in_bt):
    for i in range(len(actions_of_interest)):
        if actions_of_interest[i] != action_taken[i]:
            break
    return xt_in_bt(x_trajec[:i+1])


def compute_pvalues_hoeffdings(obs_data, twin_data, ylo, yup):
    plo = 1*(obs_data['Ylo'].mean() <= twin_data['outcome'].mean()) + 4*np.exp(-2 * (1/(1/np.sqrt(len(obs_data)) + 1/np.sqrt(len(twin_data))) * (obs_data['Ylo'].mean() - twin_data['outcome'].mean())/(yup - ylo))**2)*(obs_data['Ylo'].mean() > twin_data['outcome'].mean())
    pup = 1*(obs_data['Yup'].mean() >= twin_data['outcome'].mean()) + 4*np.exp(-2 * (1/(1/np.sqrt(len(obs_data)) + 1/np.sqrt(len(twin_data))) * (twin_data['outcome'].mean() - obs_data['Yup'].mean())/(yup - ylo))**2)*(obs_data['Yup'].mean() < twin_data['outcome'].mean())
    plo, pup = min(1, plo), min(1, pup)
    return plo, pup


def compute_pvalues_bootstrap(obs_data, twin_data, ylo, yup, reverse_percentile, B=100):
    df = pd.DataFrame()
    Qobs_all_data = obs_data['outcome'].mean()
    Qlo_all_data = obs_data['Ylo'].mean()
    Qup_all_data = obs_data['Yup'].mean()
    Qtwin_all_data = twin_data['outcome'].mean()
    for j in range(B):
        obs_resampled = resample(obs_data, n_samples=len(obs_data))
        Qobs = obs_resampled['outcome'].mean()
        Qlo = obs_resampled['Ylo'].mean()
        Qup = obs_resampled['Yup'].mean()
        twin_resampled = resample(twin_data, n_samples=len(twin_data))
        Qtwin = twin_resampled['outcome'].mean()
        row_append = pd.DataFrame.from_dict({'Qobs': [Qobs], 'Qlo': [Qlo], 'Qup': [Qup], 'Qobs_all_data': [Qobs_all_data], 'Qtwin': [Qtwin], 'Qtwin_all_data': [Qtwin_all_data], 'ylo': [ylo], 'yup': [yup], 'Qlo_all_data': [Qlo_all_data], 'Qup_all_data': [Qup_all_data],})
        df = pd.concat([df, row_append], ignore_index=True)
    p_lb, p_ub = 1, 1
    for alpha in np.linspace(1e-6, 1.0, 500):
        if reverse_percentile:
            twin_lb = 2*df['Qtwin_all_data'].mean() - df['Qtwin'].quantile(1-alpha)
            twin_ub = 2*df['Qtwin_all_data'].mean() - df['Qtwin'].quantile(alpha)
            lb = 2*df['Qlo_all_data'].mean() - df['Qlo'].quantile(1 - alpha)
            ub = 2*df['Qup_all_data'].mean() - df['Qup'].quantile(alpha)
        else:
            twin_lb = df['Qtwin'].quantile(alpha)
            twin_ub = df['Qtwin'].quantile(1 - alpha)
            lb = df['Qlo'].quantile(alpha)
            ub = df['Qup'].quantile(1 - alpha)
        if twin_ub < lb and p_lb == 1:
            p_lb = alpha
        if twin_lb > ub and p_ub == 1:
            p_ub = alpha
    return round(p_lb, 6), round(p_ub, 6), df


def test_hypothesis(obs_data, twin_data, f, at, xt_in_bt, ylo, yup, hoeffdings=True, reverse_percentile=True):
    """
    Returns p-values for H_{lo} and H_{up} for a given set of parameters.
    If using bootstrapping, the function also outputs the dataframe with bootstrapped distributions

    Parameters
    ----------
    obs_data    : pd.DataFrame
        observational dataset
    twin_data   : pd.DataFrame
        twin simulated dataset
    f           : callable
        the outcome function, which takes X_{0:t}(a_{1:t}) as input and outputs a scalar.
    at          : array
        action sequence $a_{1:t}$
    xt_in_bt    : callable
        takes $X_{0:t'}(a_{1:t'})$ as input for some $t' ≤ t$
        and outputs a boolean, corresponding to whether $X_{0:t'}(a_{1:t'}) \in B_{0:t'}$
    hoeffdings  : bool
        use Hoeffding's inequality for hypothesis testing
    """
    obs_data_filtered = obs_data.loc[obs_data.apply(lambda x: xn_in_bn(at, x['actions'], x['x_t'], xt_in_bt), axis=1)].copy()
    twin_data_filtered = twin_data.loc[(twin_data['actions']==at) & (twin_data['x_t'].apply(lambda x: xt_in_bt(x)))].copy()
    obs_data_filtered.loc[:, 'outcome'] = obs_data_filtered['x_t'].apply(lambda x: f(x))
    twin_data_filtered.loc[:, 'outcome'] = twin_data_filtered['x_t'].apply(lambda x: f(x))
    obs_data_filtered.loc[:, 'Ylo'] = (obs_data_filtered['actions']==at)*obs_data_filtered['outcome'] + (obs_data_filtered['actions']!=at)*ylo
    obs_data_filtered.loc[:, 'Yup'] = (obs_data_filtered['actions']==at)*obs_data_filtered['outcome'] + (obs_data_filtered['actions']!=at)*yup
    df = None
    if hoeffdings:
        plo, pup = compute_pvalues_hoeffdings(obs_data_filtered, twin_data_filtered, ylo, yup)
    else:
        plo, pup, df =  compute_pvalues_bootstrap(obs_data_filtered, twin_data_filtered, ylo, yup, reverse_percentile)
    return plo, pup, df

