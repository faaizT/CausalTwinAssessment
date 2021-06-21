import argparse
import logging
import pandas as pd
import numpy as np
import os
import glob
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from bareinboim_bounds.Utils import *

nra = 5
nr_reps = 1

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

column_mappings = {
    'Albumin - BloodConcentration (mg/L)': 'Albumin',
    'ArterialCarbonDioxidePressure (mmHg)': 'paCO2',
    'ArterialOxygenPressure (mmHg)': 'paO2',
    'Bicarbonate - BloodConcentration (mg/L)': 'HCO3',
    'BloodPH (None)': 'Arterial_pH',
    'Calcium - BloodConcentration (mg/L)': 'Calcium',
    'Chloride - BloodConcentration (mg/L)': 'Chloride',
    'Creatinine - BloodConcentration (mg/L)': 'Creatinine',
    'DiastolicArterialPressure (mmHg)': 'DiaBP',
    'Glucose - BloodConcentration (mg/L)': 'Glucose',
    'Lactate - BloodConcentration (mg/L)': 'Arterial_lactate',
    'MeanArterialPressure (mmHg)': 'MeanBP',
    'Potassium - BloodConcentration (mg/L)': 'Potassium',
    'RespirationRate (1/min)': 'RR',
    'SkinTemperature (degC)': 'Temp_C',
    'Sodium - BloodConcentration (mg/L)': 'Sodium',
    'SystolicArterialPressure (mmHg)': 'SysBP',
    'WhiteBloodCellCount (ct/uL)': 'WBC_count',
    'HeartRate (1/min)': 'HR'
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
    'Lactate': 'Lactate Blood Concentration (mg/L)',
    'MeanBP': 'Mean Arterial Pressure (mmHg)',
    'Potassium': 'Potassium Blood Concentration (mg/L)',
    'RR': 'Respiration Rate (1/min)',
    'Temp_C': 'Skin Temperature (C)',
    'Sodium': 'Sodium Blood Concentration (mg/L)',
    'WBC_count': 'White Blood Cell Count (ct/uL)',
    'HR': 'Heart Rate (1/min)'
}


def write_to_file(file_name, col_name, num_rejected):
    with open(file_name, 'a', 1) as f:
        f.write(col_name + ',' + str(num_rejected) + os.linesep)


def main(args):
    MIMICtable = pd.read_csv(args.obs_path + '/MIMIC-1hourly-length-5.csv')
    MIMICtable = MIMICtable.sort_values(by=['icustay_id', 'bloc'], ignore_index=True)
    age_ranked = rankdata(MIMICtable['age'])/len(MIMICtable)
    age_bins = np.floor((age_ranked + 0.2499999999)*4)
    median_ages = [MIMICtable.loc[age_bins==1, 'age'].median(), MIMICtable.loc[age_bins==2, 'age'].median(), MIMICtable.loc[age_bins==3, 'age'].median(), MIMICtable.loc[age_bins==4, 'age'].median()]
    MIMICtable = MIMICtable.rename(columns={"age": "age_raw"})
    MIMICtable['age'] = age_bins

    extension = 'final_.csv'
    all_filenames = [i for i in glob.glob(f'{args.pulse_path}/*{extension}')]
    pulse_data = pd.concat([pd.read_csv(f) for f in all_filenames ])
    pulse_data = pulse_data.rename(columns={'id': 'icustay_id'})
    pulse_data['icustay_id'] = pulse_data['icustay_id'].astype(int)
    pulse_data = pulse_data.reset_index(drop=True)
    pulse_data = pulse_data.sort_values(by=['icustay_id', 'SimulationTime(s)'], ignore_index=True)
    pulse_data['bloc'] = np.arange(len(pulse_data))%5 + 1

    pulse_rename = {}
    for k, v in column_mappings.items():
        pulse_rename.update({k: f"{v}"})

    pulse_data = pulse_data.rename(columns=pulse_rename)

    pulse_data = pulse_data.merge(MIMICtable[['gender', 'age', 'Weight_kg', 'icustay_id', 'bloc']], left_on=['icustay_id', 'bloc'], right_on=['icustay_id', 'bloc'])
    pulse_data = pulse_data.rename(columns={'age': 'age_raw'})


    logging.info('Creating action bins')
    nact = nra**2
    input_1hourly_nonzero = MIMICtable.loc[MIMICtable['input_1hourly']>0, 'input_1hourly']
    iol_ranked = rankdata(input_1hourly_nonzero)/len(input_1hourly_nonzero) # excludes zero fluid (will be action 1)
    iof = np.floor((iol_ranked + 0.2499999999)*4) # converts iv volume in 4 actions
    io = np.ones(len(MIMICtable)) # array of ones, by default
    io[MIMICtable['input_1hourly']>0] = iof + 1 # where more than zero fluid given: save actual action
    vc = MIMICtable['max_dose_vaso'].copy()
    vc_nonzero = MIMICtable.loc[MIMICtable['max_dose_vaso']!=0, 'max_dose_vaso']
    vc_ranked = rankdata(vc_nonzero)/len(vc_nonzero)
    vcf = np.floor((vc_ranked + 0.2499999999)*4) # converts to 4 bins
    vcf[vcf==0] = 1
    vc[vc!=0] = vcf + 1
    vc[vc==0] = 1
    # median dose of drug in all bins
    ma1 = [MIMICtable.loc[io==1, 'input_1hourly'].median(), MIMICtable.loc[io==2, 'input_1hourly'].median(), MIMICtable.loc[io==3, 'input_1hourly'].median(), MIMICtable.loc[io==4, 'input_1hourly'].median(), MIMICtable.loc[io==5, 'input_1hourly'].median()]
    ma2 = [MIMICtable.loc[vc==1, 'max_dose_vaso'].median(), MIMICtable.loc[vc==2, 'max_dose_vaso'].median(), MIMICtable.loc[vc==3, 'max_dose_vaso'].median(), MIMICtable.loc[vc==4, 'max_dose_vaso'].median(), MIMICtable.loc[vc==5, 'max_dose_vaso'].median()]
    med = pd.DataFrame(data={'IV':io, 'VC': vc})
    med = med.astype({'IV': 'int32', 'VC': 'int32'})
    uniqueValues = med.drop_duplicates().reset_index(drop=True)
    uniqueValueDoses = pd.DataFrame()
    for index, row in uniqueValues.iterrows():
        uniqueValueDoses.at[index, 'IV'], uniqueValueDoses.at[index, 'VC'] = ma1[row['IV']-1], ma2[row['VC']-1]

    actionbloc = pd.DataFrame()
    for index, row in med.iterrows():
        actionbloc.at[index, 'action_bloc'] = uniqueValues.loc[(uniqueValues['IV'] == row['IV']) & (uniqueValues['VC'] == row['VC'])].index.values[0]+1
    actionbloc = actionbloc.astype({'action_bloc':'int32'})
    logging.info('Action bins created')
    logging.info(f'Outcome: {args.col_name}')

    num_rej_hyps, p_values, rej_hyps, trajec_actions, pulse_trajec_actions  = do_hypothesis_testing(args.col_name, MIMICtable, pulse_data, args.col_bin_num, actionbloc)
    write_to_file(f'{args.hyp_test_dir}/rej_hyp_nums.csv', args.col_name, num_rej_hyps)
    trajec_actions.to_csv(f'{args.hyp_test_dir}/trajec_actions_{args.col_name}.csv')
    pulse_trajec_actions.to_csv(f'{args.hyp_test_dir}/pulse_trajec_actions_{args.col_name}.csv')
    rej_hyps.to_csv(f'{args.hyp_test_dir}/rej_hyps_{args.col_name}.csv')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--col_name", help="Column name to run hypothesis tests for", type=str, required=True)
    parser.add_argument("--col_bin_num", help="number of column bins", type=int, default=5)
    parser.add_argument("--obs_path", help="path to observational data directory", default="/data/ziz/taufiq/export-dir")
    parser.add_argument("--pulse_path", help="path to pulse data directory", default="/data/ziz/taufiq/pulse-data-5-step")
    parser.add_argument("--hyp_test_dir", help="Directory to save hypothesis test info", default="/data/ziz/taufiq/hyp-test-dir")
    args = parser.parse_args()
    if not os.path.exists(f'{args.hyp_test_dir}/rej_hyp_nums.csv'):
        with open(f'{args.hyp_test_dir}/rej_hyp_nums.csv', "w") as f:
            f.write('Outcome Y,# rejected hypotheses' + os.linesep)

    main(args)

