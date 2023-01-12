import argparse
import logging
import pandas as pd
import numpy as np
import os
import glob
import re
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from pulse.Utils import *
from utils import str2bool

nra = 5
nr_reps = 1

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

column_names_unit = {
    "Albumin": "Albumin Blood Concentration (mg/L)",
    "paCO2": "Arterial $CO_2$ Pressure (mmHg)",
    "paO2": "Arterial $O_2$ Pressure (mmHg)",
    "HCO3": "Bicarbonate Blood Concentration (mg/L)",
    "Arterial_pH": "Arterial pH",
    "Arterial_lactate": "Lactate Blood Concentration (mg/L)",
    "Calcium": "Calcium Blood Concentration (mg/L)",
    "Chloride": "Chloride Blood Concentration (mg/L)",
    "Creatinine": "Creatinine Blood Concentration (mg/L)",
    "DiaBP": "Diastolic Arterial Pressure (mmHg)",
    "SysBP": "Systolic Arterial Pressure (mmHg)",
    "Glucose": "Glucose Blood Concentration (mg/L)",
    "Lactate": "Lactate Blood Concentration (mg/L)",
    "MeanBP": "Mean Arterial Pressure (mmHg)",
    "Potassium": "Potassium Blood Concentration (mg/L)",
    "RR": "Respiration Rate (1/min)",
    "Temp_C": "Skin Temperature (C)",
    "Sodium": "Sodium Blood Concentration (mg/L)",
    "WBC_count": "White Blood Cell Count (ct/uL)",
    "HR": "Heart Rate (1/min)",
}


def load_pulse_data(args, MIMICtable):
    column_mappings = {
        "Albumin - BloodConcentration (mg/L)": "Albumin",
        "ArterialCarbonDioxidePressure (mmHg)": "paCO2",
        "ArterialOxygenPressure (mmHg)": "paO2",
        "Bicarbonate - BloodConcentration (mg/L)": "HCO3",
        "BloodPH (None)": "Arterial_pH",
        "Calcium - BloodConcentration (mg/L)": "Calcium",
        "Chloride - BloodConcentration (mg/L)": "Chloride",
        "Creatinine - BloodConcentration (mg/L)": "Creatinine",
        "DiastolicArterialPressure (mmHg)": "DiaBP",
        "Glucose - BloodConcentration (mg/L)": "Glucose",
        "Lactate - BloodConcentration (mg/L)": "Arterial_lactate",
        "MeanArterialPressure (mmHg)": "MeanBP",
        "Potassium - BloodConcentration (mg/L)": "Potassium",
        "RespirationRate (1/min)": "RR",
        "SkinTemperature (degC)": "Temp_C",
        "Sodium - BloodConcentration (mg/L)": "Sodium",
        "SystolicArterialPressure (mmHg)": "SysBP",
        "WhiteBloodCellCount (ct/uL)": "WBC_count",
        "HeartRate (1/min)": "HR",
    }
    if args.preprocess_pulse_data:
        extension = "final_.csv"
        all_filenames = [i for i in glob.glob(f"{args.sim_data_path}/*{extension}")]
        if len(all_filenames) == 0:
            raise ValueError(
                f"No files found ending with {extension} tag. Make sure sim_data_path points to the directory containing pulse generated raw files"
            )
        logging.info(f"Total number of simulated trajectories: {len(all_filenames)}")
        sim_data = pd.concat([pd.read_csv(f) for f in all_filenames])
        sim_data["icustay_id"] = sim_data["icustay_id"].astype(int)
        sim_data["bloc"] = np.arange(len(sim_data)) % 5 + 1
        sim_data = sim_data.reset_index(drop=True)
        sim_data["order"] = np.arange(len(sim_data))

        sim_rename = {}
        for k, v in column_mappings.items():
            sim_rename.update({k: f"{v}"})

        sim_data = sim_data.rename(columns=sim_rename)
        sim_data = sim_data.merge(
            MIMICtable[["icustay_id", "age"]].drop_duplicates(inplace=False),
            left_on=[
                "icustay_id",
            ],
            right_on=[
                "icustay_id",
            ],
            sort=False,
            how="left",
        )
        sim_data.sort_values(by=["order"], inplace=True)
    else:
        sim_data = pd.read_csv(args.sim_data_path)
    sim_data = find_action_bin(MIMICtable, sim_data)
    return sim_data


def create_action_bins(MIMICtable):
    logging.info("Creating action bins")
    nact = nra**2
    input_1hourly_nonzero = MIMICtable.loc[
        MIMICtable["input_1hourly"] > 0, "input_1hourly"
    ]
    iol_ranked = rankdata(input_1hourly_nonzero) / len(
        input_1hourly_nonzero
    )  # excludes zero fluid (will be action 1)
    iof = np.floor((iol_ranked + 0.2499999999) * 4)  # converts iv volume in 4 actions
    io = np.ones(len(MIMICtable))  # array of ones, by default
    io[MIMICtable["input_1hourly"] > 0] = (
        iof + 1
    )  # where more than zero fluid given: save actual action
    vc = MIMICtable["max_dose_vaso"].copy()
    vc_nonzero = MIMICtable.loc[MIMICtable["max_dose_vaso"] != 0, "max_dose_vaso"]
    vc_ranked = rankdata(vc_nonzero) / len(vc_nonzero)
    vcf = np.floor((vc_ranked + 0.2499999999) * 4)  # converts to 4 bins
    vcf[vcf == 0] = 1
    vc[vc != 0] = vcf + 1
    vc[vc == 0] = 1
    # median dose of drug in all bins
    ma1 = [
        MIMICtable.loc[io == 1, "input_1hourly"].median(),
        MIMICtable.loc[io == 2, "input_1hourly"].median(),
        MIMICtable.loc[io == 3, "input_1hourly"].median(),
        MIMICtable.loc[io == 4, "input_1hourly"].median(),
        MIMICtable.loc[io == 5, "input_1hourly"].median(),
    ]
    ma2 = [
        MIMICtable.loc[vc == 1, "max_dose_vaso"].median(),
        MIMICtable.loc[vc == 2, "max_dose_vaso"].median(),
        MIMICtable.loc[vc == 3, "max_dose_vaso"].median(),
        MIMICtable.loc[vc == 4, "max_dose_vaso"].median(),
        MIMICtable.loc[vc == 5, "max_dose_vaso"].median(),
    ]
    med = pd.DataFrame(data={"IV": io, "VC": vc})
    med = med.astype({"IV": "int32", "VC": "int32"})
    uniqueValues = med.drop_duplicates().reset_index(drop=True)
    uniqueValueDoses = pd.DataFrame()
    for index, row in uniqueValues.iterrows():
        uniqueValueDoses.at[index, "IV"], uniqueValueDoses.at[index, "VC"] = (
            ma1[row["IV"] - 1],
            ma2[row["VC"] - 1],
        )

    actionbloc = pd.DataFrame()
    for index, row in med.iterrows():
        actionbloc.at[index, "action_bloc"] = (
            uniqueValues.loc[
                (uniqueValues["IV"] == row["IV"]) & (uniqueValues["VC"] == row["VC"])
            ].index.values[0]
            + 1
        )
    actionbloc = actionbloc.astype({"action_bloc": "int32"})
    logging.info("Action bins created")
    MIMICtable["A"] = actionbloc
    return MIMICtable


def find_action_bin(obs_data, twin_data):
    logging.info(
        "Finding the action bins corresponding to each administered dose in twin"
    )
    max_doses = obs_data[["A", "input_1hourly", "max_dose_vaso"]].groupby(by="A").max()
    min_doses = obs_data[["A", "input_1hourly", "max_dose_vaso"]].groupby(by="A").min()
    twin_data["A"] = twin_data.apply(
        lambda x: min_doses[
            (x["input_1hourly"] <= max_doses["input_1hourly"])
            & (x["input_1hourly"] >= min_doses["input_1hourly"])
            & (x["median_dose_vaso"] <= max_doses["max_dose_vaso"])
            & (x["median_dose_vaso"] >= min_doses["max_dose_vaso"])
        ].index.values[0],
        axis=1,
    )
    return twin_data


def load_mimic_data(obs_path):
    if not os.path.exists(obs_path):
        raise ValueError(
            f"Observational data not found at {obs_path}. Please specify the correct path to observational data using --obs_path argument"
        )
    MIMICtable = pd.read_csv(obs_path)
    MIMICtable = MIMICtable.sort_values(by=["icustay_id", "bloc"], ignore_index=True)
    # only include patients for whom the first 5 hours of data is available
    MIMICtable = (
        MIMICtable.groupby("icustay_id")
        .filter(lambda x: list(x["bloc"])[:5] == list(range(1, 6)))
        .groupby(by="icustay_id")
        .head(5)
    )
    age_ranked = rankdata(MIMICtable["age"]) / len(MIMICtable)
    age_bins = np.floor((age_ranked + 0.2499999999) * 4)
    median_ages = [
        MIMICtable.loc[age_bins == 1, "age"].median(),
        MIMICtable.loc[age_bins == 2, "age"].median(),
        MIMICtable.loc[age_bins == 3, "age"].median(),
        MIMICtable.loc[age_bins == 4, "age"].median(),
    ]
    MIMICtable = MIMICtable.rename(columns={"age": "age_raw"})
    MIMICtable["age"] = age_bins
    return MIMICtable


def get_actions_gender_age_df(trajectories):
    median_doses = (
        trajectories.groupby(by=["A"])
        .median()[["input_1hourly", "median_dose_vaso"]]
        .reset_index()
    )
    trajec_actions = trajectories.drop(columns=["input_1hourly", "median_dose_vaso"])
    trajec_actions = pd.merge(
        left=trajec_actions, right=median_doses, left_on="A", right_on="A"
    )
    trajec_actions = trajectories.loc[trajectories["bloc"] == 1].reset_index(drop=True)
    trajec_actions_t1 = trajectories.loc[trajectories["bloc"] == 2].reset_index(
        drop=True
    )
    trajec_actions_t2 = trajectories.loc[trajectories["bloc"] == 3].reset_index(
        drop=True
    )
    trajec_actions_t3 = trajectories.loc[trajectories["bloc"] == 4].reset_index(
        drop=True
    )
    trajec_actions_t1.rename(
        columns={
            "A": "A_1",
            "input_1hourly": "input_1hourly_1",
            "median_dose_vaso": "median_dose_vaso_1",
        },
        inplace=True,
    )
    trajec_actions_t2.rename(
        columns={
            "A": "A_2",
            "input_1hourly": "input_1hourly_2",
            "median_dose_vaso": "median_dose_vaso_2",
        },
        inplace=True,
    )
    trajec_actions_t3.rename(
        columns={
            "A": "A_3",
            "input_1hourly": "input_1hourly_3",
            "median_dose_vaso": "median_dose_vaso_3",
        },
        inplace=True,
    )
    trajec_actions = pd.merge(
        left=trajec_actions,
        right=trajec_actions_t1[["A_1", "input_1hourly_1", "median_dose_vaso_1"]],
        left_index=True,
        right_index=True,
    )
    trajec_actions = pd.merge(
        left=trajec_actions,
        right=trajec_actions_t2[["A_2", "input_1hourly_2", "median_dose_vaso_2"]],
        left_index=True,
        right_index=True,
    )
    trajec_actions = pd.merge(
        left=trajec_actions,
        right=trajec_actions_t3[["A_3", "input_1hourly_3", "median_dose_vaso_3"]],
        left_index=True,
        right_index=True,
    )
    trajec_actions.loc[:, "actions"] = trajec_actions.loc[
        :, ["A", "A_1", "A_2", "A_3"]
    ].apply(lambda x: tuple(x), axis=1)
    trajec_actions.loc[:, "input_1hourly"] = trajec_actions.loc[
        :, ["input_1hourly", "input_1hourly_1", "input_1hourly_2", "input_1hourly_3"]
    ].apply(lambda x: tuple(x), axis=1)
    trajec_actions.loc[:, "median_dose_vaso"] = trajec_actions.loc[
        :,
        [
            "median_dose_vaso",
            "median_dose_vaso_1",
            "median_dose_vaso_2",
            "median_dose_vaso_3",
        ],
    ].apply(lambda x: tuple(x), axis=1)
    action_count = (
        trajec_actions["actions"]
        .value_counts()
        .reset_index()
        .rename(columns={"actions": "actions_count", "index": "actions"}, inplace=False)
    )
    trajec_actions = pd.merge(
        left=trajec_actions, right=action_count, left_on="actions", right_on="actions"
    )
    return trajec_actions.loc[
        :,
        [
            "actions",
            "gender",
            "age",
            "actions_count",
            "input_1hourly",
            "median_dose_vaso",
        ],
    ]


def load_observational_data_and_split(args, load=False):
    MIMICtable = load_mimic_data(args.obs_path)
    MIMICtable = create_action_bins(MIMICtable)
    icustay_ids_held_back = MIMICtable.loc[
        np.random.binomial(1, 0.01, (len(MIMICtable),)) == 1, "icustay_id"
    ].unique()
    MIMIC_data_held_back = (
        MIMICtable[MIMICtable["icustay_id"].isin(icustay_ids_held_back)]
        .reset_index(drop=True)
        .copy()
    )
    MIMIC_data_held_back.to_csv(
        args.hyp_test_dir + "/MIMIC-1hourly-held-back.csv", index=False
    )
    MIMIC_data_used = (
        MIMICtable[~(MIMICtable["icustay_id"].isin(icustay_ids_held_back))]
        .reset_index(drop=True)
        .copy()
    )
    MIMIC_data_used.to_csv(
        args.hyp_test_dir + "/MIMIC-1hourly-not-held-back.csv", index=False
    )
    actions_df = get_actions_gender_age_df(MIMIC_data_held_back)
    actions_df.drop_duplicates().reset_index(drop=True).to_csv(
        args.hyp_test_dir + "/MIMIC-actions-to-generate.csv", index=False
    )
    return MIMICtable, MIMIC_data_held_back, MIMIC_data_used


def main(args):
    np.random.seed(0)
    if args.col_name not in column_names_unit:
        raise ValueError(f"Column name must be one of {list(column_names_unit.keys())}")

    (
        MIMICtable,
        MIMIC_data_held_back,
        MIMIC_data_used,
    ) = load_observational_data_and_split(args, load=True)
    sim_data = load_pulse_data(args, MIMICtable)

    logging.info(f"Outcome: {args.col_name}")
    (
        num_rej_hyps,
        p_values,
        rej_hyps,
        total_hypotheses,
        pruned_hypotheses,
        trajec_actions,
        sim_trajec_actions,
    ) = do_hypothesis_testing(
        args.col_name,
        MIMICtable,
        MIMIC_data_held_back,
        MIMIC_data_used,
        sim_data,
        args.col_bin_num,
        args.hyp_test_dir,
        args.use_kmeans,
        args.reverse_percentile,
        args.heoffdings,
    )

    rej_hyps.to_csv(
        f"{args.hyp_test_dir}/rej_hyps_{args.col_name}_hoeff{args.heoffdings}.csv"
    )
    p_values.to_csv(
        f"{args.hyp_test_dir}/p_values_{args.col_name}_hoeff{args.heoffdings}.csv"
    )
    pruned_hypotheses.to_csv(
        f"{args.hyp_test_dir}/pruned_hyps_{args.col_name}_hoeff{args.heoffdings}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--col_name",
        help="Column name to consider as the outcome of interest for hypothesis tests",
        type=str,
        default="Calcium",
    )
    parser.add_argument(
        "--col_bin_num", help="number of column bins", type=int, default=2
    )
    parser.add_argument(
        "--obs_path",
        help="path to preprocessed MIMIC data",
        default="/data/ziz/taufiq/export-dir/MIMIC-1hourly-processed.csv",
    )
    parser.add_argument(
        "--sim_data_path",
        help="path to twin data",
        default="./twin_data/pulse_data_combined.csv",
    )
    parser.add_argument(
        "--hyp_test_dir",
        help="Directory to save hypothesis test results",
        default="./output_files",
    )
    parser.add_argument(
        "--heoffdings",
        help="Use heoffdings inequality for hypothesis testing. If False, use percentile bootstrap by default",
        default="False",
        type=str2bool,
    )
    parser.add_argument(
        "--preprocess_pulse_data",
        help="Preprocess Pulse data from raw trajectory files as generated by Pulse."
        + "If True, sim_data_path must point to the directory containing raw pulse generated files.",
        default="False",
        type=str2bool,
    )
    parser.add_argument(
        "--use_kmeans",
        help="Use k-means to discretize the state space",
        default="False",
        type=str2bool,
    )
    parser.add_argument(
        "--reverse_percentile",
        help="Use reverse percentile bootstrap",
        type=str2bool,
        default="False",
    )
    args = parser.parse_args()

    main(args)
