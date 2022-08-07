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


def write_to_file(file_name, col_name, num_rejected, total_hypotheses, sofa_bin):
    with open(file_name, "a", 1) as f:
        f.write(
            col_name
            + ","
            + str(num_rejected)
            + ","
            + str(total_hypotheses)
            + ","
            + str(sofa_bin)
            + os.linesep
        )


def load_pulse_data(sim_path, MIMICtable, load_generated):
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
    extension = "final_.csv"
    all_filenames = [i for i in glob.glob(f"{args.sim_path}/*{extension}")]
    if load_generated:
        all_filenames += [
            i for i in glob.glob(f"{args.sim_path}/synthetic_data/*{extension}")
        ]

    logging.info(f"Total number of simulated trajectories: {len(all_filenames)}")
    sim_data = pd.concat([pd.read_csv(f) for f in all_filenames])
    sim_data = sim_data.rename(columns={"id": "icustay_id"})
    sim_data["icustay_id"] = sim_data["icustay_id"].astype(int)
    sim_data = sim_data.reset_index(drop=True)
    sim_data = sim_data.sort_values(
        by=["icustay_id", "SimulationTime(s)"], ignore_index=True
    )
    sim_data["bloc"] = np.arange(len(sim_data)) % 5 + 1

    sim_rename = {}
    for k, v in column_mappings.items():
        sim_rename.update({k: f"{v}"})

    sim_data = sim_data.rename(columns=sim_rename)
    sim_data = sim_data.merge(
        MIMICtable[["gender", "age", "Weight_kg", "icustay_id", "bloc"]],
        left_on=["icustay_id", "bloc"],
        right_on=["icustay_id", "bloc"],
    )
    return sim_data


def load_biogears_data(sim_path, MIMICtable):
    column_mappings = {
        "Albumin-BloodConcentration(ug/mL)": "Albumin",
        "ArterialCarbonDioxidePressure(mmHg)": "paCO2",
        "ArterialOxygenPressure(mmHg)": "paO2",
        "Bicarbonate-BloodConcentration(ug/mL)": "HCO3",
        "ArterialBloodPH": "Arterial_pH",
        "Calcium-BloodConcentration(ug/mL)": "Calcium",
        "Chloride-BloodConcentration(ug/mL)": "Chloride",
        "Creatinine-BloodConcentration(ug/mL)": "Creatinine",
        "DiastolicArterialPressure(mmHg)": "DiaBP",
        "Glucose-BloodConcentration(ug/mL)": "Glucose",
        "Lactate-BloodConcentration(ug/mL)": "Arterial_lactate",
        "MeanArterialPressure(mmHg)": "MeanBP",
        "Potassium-BloodConcentration(ug/mL)": "Potassium",
        "RespirationRate(1/min)": "RR",
        "SkinTemperature(degC)": "Temp_C",
        "Sodium-BloodConcentration(ug/mL)": "Sodium",
        "SystolicArterialPressure(mmHg)": "SysBP",
        "WhiteBloodCellCount(ct/uL)": "WBC_count",
        "HeartRate(1/min)": "HR",
    }
    extension = ".csv"
    all_filenames = [i for i in glob.glob(f"{args.sim_path}/*{extension}")]
    biogears_data = pd.DataFrame()
    for f in all_filenames:
        if os.path.getsize(f) > 0:
            df = pd.read_csv(f)
            m = re.search("SimulateMIMIC_(.+?)_.csv", f)
            if m:
                icustay_id = m.group(1)
                df["icustay_id"] = int(icustay_id)
            biogears_data = biogears_data.append(df, ignore_index=True)

    times = [600.02, 3600.02, 7200.02, 10800.02, 14400.02]
    biogears_data = biogears_data[biogears_data["Time(s)"].isin(times)].reset_index(
        drop=True
    )
    biogears_data.loc["icustay_id"] = biogears_data["icustay_id"].astype(int)
    icustayids = []
    for icustay_id in biogears_data["icustay_id"].unique():
        if (biogears_data["icustay_id"] == icustay_id).sum() == 5:
            icustayids.append(icustay_id)
    biogears_data = biogears_data[
        biogears_data["icustay_id"].isin(icustayids)
    ].reset_index(drop=True)
    biogears_data = biogears_data.sort_values(
        by=["icustay_id", "Time(s)"], ignore_index=True
    )
    biogears_data["bloc"] = np.arange(len(biogears_data)) % 5 + 1
    biogears_rename = {}
    for k, v in column_mappings.items():
        biogears_rename.update({k: f"{v}"})
    biogears_data = biogears_data.rename(columns=biogears_rename)
    biogears_data = biogears_data.merge(
        MIMICtable[["gender", "age", "Weight_kg", "icustay_id", "bloc"]],
        left_on=["icustay_id", "bloc"],
        right_on=["icustay_id", "bloc"],
    )
    return biogears_data


def create_action_bins(MIMICtable):
    logging.info("Creating action bins")
    nact = nra ** 2
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


def load_mimic_data(obs_path, load_generated):
    MIMICtable = pd.read_csv(args.obs_path + "/MIMIC-1hourly-length-5-combined.csv")
    if load_generated:
        logging.info("Using VAE generated observational data for hypothesis testing")
        MIMIC_generated_males = pd.read_csv(
            args.obs_path + "/MIMIC-generated-length-5-gender-0.0.csv"
        )
        MIMIC_generated_females = pd.read_csv(
            args.obs_path + "/MIMIC-generated-length-5-gender-1.0.csv"
        )
        MIMICtable = pd.concat(
            [MIMICtable, MIMIC_generated_males, MIMIC_generated_females],
            ignore_index=True,
        )
    MIMICtable = MIMICtable.sort_values(by=["icustay_id", "bloc"], ignore_index=True)
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


def main(args):
    if args.saved_dir is not None:
        logging.info(f"Using saved processed data from dir {args.saved_dir}")
    else:
        logging.info("Not using saved data")

    MIMICtable = load_mimic_data(args.obs_path, args.load_generated)
    if not args.load_generated:
        MIMICtable_generated = load_mimic_data(args.obs_path, load_generated=True)
    else:
        MIMICtable_generated = MIMICtable.copy()
    if args.sim_name == "pulse":
        sim_data = load_pulse_data(
            args.sim_path, MIMICtable_generated, load_generated=True
        )
    else:
        sim_data = load_biogears_data(args.sim_path, MIMICtable)

    logging.info(f"Outcome: {args.col_name}")
    if args.saved_dir is not None:
        (
            num_rej_hyps,
            p_values,
            rej_hyps,
            total_hypotheses,
            pruned_hypotheses,
            trajec_actions,
            sim_trajec_actions,
        ) = do_hypothesis_testing_saved(
            args.col_name,
            args.saved_dir,
            sim_data,
            MIMICtable,
            args.sofa_bin,
            args.use_kmeans,
            args.reverse_percentile,
            args.heoffdings,
            args.pruning,
            args.discretised_outcome,
        )
    else:
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
            sim_data,
            args.col_bin_num,
            args.hyp_test_dir,
            args.use_kmeans,
            args.reverse_percentile,
            args.heoffdings,
            args.pruning,
            args.discretised_outcome,
        )

    write_to_file(
        f"{args.hyp_test_dir}/rej_hyp_nums_hoeff{args.heoffdings}.csv",
        args.col_name,
        num_rej_hyps,
        total_hypotheses,
        args.sofa_bin,
    )
    if args.sofa_bin is None:
        rej_hyps.to_csv(f"{args.hyp_test_dir}/rej_hyps_{args.col_name}_hoeff{args.heoffdings}.csv")
        p_values.to_csv(f"{args.hyp_test_dir}/p_values_{args.col_name}_hoeff{args.heoffdings}.csv")
    else:
        rej_hyps.to_csv(
            f"{args.hyp_test_dir}/rej_hyps_sofabin_{args.sofa_bin}_{args.col_name}_hoeff{args.heoffdings}.csv"
        )
        p_values.to_csv(
            f"{args.hyp_test_dir}/p_values_sofabin_{args.sofa_bin}_{args.col_name}_hoeff{args.heoffdings}.csv"
        )
    pruned_hypotheses.to_csv(f"{args.hyp_test_dir}/pruned_hyps_{args.col_name}_hoeff{args.heoffdings}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--col_name",
        help="Column name to run hypothesis tests for",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sim_name", help="Simulator name (pulse/biogears)", type=str, default="pulse"
    )
    parser.add_argument(
        "--col_bin_num", help="number of column bins", type=int, default=5
    )
    parser.add_argument(
        "--obs_path",
        help="path to observational data directory",
        default="/data/ziz/taufiq/export-dir",
    )
    parser.add_argument(
        "--sim_path",
        help="path to sim data directory",
        default="/data/ziz/taufiq/pulse-data-5-step",
    )
    parser.add_argument(
        "--hyp_test_dir",
        help="Directory to save hypothesis test info",
        default="/data/localhost/not-backed-up/taufiq/HypothesisTesting/dry-run",
    )
    parser.add_argument(
        "--saved_dir", 
        help="Location of saved processed data", 
        default=None,
    )
    parser.add_argument(
        "--sofa_bin",
        help="Splits data into SOFA bins before running hypothesis tests",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--load_generated",
        help="Use VAE generated observational data for hypothesis testing",
        default="False",
        type=str2bool,
    )
    parser.add_argument(
        "--heoffdings",
        help="Use heoffdings inequality for hypothesis testing",
        default="False",
        type=str2bool,
    )
    parser.add_argument(
        "--use_kmeans",
        help="Use k-means to discretize the state space",
        default="True",
        type=str2bool,
    )
    parser.add_argument(
        "--reverse_percentile",
        help="Use reverse percentile bootstrap",
        type=str2bool,
        default="False",
    )
    parser.add_argument(
        "--pruning",
        help="Use pruning with Hoeffding",
        type=str2bool,
        default="False",
    )
    parser.add_argument(
        "--discretised_outcome",
        help="Run tests on discretised outcomes",
        type=str2bool,
        default="False",
    )
    args = parser.parse_args()

    if not os.path.exists(f"{args.hyp_test_dir}/rej_hyp_nums.csv"):
        with open(f"{args.hyp_test_dir}/rej_hyp_nums.csv", "w") as f:
            f.write(
                "Outcome Y,# rejected hypotheses,Total # hypotheses,SOFA bin"
                + os.linesep
            )
    main(args)