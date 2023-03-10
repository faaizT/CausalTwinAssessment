import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import logging
import pandas as pd
import numpy as np
import glob
import re
from sklearn.cluster import KMeans
from scipy.stats import rankdata
from pulse.Utils import *
from utils import str2bool
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl


column_names_dict = {
    "Albumin": "Albumin Blood Concentration",
    "paCO2": "Arterial $CO_2$ Pressure",
    "paO2": "Arterial $O_2$ Pressure",
    "HCO3": "Bicarbonate Blood Concentration",
    "Arterial_pH": "Arterial pH",
    "Arterial_lactate": "Lactate Blood Concentration",
    "Calcium": "Calcium Blood Concentration",
    "Chloride": "Chloride Blood Concentration",
    "Creatinine": "Creatinine Blood Concentration",
    "DiaBP": "Diastolic Arterial Pressure",
    "SysBP": "Systolic Arterial Pressure",
    "Glucose": "Glucose Blood Concentration",
    "MeanBP": "Mean Arterial Pressure",
    "Potassium": "Potassium Blood Concentration",
    "RR": "Respiration Rate",
    "Temp_C": "Skin Temperature",
    "Sodium": "Sodium Blood Concentration",
    "WBC_count": "White Blood Cell Count",
    "HR": "Heart Rate",
}


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
    "MeanBP": "Mean Arterial Pressure (mmHg)",
    "Potassium": "Potassium Blood Concentration (mg/L)",
    "RR": "Respiration Rate (1/min)",
    "Temp_C": "Skin Temperature (C)",
    "Sodium": "Sodium Blood Concentration (mg/L)",
    "WBC_count": "White Blood Cell Count (ct/uL)",
    "HR": "Heart Rate (1/min)",
}


def get_hoeffding_bounds(index, row, alpha=0.05):
    ylo, yup = row["y_lo"], row["y_up"]
    nobs, nsim = row["n_obs"], row["n_sim"]
    delta_obs = (yup - ylo) * np.sqrt(1 / (2 * nobs) * np.log(4 / alpha))
    delta_sim = (yup - ylo) * np.sqrt(1 / (2 * nsim) * np.log(4 / alpha))
    ylb_lb = np.mean(row["Y_lb_mean"]) - delta_obs
    ylb_ub = np.mean(row["Y_lb_mean"]) + delta_obs
    yub_lb = np.mean(row["Y_ub_mean"]) - delta_obs
    yub_ub = np.mean(row["Y_ub_mean"]) + delta_obs
    ysim_lb = np.mean(np.clip(row["ysim_values"], ylo, yup)) - delta_sim
    ysim_ub = np.mean(np.clip(row["ysim_values"], ylo, yup)) + delta_sim
    return [ylb_lb, ylb_ub], [ysim_lb, ysim_ub], [yub_lb, yub_ub]


def add_interval(
    ax,
    xdata,
    ydata,
    color,
    left,
    caps="  ",
):
    line = ax.add_line(
        mpl.lines.Line2D(xdata, ydata, color=color, linewidth=4, alpha=0.4),
    )
    anno_args1 = {"ha": "left", "va": "center", "size": 16, "color": color}
    anno_args2 = {"ha": "right", "va": "center", "size": 16, "color": color}
    if left:
        a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), alpha=0.8, **anno_args2)
    else:
        a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), alpha=0.8, **anno_args1)


def generate_hoeff_intervals_one_sided(
    index, row, results_directory, total_hypotheses, lo=True
):
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [4, 1]})
    plt.style.use("ggplot")
    min_x, max_x = row["y_lo"], row["y_up"]

    axs[0].hist(
        np.array(row["yobs_values"]),
        label="Obs. data",
        density=True,
        alpha=0.4,
        bins="auto",
        color="blue",
    )
    axs[0].hist(
        row["ysim_values"],
        label="Twin data",
        density=True,
        alpha=0.4,
        bins="auto",
        color="red",
    )
    p_lb, p_ub = row["p_lb"], row["p_ub"]
    rejected = (row["rejected_holms_lb"]) or (row["rejected_holms_ub"])
    figtitle = f"{row['col']}_hyp_{index}"

    ylb_interval, ysim_interval, yub_interval = get_hoeffding_bounds(
        index, row, alpha=0.05 / total_hypotheses
    )
    ylb_interval = np.clip(
        ylb_interval,
        row["y_lo"],
        row["y_up"],
    )
    ysim_interval = np.clip(
        ysim_interval,
        row["y_lo"],
        row["y_up"],
    )
    yub_interval = np.clip(
        yub_interval,
        row["y_lo"],
        row["y_up"],
    )

    if lo:
        rejected = ysim_interval[1] < ylb_interval[0]
        add_interval(
            axs[1],
            [ylb_interval[0], max_x],
            [0, 0],
            caps="||",
            color="blue",
            left=True,
        )
        add_interval(
            axs[1],
            [min_x, ysim_interval[1]],
            [0.5, 0.5],
            caps="||",
            color="red",
            left=False,
        )
        axs[1].plot([ylb_interval[0], max_x], [0, 0], lw=6, color="blue", alpha=0.0)
        axs[1].plot([min_x, ysim_interval[1]], [0.5, 0.5], lw=6, color="red", alpha=0.0)
        axs[1].set_yticks([0, 0.5])
        axs[1].set_yticklabels(["$Q_{lo}$", "$\widehat{Q}$"])
    else:
        rejected = ysim_interval[0] > yub_interval[1]
        add_interval(
            axs[1],
            [ysim_interval[0], max_x],
            [0.5, 0.5],
            caps="||",
            color="red",
            left=True,
        )
        add_interval(
            axs[1],
            [min_x, yub_interval[1]],
            [0, 0],
            caps="||",
            color="blue",
            left=False,
        )
        axs[1].plot(
            [ysim_interval[0], max_x], [0.5, 0.5], lw=6, color="blue", alpha=0.0
        )
        axs[1].plot([min_x, ylb_interval[1]], [0.0, 0.0], lw=6, color="red", alpha=0.0)
        axs[1].set_yticks([0, 0.5])
        axs[1].set_yticklabels(["$Q_{up}$", "$\widehat{Q}$"])

    axs[0].set_title(column_names_unit[row["col"]], fontsize=20)
    axs[0].set_yticks([])
    axs[0].grid(False)
    axs[1].grid(False)

    min_ylim = axs[1].get_ylim()[0]
    max_ylim = axs[1].get_ylim()[1]
    axs[1].set_ylim([-0.2, 0.7])

    axs[0].legend(
        fontsize=16,
    )

    axs[1].set_xlabel("Hoeffding's Intervals", fontsize=20)
    axs[0].tick_params(axis="both", which="major", labelsize=20)
    axs[0].tick_params(axis="both", which="minor", labelsize=20)
    axs[1].tick_params(axis="both", which="major", labelsize=20)
    axs[1].tick_params(axis="both", which="minor", labelsize=20)

    plt.tight_layout()

    plt.savefig(
        f"{results_directory}/histograms/{row['col']}_rej{rejected}/{figtitle}_with_hoeff_onesided_lo{lo}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def generate_longitudinal_plots_of_average_values(
    index,
    row,
    p_values,
    results_directory,
    total_hypotheses,
    caption=False,
    title=None,
    fig=None,
    axis=None,
):
    a = row["actions"]
    age = row["age"]
    gender = row["gender"]
    x_t = row["x_t"]

    y_twin = []
    y_twin_lb = []
    y_twin_ub = []
    y_obs = []
    y_obs_lb = []
    y_obs_ub = []
    y_obs_std = []
    p_lb_vals = []
    p_ub_vals = []
    y_lo = []
    y_up = []
    rejected = False
    save_fig = fig is None
    if save_fig:
        fig, axis = plt.subplots(1, 1, figsize=(7, 5))
    plt.style.use("fivethirtyeight")

    for t in range(1, 5):
        p_values_filtered = p_values[
            (p_values["gender"] == gender)
            & (p_values["age"] == age)
            & (p_values["actions"] == a[:t])
            & (p_values["x_t"] == x_t[:t])
            & (p_values["t"] == t)
        ]
        ylb_interval, ysim_interval, yub_interval = get_hoeffding_bounds(
            index, row=p_values_filtered.iloc[0], alpha=0.05 / total_hypotheses
        )
        p_lb_vals.append(p_values_filtered["p_lb"].values[0])
        p_ub_vals.append(p_values_filtered["p_ub"].values[0])
        y_twin.append(np.mean(p_values_filtered["Sim_exp_y"].values[0]))
        y_twin_lb.append(ysim_interval[0])
        y_twin_ub.append(ysim_interval[1])
        y_obs.append(np.mean(p_values_filtered["yobs_values"].values[0]))
        std_error_obs = (
            2
            * np.std(p_values_filtered["yobs_values"].values[0])
            / np.sqrt(len(p_values_filtered["yobs_values"].values[0]))
        )
        y_obs_lb.append(
            np.mean(p_values_filtered["yobs_values"].values[0]) - std_error_obs
        )
        y_obs_ub.append(
            np.mean(p_values_filtered["yobs_values"].values[0]) + std_error_obs
        )
        y_lo.append(ylb_interval[0])
        y_up.append(yub_interval[1])
        if not rejected:
            rejected = (
                ylb_interval[0] > ysim_interval[1] or yub_interval[1] < ysim_interval[0]
            )

    x = [1, 2, 3, 4]
    axis.plot(x, y_twin, color="red", label="$\widehat{Q}_t$", alpha=0.4)
    axis.fill_between(x, y_twin_lb, y_twin_ub, color="red", alpha=0.2)
    axis.plot(x, y_obs, color="blue", label="$Q_t^{obs}$", alpha=0.4)
    axis.plot(x, y_lo, "--", color="black", label="$Q_{lo} & Q_{up}$", alpha=0.4)
    axis.plot(x, y_up, "--", color="black", alpha=0.4)
    axis.fill_between(x, y_obs_lb, y_obs_ub, color="blue", alpha=0.2)
    axis.set_xlabel("Time (hr)", fontsize=15)
    axis.set_title(title, fontsize=15)

    if caption:
        axis.set_ylabel(column_names_unit[row["col"]], fontsize=15)
    p_lb, p_ub = row["p_lb"], row["p_ub"]
    min_p_value = np.min((np.min(p_lb_vals), np.min(p_ub_vals)))

    axis.set_xticks(range(1, 5))
    axis.tick_params(axis="both", which="major", labelsize=13)
    axis.tick_params(axis="both", which="minor", labelsize=13)
    axis.tick_params(axis="both", which="major", labelsize=13)
    axis.tick_params(axis="both", which="minor", labelsize=13)

    plt.tight_layout()

    if save_fig:
        figtitle = f"{row['col']}_hyp_{index}_caption_{caption}"
        plt.savefig(
            f"{results_directory}/longitudinal/{row['col']}_rej{rejected}/{figtitle}.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close()


def generate_longitudinal_plots_combined(
    first_ind, second_ind, p_values, results_directory, total_hypotheses, col_name
):
    fig, axis = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    plt.style.use("fivethirtyeight")
    generate_longitudinal_plots_of_average_values(
        first_ind,
        p_values.iloc[first_ind],
        p_values,
        results_directory,
        total_hypotheses=total_hypotheses,
        caption=True,
        fig=fig,
        axis=axis[0],
        title=None,
    )
    generate_longitudinal_plots_of_average_values(
        second_ind,
        p_values.iloc[second_ind],
        p_values,
        results_directory,
        total_hypotheses=total_hypotheses,
        caption=False,
        fig=fig,
        axis=axis[1],
        title=None,
    )
    figtitle = f"{col_name}_hyp_{first_ind}v{second_ind}_longitudinal"
    plt.legend(
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        f"{results_directory}/longitudinal/{figtitle}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def main(args):
    p_vals = pd.DataFrame()
    for col in column_names_dict:
        if os.path.exists(
            f"{args.results_dir}/p_values_{args.col_name}_hoeffFalse.csv"
        ):
            p_values = pd.read_csv(
                f"{args.results_dir}/p_values_{args.col_name}_hoeffFalse.csv",
                converters={
                    "actions": eval,
                    "x_t": eval,
                    "Y_lb_mean": eval,
                    "Y_ub_mean": eval,
                    "Sim_exp_y": eval,
                    "yobs_values": eval,
                    "ysim_values": eval,
                },
            )
            if len(p_values) > 0:
                p_values = p_values.loc[p_values["t"] > 0]
                p_values["col"] = col
                p_values["p_lb"] = p_values["p_lb"].clip(0, 1)
                p_values["p_ub"] = p_values["p_ub"].clip(0, 1)
                p_vals = pd.concat([p_vals, p_values], axis=0)

    p_vals["rejected_bonf_lb"] = p_vals["p_lb"] < 0.05 / 4 / len(p_vals)
    p_vals["rejected_bonf_ub"] = p_vals["p_ub"] < 0.05 / 4 / len(p_vals)
    p_vals["rejected_holms_lb"] = multipletests(
        p_vals["p_lb"], alpha=0.05 / 4, method="holm"
    )[0]
    p_vals["rejected_holms_ub"] = multipletests(
        p_vals["p_ub"], alpha=0.05 / 4, method="holm"
    )[0]

    data = {
        "columns": [],
        "total_hypotheses": [],
        "rejected_hypotheses": [],
        "percentage_of_rejected_hyp": [],
    }
    p_values = p_vals[p_vals["col"] == args.col_name]
    p_values["Exp_y"] = p_values["Exp_y"].apply(
        lambda val: eval(val) if "nan" not in val else []
    )
    p_values = p_values[p_values["Exp_y"].apply(lambda x: len(x)) > 0]
    p_values["col"] = args.col_name
    os.makedirs(
        f"{args.image_export_dir}/histograms/{args.col_name}_rejTrue", exist_ok=True
    )
    os.makedirs(
        f"{args.image_export_dir}/histograms/{args.col_name}_rejFalse", exist_ok=True
    )
    os.makedirs(
        f"{args.image_export_dir}/longitudinal/{args.col_name}_rejTrue", exist_ok=True
    )
    os.makedirs(
        f"{args.image_export_dir}/longitudinal/{args.col_name}_rejFalse", exist_ok=True
    )
    for index, row in p_values.iterrows():
        generate_hoeff_intervals_one_sided(
            index, row, args.image_export_dir, total_hypotheses=len(p_vals), lo=True
        )
        generate_hoeff_intervals_one_sided(
            index, row, args.image_export_dir, total_hypotheses=len(p_vals), lo=False
        )

    p_values_complete_trajecs = p_values[(p_values["t"] == 4)]
    for index, row in p_values_complete_trajecs.iterrows():
        generate_longitudinal_plots_of_average_values(
            index,
            row,
            p_values,
            args.image_export_dir,
            caption=True,
            total_hypotheses=len(p_vals),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--col_name",
        help="Column name to generate plots for",
        type=str,
        default="Glucose",
    )
    parser.add_argument(
        "--results_dir",
        help="Location of saved results data",
        default="/data/ziz/not-backed-up/taufiq/HypothesisTesting/hyp_testing_new_pulse_data_2/revperc",
    )
    parser.add_argument(
        "--image_export_dir",
        help="Location to save images",
        default="/data/ziz/not-backed-up/taufiq/HypothesisTesting/hyp_testing_new_pulse_data_2/images_truncated2",
    )
    args = parser.parse_args()
    main(args)
