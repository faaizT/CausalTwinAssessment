from pulse.cdm.engine import SEDataRequestManager, SEDataRequest

cols = [
    "bloc",
    "gender",
    "age",
    "elixhauser",
    "re_admission",
    "died_in_hosp",
    "died_within_48h_of_out_time",
    "mortality_90d",
    "delay_end_of_record_and_discharge_or_death",
    "SOFA",
    "SIRS",
    "Weight_kg",
    "GCS",
    "HR",
    "SysBP",
    "MeanBP",
    "DiaBP",
    "RR",
    "SpO2",
    "Temp_C",
    "FiO2_1",
    "Potassium",
    "Sodium",
    "Chloride",
    "Glucose",
    "BUN",
    "Creatinine",
    "Magnesium",
    "Calcium",
    "Ionised_Ca",
    "CO2_mEqL",
    "SGOT",
    "SGPT",
    "Total_bili",
    "Albumin",
    "Hb",
    "WBC_count",
    "Platelets_count",
    "PTT",
    "PT",
    "INR",
    "Arterial_pH",
    "paO2",
    "paCO2",
    "Arterial_BE",
    "HCO3",
    "Arterial_lactate",
    "mechvent",
    "Shock_Index",
    "PaO2_FiO2",
    "max_dose_vaso",
    "input_total",
    "output_total",
    "output_1hourly",
    "cumulated_balance",
]

column_mappings = {
    "HR": "HeartRate",
    "SysBP": "SystolicArterialPressure",
    "DiaBP": "DiastolicArterialPressure",
    "RR": "RespirationRate",
    # "Potassium": "Potassium",
    # "Glucose": "Glucose",
    "BUN": "BloodUreaNitrogenConcentration",
    # "Creatinine": "Creatinine",
    # "Calcium": "Calcium",
    # "Albumin": "Albumin",
    # "Hb": "HemoglobinContent", --> how to get blood volume
    "WBC_count": "WhiteBloodCellCount",
    "Arterial_pH": "BloodPH",
    "paO2": "ArterialOxygenPressure",
    "paCO2": "ArterialCarbonDioxidePressure",
    # "HCO3": "Bicarbonate",
    # "Arterial_lactate": "Lactate",
    "PaO2_FiO2": "SaturationAndFractionOfInspiredOxygenRatio"
}

s0_cols = ["HR", "SysBP", "DiaBP", "RR"]
dummy_cols = s0_cols + ["Potassium", "Glucose", "BUN", "Creatinine", "Calcium", "Albumin", "Hb", "WBC_count", "Arterial_pH", "paO2", "paCO2", "HCO3", "Arterial_lactate", "PaO2_FiO2"]
static_cols = ["gender", "age", "Weight_kg"]
action_cols = ["input_1hourly", "median_dose_vaso", "mechvent"]

request_dict = {
    "HeartRate": "1/min",
    "ArterialPressure": "mmHg",
    "MeanArterialPressure": "mmHg",
    "SystolicArterialPressure": "mmHg",
    "DiastolicArterialPressure": "mmHg",
    "OxygenSaturation": None,
    "EndTidalCarbonDioxidePressure": "mmHg",
    "ArterialOxygenPressure": "mmHg",
    "ArterialCarbonDioxidePressure": "mmHg",
    "RespirationRate": "1/min",
    "SkinTemperature": "degC",
    "CardiacOutput": "L/min",
    "BloodVolume": "mL",
    "BloodUreaNitrogenConcentration": "mg/dL",
    "HemoglobinContent": "g",
    "WhiteBloodCellCount": "ct/uL",
    "BloodPH": None,
    "CarbonDioxideSaturation": None,
    "SaturationAndFractionOfInspiredOxygenRatio": None,
    # "Potassium": "mg/L",
    # "Sodium": "mg/L",
    # "Chloride": "mg/L",
    # "Glucose": "mg/L",
    # "Creatinine": "mg/L",
    # "Calcium": "mg/L",
    # "Albumin": "mg/L",
    # "Bicarbonate": "mg/L",
    # "Lactate": "mg/L",
}

substance_requests = ['Potassium', 'Sodium', 'Chloride', 'Glucose', 'Creatinine', 'Calcium', 'Albumin', 'Bicarbonate', 'Lactate']

def get_data_req_mgr():
    data_requests = []
    for request, unit in request_dict.items():
        if request in substance_requests:
            data_requests.append(SEDataRequest.create_substance_request(request, "BloodConcentration", unit))
        else:
            data_requests.append(SEDataRequest.create_physiology_request(request, unit=unit))
    return SEDataRequestManager(data_requests)

