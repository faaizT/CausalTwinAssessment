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
    "RR": "RespirationRate"
}

s0_cols = ["HR", "SysBP", "DiaBP", "RR", "height", "blood_volume"]
dummy_cols = ["HR", "SysBP", "DiaBP", "RR"]
static_cols = ["gender", "age", "Weight_kg"]
action_cols = ["input_1hourly", "median_dose_vaso"]

request_dict = {
    "HeartRate": "1/min",
    "ArterialPressure": "mmHg",
    "MeanArterialPressure": "mmHg",
    "SystolicArterialPressure": "mmHg",
    "DiastolicArterialPressure": "mmHg",
    "OxygenSaturation": None,
    "EndTidalCarbonDioxidePressure": "mmHg",
    "RespirationRate": "1/min",
    "SkinTemperature": "degC",
    "CardiacOutput": "L/min",
    "BloodVolume": "mL"
}

def get_data_req_mgr():
    data_requests = []
    for request, unit in request_dict.items():
        data_requests.append(SEDataRequest.create_physiology_request(request, unit=unit))
    return SEDataRequestManager(data_requests)

