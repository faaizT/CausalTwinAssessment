cols = [
    'Temp_C', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR',
    'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Creatinine', 'Calcium',
    'Albumin', 'WBC_count', 'Arterial_pH', 'HCO3', 'Arterial_lactate'
]

pulse_cols = [
       'Albumin', 'paCO2', 'paO2', 'ArterialPressure (mmHg)', 'HCO3',
       'Arterial_pH', 'BloodUreaNitrogenConcentration (mg/dL)',
       'BloodVolume (mL)', 'Calcium', 'CarbonDioxideSaturation (None)',
       'CardiacOutput (L/min)', 'Chloride', 'Creatinine', 'DiaBP',
       'EndTidalCarbonDioxidePressure (mmHg)', 'Glucose', 'HR',
       'HemoglobinContent (g)', 'Arterial_lactate', 'MeanBP',
       'OxygenSaturation (None)', 'Potassium', 'RR', 'PaO2_FiO2',
       'Temp_C', 'Sodium', 'SysBP', 'WBC_count',
]

biogears_cols = [
       'HR', 'MeanBP', 'SysBP', 'DiaBP', 'CardiacOutput(mL/min)',
       'HemoglobinContent(g)', 'CentralVenousPressure(mmHg)', 'Hematocrit',
       'Arterial_pH', 'WBC_count', 'UrineProductionRate(mL/s)', 'RR',
       'OxygenSaturation', 'CarbonDioxideSaturation', 'CoreTemperature(degC)',
       'Temp_C', 'HCO3', 'Creatinine', 'Arterial_lactate', 'Glucose', 'Sodium',
       'Potassium', 'Chloride', 'Calcium', 'Albumin']

action_cols = [
    'A_t'
]

static_cols = [
    "gender", "age", "Weight_kg"
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


