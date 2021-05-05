from pulse.cdm.patient import SEPatient, eSex
from pulse.cdm.scalars import TimeUnit, LengthUnit, MassUnit, FrequencyUnit, PressureUnit, VolumeUnit
import pandas as pd
import json


def reset_extreme_readings(pt: SEPatient):
    if pt.has_age():
        age = pt.get_age().get_value(TimeUnit.day)/365
        ageMin, ageMax = 18, 65
        if age < ageMin:
            pt.get_age().set_value(ageMin*365, TimeUnit.day)
        elif age > ageMax:
            pt.get_age().set_value(ageMax*365, TimeUnit.day)
    if pt.has_height():
        heightMinMale_cm = 163.0
        heightMaxMale_cm = 190.0
        heightMinFemale_cm = 151.0
        heightMaxFemale_cm = 175.5

        if pt.get_sex() == eSex.Male:
            heightMin, heightMax = heightMinMale_cm, heightMaxMale_cm
        else:
            heightMin, heightMax = heightMinFemale_cm, heightMaxFemale_cm

        height = pt.get_height().get_value(LengthUnit.cm)
        if height < heightMin:
            pt.get_height().set_value(heightMin, LengthUnit.cm)
        elif height > heightMax:
            pt.get_height().set_value(heightMax, LengthUnit.cm)
    else:
        heightStandardMale_cm, heightStandardFemale_cm = 177.0, 163.0
        if pt.get_sex() == eSex.Male:
            pt.get_height().set_value(heightStandardMale_cm, LengthUnit.cm)
        else:
            pt.get_height().set_value(heightStandardFemale_cm, LengthUnit.cm)

    if pt.has_weight():
        BMIObese_kg_per_m2 = 30.0
        BMISeverelyUnderweight_kg_per_m2 = 16.0

        weight = pt.get_weight().get_value(MassUnit.kg)
        BMI = weight/((pt.get_height().get_value(LengthUnit.cm)/100)**2)
        if BMI > BMIObese_kg_per_m2:
            pt.get_weight().set_value(BMIObese_kg_per_m2*(pt.get_height().get_value(LengthUnit.cm)/100)**2, MassUnit.kg)
        elif BMI < BMISeverelyUnderweight_kg_per_m2:
            pt.get_weight().set_value(BMISeverelyUnderweight_kg_per_m2*(pt.get_height().get_value(LengthUnit.cm)/100)**2, MassUnit.kg)

    if pt.has_body_fat_fraction():
        fatFractionMaxMale = 0.25
        fatFractionMaxFemale = 0.32
        fatFractionMinMale = 0.02
        fatFractionMinFemale = 0.10

        if pt.get_sex() == eSex.Male:
            fatFractionMax, fatFractionMin = fatFractionMaxMale, fatFractionMinMale
        else:
            fatFractionMax, fatFractionMin = fatFractionMaxFemale, fatFractionMinFemale

        fatFraction = pt.get_body_fat_fraction().get_value()
        if fatFraction > fatFractionMax:
            pt.get_body_fat_fraction().set(fatFractionMax)
        elif fatFraction < fatFractionMin:
            pt.get_body_fat_fraction().set(fatFractionMin)

    if pt.has_heart_rate_baseline():
        heart_rate = pt.get_heart_rate_baseline().get_value(FrequencyUnit.Per_min)
        heartRateTachycardia_bpm = 110
        heartRateBradycardia_bpm = 50
        if heart_rate < heartRateBradycardia_bpm:
            pt.get_heart_rate_baseline().set_value(heartRateBradycardia_bpm, FrequencyUnit.Per_min)
        elif heart_rate > heartRateTachycardia_bpm:
            pt.get_heart_rate_baseline().set_value(heartRateTachycardia_bpm, FrequencyUnit.Per_min)

    if pt.has_systolic_arterial_pressure_baseline():
        systolicMax_mmHg = 120.0
        systolicMin_mmHg = 90.0
        systolic = pt.get_systolic_arterial_pressure_baseline().get_value(PressureUnit.mmHg)
        if systolic > systolicMax_mmHg:
            pt.get_systolic_arterial_pressure_baseline().set_value(systolicMax_mmHg, PressureUnit.mmHg)
        elif systolic < systolicMin_mmHg:
            pt.get_systolic_arterial_pressure_baseline().set_value(systolicMin_mmHg, PressureUnit.mmHg)
    else:
        systolicStandard_mmHg = 114.0
        pt.get_systolic_arterial_pressure_baseline().set_value(systolicStandard_mmHg, PressureUnit.mmHg)

    if pt.has_diastolic_arterial_pressure_baseline():
        diastolicMax_mmHg = 80.0
        diastolicMin_mmHg = 60.0
        diastolic = pt.get_diastolic_arterial_pressure_baseline().get_value(PressureUnit.mmHg)
        diastolicMax_allowed = min(diastolicMax_mmHg, 0.75*pt.get_systolic_arterial_pressure_baseline().get_value(PressureUnit.mmHg))
        if diastolic > diastolicMax_allowed:
            pt.get_diastolic_arterial_pressure_baseline().set_value(diastolicMax_allowed, PressureUnit.mmHg)
        elif diastolic < diastolicMin_mmHg:
            pt.get_diastolic_arterial_pressure_baseline().set_value(diastolicMin_mmHg, PressureUnit.mmHg)
    else:
        diastolicStandard_mmHg = 73.5
        diastolicMax_allowed = min(diastolicStandard_mmHg, 0.75*pt.get_systolic_arterial_pressure_baseline().get_value(PressureUnit.mmHg))
        pt.get_diastolic_arterial_pressure_baseline().set_value(diastolicMax_allowed, PressureUnit.mmHg)

    if pt.has_blood_volume_baseline():
        computedBloodVolume_mL = 65.6*pt.get_weight().get_value(MassUnit.kg)**2
        bloodVolumeMin_mL = computedBloodVolume_mL * 0.85
        bloodVolumeMax_mL = computedBloodVolume_mL * 1.15
        bloodVolume = pt.get_blood_volume_baseline().get_value(VolumeUnit.mL)
        if bloodVolume > bloodVolumeMax_mL:
            pt.get_blood_volume_baseline().set_value(bloodVolumeMax_mL, VolumeUnit.mL)
        elif bloodVolume < bloodVolumeMin_mL:
            pt.get_blood_volume_baseline().set_value(bloodVolumeMin_mL, VolumeUnit.mL)

    if pt.has_respiration_rate_baseline():
        respirationRateMax_bpm = 20.0
        respirationRateMin_bpm = 8.0
        respirationRate = pt.get_respiration_rate_baseline().get_value(FrequencyUnit.Per_min)
        if respirationRate > respirationRateMax_bpm:
            pt.get_respiration_rate_baseline().set_value(respirationRateMax_bpm, FrequencyUnit.Per_min)
        elif respirationRate < respirationRateMin_bpm:
            pt.get_respiration_rate_baseline().set_value(respirationRateMin_bpm, FrequencyUnit.Per_min)

    if pt.has_right_lung_ratio():
        rightLungRatioMax = 0.60
        rightLungRatioMin = 0.50
        rightLungRatio = pt.get_right_lung_ratio().get_value()
        if rightLungRatio > rightLungRatioMax:
            pt.get_right_lung_ratio().set_value(rightLungRatioMax)
        elif rightLungRatio < rightLungRatioMin:
            pt.get_right_lung_ratio().set(rightLungRatioMin)
