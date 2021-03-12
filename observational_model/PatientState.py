import numpy as np
import pandas as pd
import torch


class PatientState:
    def __init__(self, gender, hr, rr, sysbp, diabp, fio2, weight, height, wbc_count, socio_econ, pain_stimulus):
        self.gender = gender
        self.hr = hr
        self.rr = rr
        self.sysbp = sysbp
        self.diabp = diabp
        self.fio2 = fio2
        self.weight = weight
        self.height = height
        self.wbc_count = wbc_count
        self.socio_econ = socio_econ
        self.pain_stimulus = pain_stimulus

    def get_xt(self):
        return Xt(self.gender, self.hr + np.random.normal(0, 5),
                  self.sysbp + np.random.normal(0, 4), self.diabp + np.random.normal(0, 4))

    def get_ut(self):
        facial_expression = self.pain_stimulus + np.random.randint(-1, 2)
        return Ut(self.socio_econ, max(min(facial_expression, 9), 0))

    def as_dict(self):
        return {
            'st_gender': self.gender,
            'st_hr': self.hr,
            'st_sysbp': self.sysbp,
            'st_diabp': self.diabp,
            'st_fio2': self.fio2,
            'st_weight': self.weight,
            'st_height': self.height,
            'st_wbc_count': self.wbc_count,
            'st_socio_econ': self.socio_econ,
            'st_pain_stimulus': self.pain_stimulus
        }

    def as_tensor(self):
        tensor = torch.column_stack((self.gender,
                                     self.hr,
                                     self.rr,
                                     self.sysbp,
                                     self.diabp,
                                     self.fio2,
                                     self.weight,
                                     self.height,
                                     self.wbc_count,
                                     self.socio_econ,
                                     self.pain_stimulus)
                                    )
        return tensor.reshape(1, tensor.size(0), tensor.size(1))

    def copy(self):
        return PatientState(
            self.gender,
            self.hr,
            self.rr,
            self.sysbp,
            self.diabp,
            self.fio2,
            self.weight,
            self.height,
            self.wbc_count,
            self.socio_econ,
            self.pain_stimulus
        )


class Xt:
    def __init__(self, gender, hr, sysbp, diabp):
        self.gender = gender
        self.hr = hr
        self.sysbp = sysbp
        self.diabp = diabp

    @classmethod
    def from_series(cls, row: pd.Series):
        gender = int(row['xt_gender'])
        hr = float(row['xt_hr'])
        sysbp = float(row['xt_sysbp'])
        diabp = float(row['xt_diabp'])
        return cls(gender, hr, sysbp, diabp)

    def distance(self, xt):
        return np.abs(self.gender - xt.gender) * 100 + \
               np.abs(self.hr - xt.hr) + \
               np.abs(self.sysbp - xt.sysbp) + \
               np.abs(self.diabp - xt.diabp)

    def as_dict(self):
        return {
            'xt_gender': self.gender,
            'xt_hr': self.hr,
            'xt_sysbp': self.sysbp,
            'xt_diabp': self.diabp
        }

    def copy(self):
        return Xt(
            self.gender,
            self.hr,
            self.sysbp,
            self.diabp
        )


class Ut:
    def __init__(self, socio_econ, facial_expression):
        self.socio_econ = socio_econ
        self.facial_expression = facial_expression

    def as_dict(self):
        return {
            'ut_socio_econ': self.socio_econ,
            'ut_facial_expression': self.facial_expression
        }

    def copy(self):
        return Ut(
            self.socio_econ,
            self.facial_expression
        )
