import numpy as np


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
        facial_expression = self.pain_stimulus + np.random.randint(-1,2)
        return Ut(self.socio_econ, max(min(facial_expression, 9), 0))


class Xt:
    def __init__(self, gender, hr, sysbp, diabp):
        self.gender = gender
        self.hr = hr
        self.sysbp = sysbp
        self.diabp = diabp


class Ut:
    def __init__(self, socio_econ, facial_expression):
        self.socio_econ = socio_econ
        self.facial_expression = facial_expression

