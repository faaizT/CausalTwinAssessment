import numpy as np
from observational_model.Model import Model
from observational_model.PatientState import PatientState


class RealSimulator(Model):
    def transition_to_next_state(self, action):
        self.current_hr += np.random.normal(action * 8, action ** 2 * 4 + 0.1) * (1 + self.current_socio_econ) / 3
        self.current_rr += np.random.normal(action * 2, action ** 2 * 1 + 0.1) * (1 + self.current_socio_econ) / 3
        self.current_sysbp += np.random.normal(
            action * 5 * (1 - self.current_gender) + self.current_gender * action * 3, action * 5 + 0.1)
        self.current_diabp += np.random.normal(
            action * 5 * (1 - self.current_gender) + self.current_gender * action * 3, action * 5 + 0.1)
        self.current_wbc_count -= np.random.normal(action / 3 * 1000, ((action + 1) / 4) ** 2 * 1000)
        if self.current_pain_stimulus > 0:
            self.current_pain_stimulus -= np.random.binomial(1, 0.5)
        self.current_state = PatientState(gender=self.current_gender, hr=self.current_hr, rr=self.current_rr,
                                          sysbp=self.current_sysbp, diabp=self.current_diabp,
                                          height=self.current_height,
                                          weight=self.current_weight, wbc_count=self.current_wbc_count,
                                          socio_econ=self.current_socio_econ,
                                          pain_stimulus=self.current_pain_stimulus, fio2=self.current_fio2)


def get_simulator(simulator_name):
    if simulator_name == "real":
        return RealSimulator()
    elif simulator_name == "original":
        return Model()
    else:
        raise Exception("Simulator name incorrect. Must be one of [real, original]")
