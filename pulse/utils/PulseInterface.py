import numpy as np

class PulsePhysiologyEnginePool:
    def __init__(self):
        self.patients = []
    
    def create_patient(self, idx):
        self.patients.append(Patient(idx))
        return self.patients[idx]
    
    def get_patients(self):
        return self.patients

    def advance_time_s(self, time):
        for patient in self.patients:
            patient.advance_time(time)
    
class Patient:
    def __init__(self, idx):
        self.id = idx
        self.actions = []
        self.sex = 0
        self.weight = np.random.normal(70, 1)
        self.age = np.random.normal(55, 10)
        self.hr = np.random.normal(70, 5)
    
    def advance_time(self, time):
        for action in self.actions:
            self.hr += np.random.normal(action, 0.01)
        self.actions = []

    def results(self):
        return self.hr