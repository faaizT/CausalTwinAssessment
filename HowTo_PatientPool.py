from pulse.utils.PulseInterface import *
from gumbel_max_sim.utils.ObservationalDataset import ObservationalDataset
from pulse.utils.MIMICDataset import cols, action_cols

def HowTo_PatientPool():
    pool = PulsePhysiologyEnginePool()
    p1 = pool.create_patient(0)
    p2 = pool.create_patient(1)
    
    pool.advance_time_s(60)

    for p in pool.get_patients():
        if p.results() < 80.:
            p.actions.append(1)
        
    pool.advance_time_s(60)
    
    for p in pool.get_patients():
        print("HR is %.4f" % p.results())

HowTo_PatientPool()
dataset = ObservationalDataset('/data/localhost/taufiq/export-dir/MIMIC-1hourly-processed.csv', xt_columns=cols, action_columns=action_cols, id_column="icustay_id")
print(dataset[1])