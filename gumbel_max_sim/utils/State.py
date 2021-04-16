import torch
from sepsisSimDiabetes.State import State


class State(State):
    NUM_OBS_STATES = 720
    NUM_HID_STATES = 2  # Binary value of diabetes
    NUM_PROJ_OBS_STATES = int(720 / 5)  # Marginalizing over glucose
    NUM_FULL_STATES = int(NUM_OBS_STATES * NUM_HID_STATES)

    def __init__(self, hr_state, sysbp_state, percoxyg_state, glucose_state, antibiotic_state, vaso_state, vent_state, diabetic_idx):
        self.hr_state = hr_state
        self.sysbp_state = sysbp_state
        self.percoxyg_state = percoxyg_state
        self.glucose_state = glucose_state
        self.antibiotic_state = antibiotic_state
        self.vaso_state = vaso_state
        self.vent_state = vent_state
        self.diabetic_idx = diabetic_idx

    def set_state_by_idx(self, state_idx):
        mod_idx = state_idx
        term_base = State.NUM_FULL_STATES
        self.diabetic_idx = torch.floor(mod_idx/term_base)
        mod_idx %= term_base
        term_base /= 3

        self.hr_state = torch.floor(mod_idx / term_base)
        mod_idx %= term_base
        term_base /= 3

        self.sysbp_state = torch.floor(mod_idx / term_base)
        mod_idx %= term_base
        term_base /= 2

        self.percoxyg_state = torch.floor(mod_idx / term_base)
        mod_idx %= term_base
        term_base /= 5

        self.glucose_state = torch.floor(mod_idx / term_base)
        mod_idx %= term_base
        term_base /= 2

        self.antibiotic_state = torch.floor(mod_idx / term_base)
        mod_idx %= term_base
        term_base /= 2

        self.vaso_state = torch.floor(mod_idx / term_base)
        mod_idx %= term_base
        term_base /= 2

        self.vent_state = torch.floor(mod_idx / term_base)
    
    def get_state_idx(self):
        categ_num = torch.stack([2, 3, 3, 2, 5, 2, 2, 2]*self.glucose_state.size(0))
        state_categs = torch.column_stack((
            self.diabetic_idx,
            self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state,))

        return (state_categs*categ_num).sum(axis=1)

    def get_state_tensor(self):
        return torch.column_stack((
            self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state,
            self.diabetic_idx,
        )).float()