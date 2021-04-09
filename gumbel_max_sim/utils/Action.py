from sepsisSimDiabetes.Action import Action
import torch

class Action(Action):
    NUM_ACTIONS_TOTAL = 8
    def __init__(self, action_idx=None, selected_actions=None):
        assert (selected_actions is not None and action_idx is None) \
            or (selected_actions is None and action_idx is not None), \
            "must specify either set of action strings or action index"
        if action_idx is not None:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL/2
            self.antibiotic = torch.floor(mod_idx/term_base)
            mod_idx %= term_base
            term_base /= 2
            self.ventilation = torch.floor(mod_idx/term_base)
            mod_idx %= term_base
            term_base /= 2
            self.vasopressors = torch.floor(mod_idx/term_base)

    def get_action_idx(self):
        return 4*self.antibiotic + 2*self.ventilation + self.vasopressors