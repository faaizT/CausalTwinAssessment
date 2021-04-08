# from sepsisSimDiabetes.State import State
# from sepsisSimDiabetes.Action import Action
# from sepsisSimDiabetes.MDP import MDP
# import torch


# class MdpPyro(MDP):
#     def __init__(self, init_state_idx=None, init_state_idx_type="obs"):
#         super().__init__(
#             init_state_idx=init_state_idx, init_state_idx_type=init_state_idx_type
#         )

#     # def pyro_transition()

#     # def transition_antibiotics_on(self):
#     #     """
#     #     antibiotics state on
#     #     heart rate, sys bp: hi -> normal w.p. .5
#     #     """
#     #     self.state.antibiotic_state = 1
#     #     if self.state.hr_state == 2:
#     #         hr_probs = torch.tensor([0.0, 0.5, 0.5])
#     #     else:
#     #         hr_probs = torch.zeros(3)
#     #         hr_probs[self.state.hr_state] = 1.0
#     #     if self.state.sysbp_state == 2:
#     #         sysbp_probs = torch.tensor([0.0, 0.5, 0.5])
#     #     else:
#     #         sysbp_probs = torch.zeros(3)
#     #         sysbp_probs[self.state.sysbp_state] = 1.0

#     # def transition_antibiotics_off(self):
#     #     """
#     #     antibiotics state off
#     #     if antibiotics was on: heart rate, sys bp: normal -> hi w.p. .1
#     #     """
#     #     if self.state.antibiotic_state == 1:
#     #         if self.state.hr_state == 1:
#     #             hr_probs = torch.tensor([0.0, 0.9, 0.1])
#     #         else:
#     #             hr_probs = torch.zeros(3)
#     #             hr_probs[self.state.hr_state] = 1.0
#     #         if self.state.sysbp_state == 1:
#     #             sysbp_probs = torch.tensor([0.0, 0.9, 0.1])
#     #         else:
#     #             sysbp_probs = torch.zeros(3)
#     #             sysbp_probs[self.state.sysbp_state] = 1.0
#     #         self.state.antibiotic_state = 0

#     # def transition_vent_on(self):
#     #     """
#     #     ventilation state on
#     #     percent oxygen: low -> normal w.p. .7
#     #     """
#     #     self.state.vent_state = 1
#     #     if self.state.percoxyg_state == 0:
#     #         percoxyg_probs = torch.tensor([0.3, 0.7, 0.0])
#     #     else:
#     #         percoxyg_probs = torch.zeros(3)
#     #         percoxyg_probs[self.state.percoxyg_state] = 1.0

#     # def transition_vent_off(self):
#     #     """
#     #     ventilation state off
#     #     if ventilation was on: percent oxygen: normal -> lo w.p. .1
#     #     """
#     #     if self.state.vent_state == 1:
#     #         if self.state.percoxyg_state == 1:
#     #             percoxyg_probs = torch.tensor([0.1, 0.9, 0.0])
#     #         else:
#     #             percoxyg_probs = torch.zeros(3)
#     #             percoxyg_probs[self.state.percoxyg_state] = 1.0
#     #         self.state.vent_state = 0

#     # def transition_vaso_on(self):
#     #     """
#     #     vasopressor state on
#     #     for non-diabetic:
#     #         sys bp: low -> normal, normal -> hi w.p. .7
#     #     for diabetic:
#     #         raise blood pressure: normal -> hi w.p. .9,
#     #             lo -> normal w.p. .5, lo -> hi w.p. .4
#     #         raise blood glucose by 1 w.p. .5
#     #     """
#     #     self.state.vaso_state = 1
#     #     if self.state.diabetic_idx == 0:
#     #         if self.state.sysbp_state == 0:
#     #             sysbp_probs = torch.tensor([0.3, 0.7, 0.0])
#     #         elif self.state.sysbp_state == 1:
#     #             sysbp_probs = torch.tensor([0.0, 0.3, 0.7])
#     #     else:
#     #         if self.state.sysbp_state == 1:
#     #             if np.random.uniform(0, 1) < 0.9:
#     #                 self.state.sysbp_state = 2
#     #         elif self.state.sysbp_state == 0:
#     #             up_prob = np.random.uniform(0, 1)
#     #             if up_prob < 0.5:
#     #                 self.state.sysbp_state = 1
#     #             elif up_prob < 0.9:
#     #                 self.state.sysbp_state = 2
#     #         if np.random.uniform(0, 1) < 0.5:
#     #             self.state.glucose_state = min(4, self.state.glucose_state + 1)