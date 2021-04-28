import pandas as pd
import torch
from torch.utils.data import Dataset


cols = [
    "hr_state",
    "sysbp_state",
    "percoxyg_state",
    "antibiotic_state",
    "vaso_state",
    "vent_state",
]


class ObservationalDataset(Dataset):
    def __init__(self, csv_file, xt_columns, action_columns, id_column="id"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        data = pd.read_csv(csv_file)
        data_filtered = data[xt_columns]
        actions_data = data[action_columns]
        self.ehr_data = []
        self.ehr_action_data = []
        trajectory = []
        actions = []
        self.lengths = []
        length_counter = 0
        for i in range(len(data)):
            if i > 0 and data.iloc[i][id_column] == data.iloc[i - 1][id_column]:
                trajectory.append(
                    torch.zeros(len(xt_columns))
                    if -1 in data_filtered.iloc[i].values
                    else torch.tensor(data_filtered.iloc[i].values)
                )
                actions.append(
                    torch.zeros(len(action_columns))
                    if -1 in actions_data.iloc[i].values
                    else torch.tensor(actions_data.iloc[i].values)
                )
                if -1 not in data_filtered.iloc[i].values:
                    length_counter += 1
            else:
                if i > 0:
                    self.ehr_data.append(torch.stack(trajectory))
                    self.ehr_action_data.append(torch.stack(actions))
                    self.lengths.append(torch.tensor(length_counter))
                trajectory = [
                    torch.zeros(len(xt_columns))
                    if -1 in data_filtered.iloc[i].values
                    else torch.tensor(data_filtered.iloc[i].values)
                ]
                actions = [
                    torch.zeros(len(action_columns))
                    if -1 in actions_data.iloc[i].values
                    else torch.tensor(actions_data.iloc[i].values)
                ]
                if -1 not in data_filtered.iloc[i].values:
                    length_counter = 1
                else:
                    length_counter = 0
        self.ehr_data.append(torch.stack(trajectory))
        self.ehr_action_data.append(torch.stack(actions))
        self.lengths.append(torch.tensor(length_counter))

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ehr_data[idx]
        actions = self.ehr_action_data[idx]
        lengths = self.lengths[idx]
        return sample, actions, lengths
