import pandas as pd
import torch
from torch.utils.data import Dataset


class ObservationalDataset(Dataset):
    def __init__(self, csv_file, xt_columns, action_columns):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        data = pd.read_csv(csv_file)
        data_filtered = data[xt_columns]
        actions_data = data[action_columns]
        self.ehr_data = []
        self.ehr_action_data = []
        self.ehr_mask = []
        trajectory = []
        actions = []
        trajectory_mask = []
        for i in range(len(data)):
            if i > 0 and data.iloc[i]["id"] == data.iloc[i - 1]["id"]:
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
                trajectory_mask.append(
                    torch.tensor(int(-1 not in data_filtered.iloc[i].values))
                )
            else:
                if i > 0:
                    self.ehr_data.append(torch.stack(trajectory))
                    self.ehr_action_data.append(torch.stack(actions))
                    self.ehr_mask.append(torch.stack(trajectory_mask))
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
                trajectory_mask = [
                    torch.tensor(int(-1 not in data_filtered.iloc[i].values))
                ]
        self.ehr_data.append(torch.stack(trajectory))
        self.ehr_action_data.append(torch.stack(actions))
        self.ehr_mask.append(torch.stack(trajectory_mask).to(torch.bool))

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ehr_data[idx]
        actions = self.ehr_action_data[idx]
        masks = self.ehr_mask[idx]
        return sample, actions, masks
