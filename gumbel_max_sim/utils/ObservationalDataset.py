import pandas as pd
import torch
from torch.utils.data import Dataset


class ObservationalDataset(Dataset):
    def __init__(self, csv_file, xt_columns, action_column):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        data = pd.read_csv(csv_file)
        data_filtered = data[xt_columns]
        actions_data = data[action_column]
        self.ehr_data = []
        self.ehr_action_data = []
        trajectory = []
        actions = []
        trajectory_mask = []
        for i in range(len(data)):
            if i > 0 and data.iloc[i]['id'] == data.iloc[i-1]['id']:
                trajectory.append(torch.tensor(data_filtered.iloc[i].values))
                actions.append(torch.tensor(actions_data.iloc[i]))
            else:
                if i > 0:
                    self.ehr_data.append(torch.stack(trajectory))
                    self.ehr_action_data.append(torch.stack(actions))
                trajectory = [torch.tensor(data_filtered.iloc[i].values)]
                actions = [torch.tensor(actions_data.iloc[i])]
        self.ehr_data.append(torch.stack(trajectory))
        self.ehr_action_data.append(torch.stack(actions))

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ehr_data[idx]
        actions = self.ehr_action_data[idx]
        return sample, actions
