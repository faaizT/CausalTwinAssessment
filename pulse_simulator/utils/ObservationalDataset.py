import pandas as pd
import torch
from torch.utils.data import Dataset


class ObservationalDataset(Dataset):
    def __init__(self, obs_file, sim_file, xt_columns, st_columns, action_columns, id_column="id", static_cols=None):
        """
        Args:
            obs_file (string): Path to the csv file with annotations.
        """
        data = pd.read_csv(obs_file)
        data_filtered = data[xt_columns]
        sim_data = pd.read_csv(sim_file)
        sim_data_filtered = sim_data[st_columns]
        actions_data = data[action_columns]
        self.static_cols = static_cols
        if static_cols is not None:
            static_data = data[static_cols]
            self.static_data = []
        self.ehr_data = []
        self.sim_data = []
        self.ehr_action_data = []
        trajectory = []
        sim_trajectory = []
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
                sim_trajectory.append(
                    torch.tensor(sim_data_filtered.iloc[i].values)
                )
                if -1 not in data_filtered.iloc[i].values:
                    length_counter += 1
            else:
                if i > 0:
                    self.ehr_data.append(torch.stack(trajectory))
                    self.sim_data.append(torch.stack(sim_trajectory))
                    self.ehr_action_data.append(torch.stack(actions))
                    self.lengths.append(torch.tensor(length_counter))
                trajectory = [
                    torch.zeros(len(xt_columns))
                    if -1 in data_filtered.iloc[i].values
                    else torch.tensor(data_filtered.iloc[i].values)
                ]
                sim_trajectory = [
                    torch.tensor(sim_data_filtered.iloc[i].values)
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
                if static_cols is not None:
                    self.static_data.append(torch.tensor(static_data.iloc[i].values))
        self.ehr_data.append(torch.stack(trajectory))
        self.sim_data.append(torch.stack(sim_trajectory))
        self.ehr_action_data.append(torch.stack(actions))
        self.lengths.append(torch.tensor(length_counter))

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ehr_data[idx]
        sim_sample = self.sim_data[idx]
        actions = self.ehr_action_data[idx]
        lengths = self.lengths[idx]
        if self.static_cols is None:
            return sample, actions, lengths
        static_data = self.static_data[idx]
        return sample, sim_sample, static_data, actions, lengths
