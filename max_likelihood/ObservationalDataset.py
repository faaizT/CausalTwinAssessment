import pandas as pd
import torch
from torch.utils.data import Dataset

cols = ['A_t', 'id', 't', 'ut_facial_expression', 'ut_socio_econ', 'xt_gender', 'xt_hr', 'xt_sysbp']


class ObservationalDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        data = pd.read_csv(csv_file)
        data = data[cols]
        self.ehr_data = []
        for i in range(len(data)):
            if i > 0 and data.iloc[i]['id'] == data.iloc[i-1]['id']:
                trajectory = torch.stack((trajectory, torch.tensor(data.iloc[i].values)))
            else:
                if i > 0:
                    self.ehr_data.append(trajectory)
                trajectory = torch.tensor(data.iloc[i].values)
        self.ehr_data.append(trajectory)

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ehr_data[idx]
        return sample
