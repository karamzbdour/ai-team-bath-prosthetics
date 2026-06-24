import torch
from torch.utils.data import Dataset
import pandas as pd
import wfdb
import numpy as np
import os

class SensorDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data.iloc[i]

class PhysionetDataset(Dataset):
    def __init__(self, record_name, db_dir, download_path, window_size=1000):
        self.window_size = window_size
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(script_dir)
        local_download_path=os.path.join(script_dir, download_path)
        os.makedirs(local_download_path, exist_ok=True)

        dat_path = os.path.join(local_download_path, f"{record_name}.dat")
        hea_path = os.path.join(local_download_path, f"{record_name}.hea")
        if os.path.exists(dat_path) and os.path.exists(hea_path):
            print("Files already exist! Skipping download.")
        else:
            print(f"Starting download of {record_name} from {db_dir} on Physionet...")
            wfdb.dl_database(db_dir, local_download_path, records=[record_name])

        local_path = os.path.join(local_download_path, record_name)
        record = wfdb.rdrecord(local_path)

        raw = record.p_signal

        # normalise data
        self.mean = np.mean(raw, axis=0)
        self.std = np.std(raw, axis=0)
        normalised_data = (raw - self.mean) / (self.std + 1e-8)
        
        complete_windows = len(normalised_data) // self.window_size
        new_length = complete_windows * self.window_size
        self.data = normalised_data[:new_length]
        
        self.data = self.data.reshape(complete_windows, self.window_size, -1) # -1 allows us to pass in data with different channel sizes
        self.data = np.transpose(self.data, (0, 2, 1)) # re-orders the dimensions

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        data_tensor = torch.tensor(self.data[i], dtype=torch.float32)
        return data_tensor, data_tensor # technically, left is input and right is expected output

# if __name__ == "__main__":
#     ds = PhysionetDataset("100", "mitdb", "..\\artifacts\\datasets\\MITdb-ECG")
# ^^^^^ this downloads the record named "100" from the URL https://physionet.org/content/mitdb/1.0.0/