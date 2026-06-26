import torch, mne, os, wfdb
from torch.utils.data import Dataset
import numpy as np

class BonnEEGDataset(Dataset):
    '''
    Specifically for use with the BonnEEG dataset which uses .txt files with a number on each line. 
    '''
    def __init__(self, file_path, window_size=1000, transform=None, target_transform=None):
        self.window_size = window_size
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find the file at {file_path}")

        with open(file_path, "r") as f:
            raw = [float(line.strip()) for line in f.readlines() if line.strip()]

        raw = np.array(raw)
        self.mean = np.mean(raw)
        self.std = np.std(raw)
        normalised_data = (raw - self.mean) / (self.std + 1e-8)
        self.data = normalised_data

        complete_windows = len(normalised_data) // self.window_size
        self.data = self.data[:complete_windows*self.window_size]
        self.data = self.data.reshape(complete_windows, 1, self.window_size)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        original = torch.tensor(self.data[i], dtype=torch.float32)
        input_tensor = original.clone()
        target_tensor = original.clone()

        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)
        
        return input_tensor, target_tensor

class PhysionetDataset(Dataset):
    '''
    For processing .hea and .dat files
    Ensure files that make up a record have names {record_name}.hea and {record_name}.dat
    '''
    def __init__(self, record_name, relative_path, transform=None, target_transform=None, window_size=1000):
        self.window_size = window_size
        self.transform = transform
        self.target_transform = target_transform
        script_dir = os.path.dirname(os.path.abspath(__file__))

        local_download_path=os.path.join(script_dir, relative_path)
        local_path = os.path.join(local_download_path, record_name)
        record = wfdb.rdrecord(local_path)

        raw = record.p_signal[:, 0:1] # a bodge to drop extra data columns

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
        original = torch.tensor(self.data[i], dtype=torch.float32)
        input_tensor = original.clone()
        target_tensor = original.clone()

        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.target_transform:
            target_tensor = self.target_transform(target_tensor)

        return input_tensor, target_tensor


class MultimodalPhysioDataset(Dataset):
    '''
    Not as useful as i had envisioned. Unfinished and unuseful currently. 
    '''
    def __init__(self, edf_file_path=None, window_size=1000, usage=0.5):
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
        raw.pick(['ECG', 'EMG chin', 'EEG F4-M1'])
        to_keep = int(raw.n_times * usage)

        self.data = raw.get_data(start=0, stop=to_keep)

        # standardise data
        self.data = (self.data - np.mean(self.data, axis=1, keepdims=True)) / np.std(self.data, axis=1, keepdims=True) 

        self.window_size = window_size
        self.num_windows = self.data.shape[1] // self.window_size

    def __len__(self):
        return self.num_windows

    def __getitem__(self, i):
        start_index = i * self.window_size
        end_index = start_index + self.window_size

        sliced_data = self.data[:, start_index:end_index]
        sliced_data = torch.FloatTensor(sliced_data)

        ecg = sliced_data[0].unsqueeze(0)
        emg = sliced_data[1].unsqueeze(0)
        eeg = sliced_data[2].unsqueeze(0)

        return ecg, emg, eeg