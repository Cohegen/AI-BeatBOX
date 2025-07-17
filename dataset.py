import os
import numpy as np
from torch.utils.data import Dataset

class BeatboxDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        waveform = np.load(file_path).astype(np.float32)
        return waveform, file_name
