import torch
from torch.utils.data import Dataset
import pandas as pd

class LanguageDetectionDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.text = self.data['Text'].tolist()
        self.language = self.data['Language'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.text[idx]
        language = self.language[idx]
        return text, language