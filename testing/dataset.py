import torch
from torch.utils.data import Dataset
import pandas as pd

class LanguageDetectionDataset(Dataset):
    """Small language detection dataset. Contains text and language columns for 17 different languages."""
    def __init__(self, path):
        # Load data from csv file
        self.data = pd.read_csv(path)
        # Get text and language columns
        self.text = self.data['Text'].tolist()
        self.language = self.data['Language'].tolist()
    # Return the length of the dataset
    def __len__(self):
        return len(self.data)
    # Return text and language at specific index
    def __getitem__(self, idx):
        text = self.text[idx]
        language = self.language[idx]
        return text, language