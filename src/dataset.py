import pandas as pd
from torch.utils.data import Dataset

#This dataset wraps a pandas DataFrame and provides
#light text preprocessing and text&target pair for model training

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row['text']).replace("#", "").replace("@", "")
        target = row['target']
        return text, target

    def __len__(self):
        return len(self.data)
