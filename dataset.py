import torch 
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocess_data
import pandas as pd

df = pd.read_csv("option_pricing_data.csv")
dataset = preprocess_data(df)

class OptionPricingDataset(Dataset):
    def __init__ (self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = OptionPricingDataset(dataset["X_train"], dataset["y_train"])
val_dataset = OptionPricingDataset(dataset["X_val"], dataset["y_val"])
test_dataset = OptionPricingDataset(dataset["X_test"], dataset["y_test"])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


