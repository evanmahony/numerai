import datatable as dt
import pandas as pd
import torch

def get_loaders(batch_size=32):
    df = dt.fread("/home/jovyan/data/numerai_training_data.csv")
    df = df.to_pandas()

    train_df = df.sample(frac = 0.8)
    test_df = df.drop(train_df.index)

    train = torch.utils.data.TensorDataset(torch.from_numpy(train_df.iloc[:, 3:-1].values), torch.from_numpy(train_df["target"].values))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test = torch.utils.data.TensorDataset(torch.from_numpy(test_df.iloc[:, 3:-1].values), torch.from_numpy(test_df["target"].values))
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader