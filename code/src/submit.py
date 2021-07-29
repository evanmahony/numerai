import datatable as dt
from model.wide.wide import NeuralNetwork
import numpy as np
import pandas as pd
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

PATH = sys.argv[1]
BATCH_SIZE = int(sys.argv[2])

model = NeuralNetwork()
model.load_state_dict(torch.load(PATH))
model.eval()

print(model)

# Setting correct device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using datatable for performace reasons
df = dt.fread("/home/jovyan/data/numerai_tournament_data.csv")
df = df.to_pandas()


def main(device: str, batch_size:int, df) -> np.array:
    data = TensorDataset(torch.from_numpy(df.iloc[:, 3:-1].values))
    data_loader = DataLoader(data,
                             batch_size=batch_size,
                             shuffle=True)

    preds = np.array([])
    with torch.no_grad():
        for X in tqdm(data_loader):
            X = X[0].to(device)
            preds = np.append(preds, model(X).numpy())

    return preds


if __name__ == "__main__":
    preds = main(device, BATCH_SIZE, df)


# evanmahony.ie
