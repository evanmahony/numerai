import datatable as dt
from model.wide.wide import NeuralNetwork
import numpy as np
import pandas as pd
import sys
import torch

PATH = sys.argv[1]
BATCH_SIZE = 32

model = NeuralNetwork()
model.load_state_dict(torch.load(PATH))
model.eval()

print(model)

df = dt.fread("/home/jovyan/data/numerai_tournament_data.csv")
df = df.to_pandas()

data = torch.utils.data.TensorDataset(torch.from_numpy(df.iloc[:, 3:-1].values))
data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

preds = np.array([])
with torch.no_grad():
    for X in data_loader:
        X = X[0].to(device)
        preds = np.append(preds, model(X).numpy())