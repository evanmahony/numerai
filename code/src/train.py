from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from model.wide.wide import NeuralNetwork
from model.wide.utils import get_loaders
import sys

if len(sys.argv) == 1:
    lr = 1e-3
    num_epochs = 5
    batch_size = 32
else:
    lr = float(sys.argv[1])
    num_epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])

print(
    f"Starting training with lr:{lr}, num_epochs:{num_epochs}, batch_size:{batch_size}")

path = f"runs/{datetime.now().strftime('%y_%m_%dT%H_%M')}"
writer = SummaryWriter(path)

train_loader, test_loader = get_loaders(batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


# Submissions are scored by spearman correlation
def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y.type(torch.float).unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    predictions = np.array([])
    targets = np.array([])

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            targets = np.append(targets, y.numpy())
            pred = model(X)
            predictions = np.append(predictions, pred)
            test_loss += loss_fn(pred, y.type(torch.float).unsqueeze(1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return [100*correct, test_loss, predictions, targets]


def main(train_loader, test_loader, model, loss_fn, optimizer):
    epochs = num_epochs

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        metrics = test(test_loader, model, loss_fn)

    writer.add_hparams({"lr": lr, "b_size": batch_size, "num_epochs": num_epochs},
                       {"accuracy": metrics[0],
                        "loss": metrics[1],
                        "correlation": correlation(pd.Series(metrics[2]), pd.Series(metrics[3]))},
                       run_name="Final")

    writer.add_text("Distribution", str(pd.Series(metrics[2]).describe()))
    torch.save(model.state_dict(), f"{path}/model")
    print("Done!")


if __name__ == "__main__":
    main(train_loader, test_loader, model, loss_fn, optimizer)
    writer.flush()
    writer.close()


# evanmahony.ie
