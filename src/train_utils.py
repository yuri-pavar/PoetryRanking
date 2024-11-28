

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def test_epoch(model, val_loader, criterion, device):
    lost_list = []
    model.eval()
    for data in tqdm(val_loader):
        input_ids = data['input_ids'].to(device)
        padding_mask = data['attention_mask'].to(device)
        genres = data['genre'].to(device)
        views = data['views'].to(device)
        labels = data['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, padding_mask, genres, views)
            loss = criterion(outputs, labels)
            lost_list.append(loss.item())

    return np.sum(lost_list)/len(val_loader)


def train_epoch(model, train_loader, optimizer, criterion, device):
    loss_list = []
    model.train()
    for data in tqdm(train_loader):
        input_ids = data['input_ids'].to(device)
        padding_mask = data['attention_mask'].to(device)
        genres = data['genre'].to(device)
        views = data['views'].to(device)
        labels = data['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, padding_mask, genres, views)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    return np.sum(loss_list)/len(train_loader)


def train(model, train_loader, val_loader, optimizer, criterion, n_epochs,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    train_loss_list, val_loss_list = [], []
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = test_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}")
        print(f" train loss: {train_loss}")
        print(f" val loss: {val_loss}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    return train_loss_list, val_loss_list
