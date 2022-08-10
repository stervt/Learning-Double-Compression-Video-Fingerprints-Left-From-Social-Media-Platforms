import torch
from torch.nn import functional as F
import os
import sys
import time
import errno
import shutil
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.utils as vutils


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader)
    n = int(size/10)
    model.train()
    num_batches = len(dataloader)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss+loss.item()
        if batch % n == 0:
            loss, current = loss.item(), batch 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss/num_batches
    print(f'training_loss: {avg_loss:>7f}')

def evaluate(dataloader, model, loss_fn, device, dtype):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{dtype}Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

def train_multi(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader)
    n = int(size/10)
    model.train()
    num_batches = len(dataloader)
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = [X[0].to(device),X[1].to(device)]
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss+loss.item()
        if batch % n == 0:
            loss, current = loss.item(), batch 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss/num_batches
    print(f'training_loss: {avg_loss:>7f}')

def evaluate_multi(dataloader, model, loss_fn, device, dtype):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = [X[0].to(device),X[1].to(device)]
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{dtype}Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

