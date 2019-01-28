'''
    Fichier pour mise en place de la fonction d'entrainement, de validation et de prédiction
'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from deeplib.history import History

def train(model, dataset, n_epoch, learning_rate):
    history = History()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(n_epoch):
        model.train()
        for j, inputs, targets in enumerate(dataset):
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        train_acc, train_loss = validate(model, dataset)
        history.save(train_acc, train_loss, learning_rate)
        print('Epoch {} - Train acc: {:.2f} - Train loss: {:.4f}'.format((i+1), train_acc, train_loss))

    return history


def validate(model, val_loader):
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch

        output = model(inputs)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).item())
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())

    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)
# def predict(): TODO : À définir
