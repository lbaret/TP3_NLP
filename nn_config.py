'''
    Fichier pour mise en place de la fonction d'entrainement, de validation et de prédiction
'''
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from deeplib.history import History
import numpy as np

def train(model, dataset, batch_size, n_epoch, learning_rate):
    history = History()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    nb_tour = (len(dataset[0]) // batch_size) + 1
    nb_batch_maxi = len(dataset[0]) % batch_size

    for i in range(n_epoch):
        model.train()
        inputs, targets = dataset
        zip_io = list(zip(inputs, targets))
        random.shuffle(zip_io)
        inputs, targets = zip(*zip_io)
        optimizer.zero_grad()
        inputs = list(inputs)
        # TODO : Il manque une dimension à mon tensor ??? Chercher à comprendre pourquoi et ou
        for j, inp in enumerate(inputs):
            const_list = [60 for k in range(0, len(inp))]
            const_tens = torch.tensor(const_list, dtype=torch.long)
            inp = torch.nn.utils.rnn.pack_padded_sequence(inp, const_tens, True)
            inputs[j] = inp
        output = model(inputs, [2, len(inputs), 50])

        loss = criterion(output, list(targets))
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
