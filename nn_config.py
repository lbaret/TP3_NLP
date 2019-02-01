'''
    Fichier pour mise en place de la fonction d'entrainement, de validation et de prédiction
'''
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from deeplib.history import History

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
        # for j in range(0, nb_tour):
        #     nb_t_batch = batch_size if j != (nb_tour - 1) else nb_batch_maxi
        #     for k in range(0, nb_t_batch):
        optimizer.zero_grad()
        output = model(list(inputs))

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
