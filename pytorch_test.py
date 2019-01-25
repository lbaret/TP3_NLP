import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk import pos_tag, word_tokenize
import challenger as chal
import collections as col
import re

# TODO : Trouver un moyen pour créer un words embedding des entités. MAJ : Toujours à réfléchir le reste est fait.
class Neural():
    def __init__(self, repertory, train, test, extension_train):

        self.data = chal.Data(repertory=repertory, train=train, test=test, extension_train=extension_train,
                              destruct=False, split=True)

        # *** Récupération et traitement des données d'entrainement ***
        self.training = []
        full_text = []  # Nous servira pour générer les embeddings
        for data_t in self.data.data_train:
            if "R" in dict(data_t[1]):
                f = open(repertory + train + "/" + data_t[0], 'r', encoding="utf-8")
                input = f.readline()
                full_text.append(input)
                f.close()
                rels = [(x, y) for x, y in data_t[1] if x == "R"]
                output = []
                if len(rels) > 0:
                    for (x, y) in rels:
                        y.pop()
                        splitted = re.split(r'\W+', y[0])
                        splitted[0] = splitted[0] + "-" + splitted[1]
                        splitted.pop(1)
                        splitted.pop(1)
                        splitted.pop(2)
                        y = splitted
                        for other, el in data_t[1]:
                            if other == y[1]:
                                y[1] = el[1]
                            if other == y[2]:
                                y[2] = el[1]
                        y_tuple = (y[0], y[1], y[2])
                        output.append(y_tuple)
                    self.training.append((input, output))

        # *** Création des Words Embeddings ***
        embeds_pos = nn.Embedding(num_embeddings=12, embedding_dim=5)
        embeds_ner = nn.Embedding(num_embeddings=4, embedding_dim=5)

        # *** POS Tagging des données d'entrainement ***
            # Pour se faire il faut récupérer la partie inputs
        inputs = [input for input, output in self.training]
        tokens = []
        pos_tags = []
        for input in inputs:
            tok = word_tokenize(input)
            tokens.append(tok)
            pos_tags.append(pos_tag(tok))

        # *** Création des Words Embeddings pour les tokens ***
        flat_tokens = [item for sublist in tokens for item in sublist]
            # On Count pour connaitre le nombre de tokens différents
        nb_mots_diff = col.Counter(flat_tokens)
        embeds_words = nn.Embedding(num_embeddings=len(nb_mots_diff), embedding_dim=1)
