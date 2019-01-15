from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.initializers import RandomNormal
from keras.initializers import lecun_normal
from keras.preprocessing import sequence
import re
import challenger as chal
import numpy as np

'''
    Pour l'extraction de relations, nous utiliserons un réseau de neuronnes LSTM
'''
class ER_LSTM:

    def __init__(self, repertory, train, test, extension_train):
        # *** Construction de notre Réseau de Neuronnes ***
        initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
        rec_initializer = lecun_normal(seed=None)
        er = LSTM(2, kernel_initializer=initializer, recurrent_initializer=rec_initializer, return_sequences=False)

        # *** Construction du modèle ***
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(1000, 1000), return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(er)
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='adam', loss='mse', metrics=['categorical_accuracy'])

        # *** Récupération de nos données d'entrainement ***
        self.data = chal.Data(repertory=repertory, train=train, test=test, extension_train=extension_train, destruct=False, split=True)

        # Entrainement du réseau
        self.training = []
        for data_t in self.data.data_train:
            if "R" in dict(data_t[1]):
                f = open(repertory + train + "/" + data_t[0], 'r', encoding="utf-8")
                input = f.readline()
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
                        output.append(y)
                    self.training.append((input, output))
        print(self.training)
        # === ENTRAINEMENT DE NOTRE RESEAU ===
        data = np.array([inputs for inputs, outputs in self.training])
        targets = np.array([outputs for inputs, outputs in self.training])
        seq = sequence.TimeseriesGenerator(data=data, targets=targets, length=(len(data)-1))
        self.model.fit_generator(seq)
