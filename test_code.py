from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.initializers import lecun_normal
from keras.preprocessing.text import Tokenizer
import challenger as chal

'''
    Pour l'extraction de relations, nous utiliserons un réseau de neuronnes LSTM
'''
class ER_LSTM:

    def __init__(self, repertory, train, test, extension_train):
        # *** Construction de notre Réseau de Neuronnes ***
        initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
        rec_initializer = lecun_normal(seed=None)
        er = LSTM(3, kernel_initializer=initializer, recurrent_initializer=rec_initializer)

        # *** Construction du modèle ***
        self.model = Sequential()
        self.model.add(er)
        self.model.add(Dropout(0.5))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        # *** Récupération de nos données d'entrainement ***
        self.data = chal.Data(repertory=repertory, train=train, test=test, extension_train=extension_train)

        # Entrainement du réseau
        for data_t in self.data.data_train:
            if "R" in dict(data_t[1]):
                f = open(repertory + train + "/" + data_t[0], 'r', encoding="utf-8")
                text = f.readline()
