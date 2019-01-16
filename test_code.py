from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Activation
from keras.initializers import RandomNormal
from keras.initializers import lecun_normal
from keras.preprocessing.text import Tokenizer
import re
import challenger as chal

'''
    Pour l'extraction de relations, nous utiliserons un réseau de neuronnes LSTM
'''
class ER_LSTM:

    def __init__(self, repertory, train, test, extension_train):

        # *** Récupération de nos données d'entrainement ***
        self.data = chal.Data(repertory=repertory, train=train, test=test, extension_train=extension_train,
                              destruct=False, split=True)

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
                        y_tuple = (y[0], y[1], y[2])
                        output.append(y_tuple)
                    self.training.append((input, output))
        print(self.training)
        # === ENTRAINEMENT DE NOTRE RESEAU ===
        # On peut utiliser une fonction utile de Keras qui convertit le texte en liste de mots embarqués (grâce au tokenizer)
        tokenizor = Tokenizer(char_level=True, lower=False, split=' ')
        tokenizor.fit_on_texts([inputs for inputs, outputs in self.training])  # On entraine nos words embeddings
        x_train = tokenizor.texts_to_matrix([inputs for inputs, outputs in self.training])
        targets = tokenizor.texts_to_matrix([outputs for inputs, outputs in self.training])

        # *** Construction de notre Réseau de Neuronnes ***
        initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)
        rec_initializer = lecun_normal(seed=None)
        er = LSTM(2, kernel_initializer=initializer, recurrent_initializer=rec_initializer, return_sequences=False)

        # *** Construction du modèle ***
        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(len(x_train), len(x_train)), return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10))
        self.model.add(Activation('relu'))
        self.model.add(er)
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10))
        self.model.add(Activation('relu'))
        self.model.add(Embedding(1000, 64, input_length=10))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['categorical_accuracy'])
        self.model.fit_generator(x_train, targets)
