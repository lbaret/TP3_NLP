'''
    Auteur : Loïc Baret
    Projet : NLP - Extraction d'informations
    Entité : Université Laval
    Faculté : Sciences et Génie
    Session : Automne 2018
    Professeur : Luc Lamontagne

    Dans le cadre de ce projet, nous travaillons sous l'environnement Anaconda.
    Il est question d'utiliser les réseaux de neuronnes afin d'extraire les informations et les relations.
    Nous testerons 4 réseaux de neuronnes différents :
        - GRU
        - LSTM
        - Convolutional LSTM
        - RNN Simple
    Le but étant non seulement de répondre aux critères du concours, mais en plus de déterminer lequel de ces 3 réseaux,
    très utilisés dans l'univers de la NLP répond le mieux aux enjeux de cette application.
'''

import os
import glob
import keras
import important as imp

# Utilisons dans un premier temps une classe pour le traitement des données
class Data :

    data_train = []
    data_test = []

    '''
    extention : without "."
    repertory : from C: file
    train : file name for train data
    test : file name for test data
    '''
    def __init__(self, repertory, train="train2", test="test_unlabelled", extension_train="ann"):
        base = os.path.dirname(os.path.abspath(__file__))
        # mod = imp.module_from_file('util', base+'\\scripts\\util.py')

        # Préparation des données d'entrainement
        os.chdir(repertory + train)
        file_form = "*." + extension_train
        files = glob.glob(file_form)

        for file in files:
            # mod.readAnn(repertory + train)
            f = open(repertory + train + "/" + file, 'r', encoding="utf-8")
            lines = f.readlines()
            inter = []
            for line in lines:
                if line[0] == "T":
                    line = line.replace("\n", "")
                    line = line.split("\t")
                    line[0] = "T"
                    inter.append(line)
                if line[0] == "R":
                    line = line.replace("\n", "")
                    line = line.split("\t")
                    line[0] = "R"
                    inter.append(line)
            f.close()
            self.data_train.append((file.replace(extension_train, "txt"), inter))

        print("Train data ready !")

        # Préparation des données de test
        os.chdir(repertory + test)
        file_form = "*.txt"     # On considère qu'on ne peut lui donner que des fichiers textes
        files = glob.glob(file_form)

        for file in files:
            f = open(repertory + test + "/" + file, 'r', encoding="utf-8")
            lines = f.readlines()
            inter = []
            for line in lines:
                line = line.replace("\n", "")
                inter.append(line)
            f.close()
            self.data_test.append((file, inter))

        print("Test data ready !")
