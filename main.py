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

import spacy_ner as spner
import test_code as rext
import challenger as chal

# LIMITATION IMPORTANTE POUR OBSERVER DES RESULTATS
import sys
sys.setrecursionlimit(5000)

'''
    ATTENTION : Changer les valeurs de la variable repertory pour pointer vers le fichier de projet TP3_NLP.
    Il en va de même si vous voulez utiliser des fichiers d'entrainement et de test différents.
'''

repertory = "C:/Users/Lobar/Desktop/TP3_NLP/"
train_file = "train2"
test_file = "semeval_articles_test"

# *** Définissons la partie NER par l'utilisation de spaCy ***
# spacy = spner.NER(repertory=repertory, train=train_file, test=test_file, extension_train="ann")

# *** Lançons l'entrainement ***
# spacy.train_recognizer()

# *** Nous pouvons obtenir une séquence de test par défaut ***
# spacy.test_recognizer()

# *** L'extraction de relation se faire
# test = rext.ER_LSTM(repertory=repertory, train=train_file, test=test_file, extension_train="ann")

data = chal.Data(repertory=repertory, train=train_file, test=test_file, extension_train="ann", destruct=True)
