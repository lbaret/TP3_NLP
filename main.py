import challenger as chal
import spacy
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer as ER
import re
import random
import sys
sys.setrecursionlimit(5000)

'''
    ATTENTION : Changer les valeurs de la variable repertory pour pointer vers le fichier de projet TP3_NLP
    Il en va de même si vous voulez utiliser des fichiers d'entrainement et de test différents
'''

repertory = "C:/Users/Lobar/Desktop/TP3_NLP/"
train_file = "train2"
test_file = "semeval_articles_test"

# *** Déclaration de l'outil NER spaCy ***
nlp = spacy.blank('en')

# *** On appelle la classe que l'on a créé précédemment ***
data = chal.Data(repertory=repertory, train=train_file, test=test_file, extension_train="ann")

# Tentons une technique de NER par patrons en utilisant la librairie spaCy
comp = nlp.create_pipe('ner')
nlp.add_pipe(comp)
comp.add_label("Task")
comp.add_label("Material")
comp.add_label("Process")
# nlp.from_disk('C:/Users/Lobar/Desktop/TP3_NLP/spacy_models')
optimizer = nlp.begin_training()
losses = {}
for training in data.data_train:
    f = open(repertory + train_file + "/" + training[0], 'r', encoding="utf-8")
    text = f.readlines()
    text = text[0]
    entities = []
    for ent in training[1]:
        if ent[0]=="T":
            splitted = re.split(r'\W+', ent[1])
            entity = (int(splitted[1]), int(splitted[2]), splitted[0])
            entities.append(entity)
    doc = nlp.make_doc(text)
    gold = GoldParse(doc, entities=entities)
    nlp.update([doc], [gold], drop=0.5, losses=losses, sgd=optimizer)
    f.close()
    ''' A MODIFIER LE CHEMIN D'ACCES '''
    nlp.to_disk("C:\\Users\\Lobar\\Desktop\\TP3_NLP\\spacy_models")

# Testons notre entrainement NER
nlp_bis = spacy.load("c:/Users/Lobar/Desktop/TP3_NLP/spacy_models")
docs = []
for testing in data.data_test:
    text = testing[1][0]
    doc = nlp_bis(text)
    docs.append(doc)
    nlp_bis.entity(doc)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
