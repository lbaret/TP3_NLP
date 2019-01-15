import challenger as chal
import spacy
from spacy.gold import GoldParse
import re

class NER:

    def __init__(self, repertory, train, test, extension_train):
        # *** Déclaration de l'outil NER spaCy ***
        self.repertory = repertory
        self.train_file = train
        self.test_file = test
        self.extension_train = extension_train
        self.nlp = spacy.blank('en')

        # *** On appelle la classe que l'on a créé précédemment ***
        self.data = chal.Data(repertory=repertory, train=train, test=test, extension_train=extension_train)


    '''
        Fonction appelée lors de l'entrainement de la NER par spaCy
    '''
    def train_recognizer(self):
        # Tentons une technique de NER par patrons en utilisant la librairie spaCy
        comp = self.nlp.create_pipe('ner')
        self.nlp.add_pipe(comp)
        comp.add_label("Task")
        comp.add_label("Material")
        comp.add_label("Process")
        # nlp.from_disk('C:/Users/Lobar/Desktop/TP3_NLP/spacy_models')
        optimizer = self.nlp.begin_training()
        losses = {}
        for training in self.data.data_train:
            f = open(self.repertory + self.train_file + "/" + training[0], 'r', encoding="utf-8")
            text = f.readlines()
            text = text[0]
            entities = []
            for ent in training[1]:
                if ent[0]=="T":
                    splitted = re.split(r'\W+', ent[1])
                    entity = (int(splitted[1]), int(splitted[2]), splitted[0])
                    entities.append(entity)
            doc = self.nlp.make_doc(text)
            gold = GoldParse(doc, entities=entities)
            self.nlp.update([doc], [gold], drop=0.5, losses=losses, sgd=optimizer)
            f.close()
            ''' A MODIFIER LE CHEMIN D'ACCES '''
            self.nlp.to_disk("C:\\Users\\Lobar\\Desktop\\TP3_NLP\\spacy_models")

    '''
        Fonction appelée lors de la réalisation des tests
    '''
    def test_recognizer(self):
        # Testons notre entrainement NER
        nlp_bis = spacy.load("c:/Users/Lobar/Desktop/TP3_NLP/spacy_models")
        self.docs = []
        for testing in self.data.data_test:
            text = testing[1][0]
            doc = nlp_bis(text)
            self.docs.append(doc)
            nlp_bis.entity(doc)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    '''
        Fonction appelée lors de la prédiction de texte
    '''
    def predict(self, text):
        # Testons notre entrainement NER
        nlp_bis = spacy.load("c:/Users/Lobar/Desktop/TP3_NLP/spacy_models")
        doc = nlp_bis(text)
        nlp_bis.entity(doc)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
