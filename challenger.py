import os
import glob

# Utilisons dans un premier temps une classe pour le traitement des données
class Data :

    data_train = []
    data_test = []
    data_test_ann = []

    '''
    extention : without "."
    repertory : from C: file
    train : file name for train data
    test : file name for test data
    '''
    def __init__(self, repertory, train="train2", test="test_unlabelled", extension_train="ann", destruct=True, split=False):
        base = os.path.dirname(os.path.abspath(__file__))
        # mod = imp.module_from_file('util', base+'\\scripts\\util.py')

        # Préparation des données d'entrainement
        os.chdir(repertory + train)
        file_form = "*." + extension_train
        files = glob.glob(file_form)

        for file in files:
            f = open(repertory + train + "/" + file, 'r', encoding="utf-8")
            lines = f.readlines()
            inter = []
            for line in lines:
                if line[0] == "T":
                    line = line.replace("\n", "")
                    line = line.split("\t")
                    if destruct:
                        line[0] = "T"
                        if not split:
                            inter.append(line)
                        else:
                            ent = line[0]
                            del line[0]
                            el = (ent, line)
                            inter.append(el)
                    else:
                        ent = line[0]
                        del line[0]
                        el = (ent, line)
                        inter.append(el)
                if line[0] == "R":
                    line = line.replace("\n", "")
                    line = line.split("\t")
                    line[0] = "R"
                    ent = line[0]
                    del line[0]
                    el = (ent, line)
                    inter.append(el)
            f.close()
            self.data_train.append((file.replace(extension_train, "txt"), inter))

        print("Train data ready !")

        # Préparation des données de test
        os.chdir(repertory + test)
        file_form = "*.txt"     # On considère qu'on ne peut lui donner que des fichiers textes
        files = glob.glob(file_form)

        if len(glob.glob("*.ann")) > 0:
            for file in files:
                f = open(repertory + test + "/" + file, 'r', encoding="utf-8")
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
                self.data_test.append((file.replace(extension_train, "txt"), inter))
        else:
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
