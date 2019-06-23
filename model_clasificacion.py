# coding=utf-8

from PyQt4.QtGui import *
from PyQt4.QtCore import *
import numpy as np
import sys
import os.path
from datetime import datetime
import scipy.sparse as sp
import functools
import sklearn.metrics.base
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import sklearn.metrics
from sklearn.metrics import classification_report,confusion_matrix


# Parametros para los clasificadores base
gk = 5
gn_neighbors = 5
gn_estimators = 10
gcriterion_rf = 'gini'  # also supported 'entropy'
gcriterion_dt = 'gini'
gkernel = 'rbf'
ggamma = 0.0
gC = 1.0
# ---------
ejecs = 0


def readDataFromFile(self, fileName):
    """This functions reads data from a file and store it in two matrices"""
    # Open the file
    try:
        file = open(fileName, 'r')
    except IOError:
        tmp = str('Error al cargar los folds, archivo: ' + str(fileName))
        self.emit(SIGNAL('infoclassif'), tmp)

        return np.array([]), np.array([])

    # Now we have to read the first line and check if it's sparse or dense
    firstLine = file.readline()
    words = firstLine.split()
    word = words[1]
    if word[:-1] == 'SPARSE':
        sparse = True  # The file is in sparse mode
    else:
        sparse = False  # The file is in dense mode

    secondLine = file.readline()
    words = secondLine.split()
    instances = int(words[1])
    thirdLine = file.readline()
    words = thirdLine.split()
    attributes = int(words[1])
    fourthLine = file.readline()
    words = fourthLine.split()
    labels = int(words[1])
    # Now we do a loop reading all the other lines
    # Then we read the file, different way depending if sparse or dense

    # The loop starts in the first line of data
    # We have to store that data in two matrices
    X = np.zeros((instances, attributes), dtype=float)
    y = np.zeros((instances, labels), dtype=int)
    numberLine = 0
    for line in file.readlines():
        putToX = True
        firstIndex = 1
        numberData = 0
        numberY = 0
        for data in line.split():
            if sparse:  # Sparse format, we have to split each data
                if data == '[':
                    putToX = False

                if putToX == True and (data != '[' and data != ']'):
                    sparseArray = data.split(':')
                    lastIndex = int(sparseArray[0])
                    for i in range(firstIndex, lastIndex - 1):
                        X[numberLine, i - 1] = float(0)
                    X[numberLine, lastIndex - 1] = float(sparseArray[1])
                    firstIndex = lastIndex - 1
                else:
                    if (data != '[') and (data != ']'):
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1

            else:  # Dense format
                if data == '[':
                    putToX = False

                if putToX == True and (data != '[' and data != ']'):
                    X[numberLine, numberData] = float(data)
                else:
                    if (data != '[') and (data != ']'):
                        # This is good for the dense format
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
            numberData += 1

        numberLine += 1
    X = sp.csr_matrix(X)
    file.close()
    return X, y


def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
    def _binary_uninterpolated_average_precision(
            y_true, y_score, pos_label=1, sample_weight=None):
        precision, recall, t = sklearn.metrics.precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

        recall[np.isnan(recall)] = 0

        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    average_precision = functools.partial(_binary_uninterpolated_average_precision,
                                          pos_label=pos_label)

    return sklearn.metrics.base._average_binary_score(average_precision, y_true, y_score,
                                                      average, sample_weight=sample_weight)


# TODO : https://nikolak.com/pyqt-threading-tutorial/
#   usar threading para ejecutar estas llamadas en paralelo, o por lo menos no dejar pillada la UI durante el proceso


#  https://stackoverflow.com/questions/2846653/how-to-use-threading-in-python#28463266

#TODO from multiprocessing import Process
# def f(name):
#    print 'hello', name
# if __name__ == '__main__':
#    p = Process(target=f, args=('bob',))
#    p.start()
#    p.join()


def make_classif(self, nfolds, fname, cl, parm, stratif, dir):

    fold_accuracy = []
    fold_hamming = []
    fold_prec = []
    fold_precm = []

    fold_auc = []
    fold_cover = []
    fold_rank = []

    if stratif == 'Random':
        outpname = dir + '/csv/'+str(cl).split('(')[0]+'_random.csv'

    if stratif == 'Labelset':
        outpname = dir + '/csv/' + str(cl).split('(')[0] + '_labelset.csv'

    if stratif == 'Iterative':
        outpname = dir + '/csv/' + str(cl).split('(')[0] + '_iterative.csv'

    if not os.path.isfile(outpname):
        fp = open(outpname, 'a')
        fp.write(
            'Timestamp;Clasif base;Parámetros;Dataset;Accuracy↑;Hamming Loss↓;Coverage↑;Ranking loss↑;'
            'Avg precision macro↑;Avg precision micro↑;ROC AUC↑;f1 score (micro)↑;Recall (micro)↑;'
            'f1 score (macro)↑;Recall (macro)↑' + '\n')
        fp.close()

    # TODO: el msnj se muestra duplicado en prueba21.xml (error en clasif con SVM)
    print 'solo una vez por ejecuc..'
    tmp = u"Ejecutando clasificación " + str(cl).split('(')[0] + " con el estratificado: " + stratif

    self.emit(SIGNAL('infoclassif'), tmp)

    prog = 0

    for i in range(0, nfolds):

        prog += 100/nfolds
        self.emit(SIGNAL("progress"), prog)
        skip = 0

        suffix = os.path.basename(str(fname))
        suffix = os.path.splitext(suffix)[0]
        route = dir + '/' + suffix + '/' + suffix + str(nfolds) + '_' + stratif.lower() + '/'

        X_train, y_train = readDataFromFile(self, route + suffix + str(i) + '.train')
        X_test, y_test = readDataFromFile(self, route + suffix + str(i) + '.test')

        if X_train.size <= 1 or X_test.size <= 1:
            self.emit(SIGNAL('infoclassif'), 'ERROR1')
            return

        for j in range(0, y_train.shape[1]):
            if len(np.unique(y_train[:, j])) == 1:
                skip = 1
        cl.fit(X_train, y_train)
        y_score = cl.predict(X_test)

        if skip == 0:
            y_prob = cl.predict_proba(X_test.todense())
            # -----------------------------------------#
            # Coverage\n",
            c = sklearn.metrics.coverage_error(y_test, y_prob.toarray(), sample_weight=None)
            fold_cover.append(c)

            # -----------------------------------------#
            # Ranking loss\n",
            rl = sklearn.metrics.label_ranking_loss(y_test, y_prob.toarray(), sample_weight=None)
            fold_rank.append(rl)
            # -----------------------------------------#
            # Mean average precision
            m = average_precision_score(y_test, y_prob.toarray(), average='macro', pos_label=1, sample_weight=None)
            fold_prec.append(m)

            m2 = average_precision_score(y_test, y_prob.toarray(), average='micro', pos_label=1, sample_weight=None)
            fold_precm.append(m2)

            # -----------------------------------------#
            # Micro-average AUC
            rmi = sklearn.metrics.roc_auc_score(y_test, y_prob.toarray(), average='micro', sample_weight=None,
                                                max_fpr=None)
            fold_auc.append(rmi)

            # -----------------------------------------#
            # Medidas: sklearn.metrics...(true,predict,..)
        acc = sklearn.metrics.accuracy_score(y_test, y_score)
        fold_accuracy.append(acc)
        # -----------------------------------------#
        hl = sklearn.metrics.hamming_loss(y_test, y_score)
        fold_hamming.append(hl)

    fd = open(outpname, 'a')
    tmp = os.path.basename(str(fname))
    s = os.path.splitext(tmp)[0]

    tstamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fd.write(str(tstamp) + ';')  # marca de tiempo

    cbase = str(cl).split('(')[1]
    cbase = cbase.split('=')[1]

    if not str(cl).split('(')[0] == 'MLkNN':
        fd.write(cbase + ';')  # Clasific base
    else:
        fd.write('MLkNN' + ';')

    fd.write(parm + ';')  # Params de los metodos

    fd.write(str(s) + ';')  # El nombre del dataset
    # fp.write("Accuracy: ")
    fd.write(str(sum(fold_accuracy) / len(fold_accuracy)) + ';')
    # fp.write("Hamming loss: ")
    fd.write(str(sum(fold_hamming) / len(fold_hamming)) + ';')

    # fp.write("Coverage: ")
    if len(fold_cover) > 0:
        fd.write(str(sum(fold_cover) / len(fold_cover)) + ';')

    # fp.write("Ranking loss: ")
    if len(fold_rank) > 0:
        fd.write(str(sum(fold_rank) / len(fold_rank)) + ';')

    # fp.write("Mean average precision (macro, micro): ")
    if len(fold_prec) > 0:
        fd.write(str(sum(fold_prec) / len(fold_prec)) + ';')
        fd.write(str(sum(fold_precm) / len(fold_precm)) + ';')

    # fp.write("Micro-average AUC: ")
    if len(fold_auc) > 0:
        fd.write(str(sum(fold_auc) / len(fold_auc)) + ';')

    d = classification_report(y_test, y_score, digits=20, output_dict=True)
    # es un dict de dicts -> en micro avg -> recall y f1-score
    #       -> en macro avg -> recall y f1-score
    for kv in d.items():
        if kv[0] == 'micro avg':
            fd.write(str(kv[1].get('f1-score')) + ';')
            fd.write(str(kv[1].get('recall')) + ';')
        if kv[0] == 'macro avg':
            fd.write(str(kv[1].get('f1-score')) + ';')
            fd.write(str(kv[1].get('recall')) + ';')
    fd.write('\n')

    self.emit(SIGNAL('infoclassif'), u"☑ Terminado")
    self.emit(SIGNAL('infoclassif'), u">>>Consulte el fichero resultante en: " + str(outpname)+'\n')


def getargs(self, metodo):

    if metodo == 'MlKNN':
        # Requiere parametro k
        k, _ = QInputDialog.getInt(self, u"Parámetro k - MlKNN", u"Introduzca el valor de k para MlKNN: ", 5,
                                   min=1, max=1000)
        result = str(k)

    else:
        crits = ('gini', 'entropy')
        kern = ('rbf', 'linear', 'poly', 'sigmoid', 'precomputed')
        if metodo == 'kNN':
            # Req n_neighbors
            n_neighbors, _ = QInputDialog.getInt(self, u"Parámetro n_neighbors - kNN", u"Introduzca el valor "
                                                                                        u"de n_neighbors para "
                                                                                        u"kNN: ", 5, min=1, max=1000)
            result = str(n_neighbors)

        if metodo == 'Random Forests':
            # Req n_estimators, criterion
            n_estimators, _ = QInputDialog.getInt(self, u"Parámetro n_estimators - Random Forests",
                                                  u"Introduzca el valor de n_estimators para Random Forests: ", 10,
                                                  min=1, max=1000)
            criterion_rf, _ = QInputDialog.getItem(self, u"Parámetro criterion - Random Forests",
                                                   u"Seleccione el valor de criterion para Random Forests: ",
                                                   crits, 0, False)
            result = (str(n_estimators) + ', ' + str(criterion_rf))

        if metodo == 'SVM':
            # Req kernel, gamma, c (gamma no tiene si kernel es linear)
            kernel, _ = QInputDialog.getItem(self, u"Parámetro kernel - SVM",
                                             u"Seleccione el valor de kernel para SVM: ",
                                             kern, 0, False)
            C, _ = QInputDialog.getDouble(self, u"Parámetro C - SVM", u"Seleccione el valor de C para SVM", 1.0,
                                          decimals=5, min=0.00001, max=10.0)

            if kernel == 'poly' or kernel == 'sigmoid' or kernel == 'rbf':
                # Necesitamos gamma
                gamma, _ = QInputDialog.getDouble(self, u'Parámetro gamma - SVM',
                                                  u"Introduzca el valor de gamma para SVM: ", 0.0, decimals=5,
                                                  min=0.00001, max=10.0)
                if gamma == 0.0:
                    gamma = 'scale'
            else:
                gamma = 'scale'

            result = str(kernel) + ', ' + str(C) + ', ' + str(gamma)

        if metodo == 'Decision Tree':
            # criterion
            criterion_dt, _ = QInputDialog.getItem(self, u"Parámetro criterion - Decision Tree",
                                                   u"Seleccione el valor de criterion para Decision Tree: ",
                                                   crits, 0, False)
            result = str(criterion_dt)
    return result


# TODO: no usada \/
def classify(self, classif, nfolds, fname):
    call = []
    for i in range(0, len(classif)):
        # main_c = classif[i][0]
        # base = classif[i][1]

        if classif[i][0] == 'Binary Relevance':
            if classif[i][1] == 'kNN':
                call.append(BinaryRelevance(classifier=KNeighborsClassifier(n_neighbors=n_neighbors),
                                       require_dense=[False, True]))
            if classif[i][1] == 'Random Forests':
                call.append(BinaryRelevance(classifier=RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=str(criterion_rf)), require_dense=[False, True]))
            if classif[i][1] == 'SVM':
                call.append(BinaryRelevance(classifier=SVC(C=C, kernel=str(kernel), gamma='scale', probability=True),
                                       require_dense=[False, True]))
            if classif[i][1] == 'Decision Tree':
                call.append(BinaryRelevance(classifier=DecisionTreeClassifier(criterion=str(criterion_dt)),
                                       require_dense=[False, True]))

        if classif[i][0] == 'Label Powerset':
            if classif[i][1] == 'kNN':
                call.append(LabelPowerset(classifier=KNeighborsClassifier(n_neighbors=n_neighbors),
                                       require_dense=[False, True]))
            if classif[i][1] == 'Random Forests':
                call.append(LabelPowerset(classifier=RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=str(criterion_rf)), require_dense=[False, True]))
            if classif[i][1] == 'SVM':
                call.append(LabelPowerset(classifier=SVC(C=C, kernel=str(kernel), gamma='scale', probability=True),
                                       require_dense=[False, True]))
            if classif[i][1] == 'Decision Tree':
                call.append(LabelPowerset(classifier=DecisionTreeClassifier(criterion=str(criterion_dt)),
                                       require_dense=[False, True]))

        if classif[i][0] == 'Classifier Chain':
            if classif[i][1] == 'kNN':
                call.append(ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=n_neighbors),
                                       require_dense=[False, True]))
            if classif[i][1] == 'Random Forests':
                call.append(ClassifierChain(classifier=RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=str(criterion_rf)), require_dense=[False, True]))
            if classif[i][1] == 'SVM':
                call.append(ClassifierChain(classifier=SVC(C=C, kernel=str(kernel), gamma='scale', probability=True),
                                       require_dense=[False, True]))
            if classif[i][1] == 'Decision Tree':
                call.append(ClassifierChain(classifier=DecisionTreeClassifier(criterion=str(criterion_dt)),
                                       require_dense=[False, True]))

        if classif[i][0] == 'MlKNN':
            call.append(MLkNN(k=k))

    global ejecs

    if nfolds > 0:
        for cl in call:
            if self.checkmt1.isChecked():
                ejecs += 1
            if self.checkmt2.isChecked():
                ejecs += 1
            if self.checkmt3.isChecked():
                ejecs += 1

        for cl in call:
            # print c
            if not os.path.exists(str(dir) + '/csv/'):
                os.makedirs(str(dir) + '/csv/')

            if self.checkmt1.isChecked():  # Se ha marcado realizar clasificac con estratif iterativo

                make_classif(self, nfolds, fname, cl, 'iterative')

            if self.checkmt2.isChecked():  # Se ha marcado realizar clasificac con estratif aleatorio

                make_classif(self, nfolds, fname, cl, 'random')

            # if os.path.isfile(str(fname)[:str(fname).rfind('.')] + '_iterative3.train'):
            if self.checkmt3.isChecked():  # Se ha marcado realizar clasificac con estratif de labelset

                make_classif(self, nfolds, fname, cl, 'labelset')
    else:
        self.emit(SIGNAL('infoclassif'), 'ERROR1')


def configure(self, classif, nfolds, fname):
    global k
    global n_neighbors
    global criterion_dt
    global n_estimators
    global criterion_rf
    global kernel
    global C
    global gamma

    print classif

    self.txt.append(u"Configurando parámetros para los algoritmos de clasificación seleccionados...")
    for i in range(0, len(classif)):
        self.child3.set("filename", str(fname))
        self.child3.set("methods", str(len(classif)))
        self.child3.set("nfolds", str(self.nfls.text()))
        if self.checkmt1.isChecked():
            print self.checkmt1.text()
            self.child3.set("stratif1", str(self.checkmt1.text()))
        else:
            self.child3.set("stratif1", '')
        if self.checkmt2.isChecked():
            print self.checkmt2.text()
            self.child3.set("stratif2", str(self.checkmt2.text()))
        else:
            self.child3.set("stratif2", '')
        if self.checkmt3.isChecked():
            print self.checkmt3.text()
            self.child3.set("stratif3", str(self.checkmt3.text()))
        else:
            self.child3.set("stratif3", '')

        self.child3.set("method"+str(i), str(classif[i][0]))
        self.child3.set("cbase"+str(i), str(classif[i][1]))
        if classif[i][0] == 'MlKNN':
            # Requiere parametro k
            k, _ = QInputDialog.getInt(self, u"Parámetro k - MlKNN", u"Introduzca el valor de k para MlKNN: ", 5)
            self.child3.set("k" + str(i), str(k))
            # print k
        else:
            crits = ('gini', 'entropy')
            kern = ('rbf', 'linear', 'poly', 'sigmoid', 'precomputed')
            if classif[i][1] == 'kNN':
                # Req n_neighbors
                n_neighbors, _ = QInputDialog.getInt(self, u"Parámetro n_neighbors - kNN", u"Introduzca el valor "
                                                                                            u"de n_neighbors para "
                                                                                            u"kNN: ", 5)
                self.child3.set("n_neighbors" + str(i), str(n_neighbors))
                # print n_neighbors
            if classif[i][1] == 'Random Forests':
                # Req n_estimators, criterion
                n_estimators, _ = QInputDialog.getInt(self, u"Parámetro n_estimators - Random Forests",
                                                   u"Introduzca el valor de n_estimators para Random Forests: ", 10)
                criterion_rf, _ = QInputDialog.getItem(self, u"Parámetro criterion - Random Forests",
                                                    u"Seleccione el valor de criterion para Random Forests: ",
                                                    crits, 0, False)
                self.child3.set("n_estimators" + str(i), str(n_estimators))
                self.child3.set("criterion_rf" + str(i), str(criterion_rf))
                # print n_estimators
                # print criterion_rf
            if classif[i][1] == 'SVM':
                # Req kernel, gamma, c (gamma no tiene si kernel es linear)
                kernel, _ = QInputDialog.getItem(self, u"Parámetro kernel - SVM",
                                                 u"Seleccione el valor de kernel para SVM: ",
                                                 kern, 0, False)
                C, _ = QInputDialog.getDouble(self, u"Parámetro C - SVM", u"Seleccione el valor de C para SVM", 1.0)

                self.child3.set("kernel" + str(i), str(kernel))
                self.child3.set("c" + str(i), str(C))

                if kernel == 'poly' or kernel == 'sigmoid' or kernel == 'rbf':
                    # Necesitamos gamma
                    gamma, _ = QInputDialog.getDouble(self, u'Parámetro gamma - SVM',
                                                      u"Introduzca el valor de gamma para SVM: ", 0.0, decimals=5)
                    if gamma == 0.0:
                        gamma = 'scale'
                    self.child3.set("gamma" + str(i), str(gamma))

            if classif[i][1] == 'Decision Tree':
                # criterion
                criterion_dt, _ = QInputDialog.getItem(self, u"Parámetro criterion - Decision Tree",
                                                       u"Seleccione el valor de criterion para Decision Tree: ",
                                                       crits, 0, False)
                self.child3.set("criterion_dt" + str(i), str(criterion_dt))
