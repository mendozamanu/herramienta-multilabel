# coding=utf-8

import functools
import os.path
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import sklearn.metrics
import sklearn.metrics.base
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from sklearn.metrics import classification_report


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


# Modificación de la implementación de Scikit-learn para evitar error de división por 0, que ocasionaba
# mean avg precision nan en muchos folds y datasets.
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


def make_classif(self, nfolds, fname, cl, parm, stratif, dir):
    fold_accuracy = []
    fold_hamming = []
    fold_prec = []
    fold_precm = []

    fold_auc = []
    fold_cover = []
    fold_rank = []

    if not os.path.exists(str(dir) + '/csv/'):
        os.makedirs(str(dir) + '/csv/')

    if stratif == 'Random':
        outpname = dir + '/csv/' + str(cl).split('(')[0] + '_random.csv'

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

    tmp = u"Ejecutando clasificación " + str(cl).split('(')[0] + " con el estratificado: " + stratif

    self.emit(SIGNAL('infoclassif'), tmp)

    prog = 0

    for i in range(0, nfolds):

        prog += 100 / nfolds
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
            try:
                y_prob = cl.predict_proba(X_test.todense())
            except:
                self.emit(SIGNAL('logcns_c'), "Error en predict_proba")
            # -----------------------------------------#
            # Coverage\n",
            try:
                c = sklearn.metrics.coverage_error(y_test, y_prob.toarray(), sample_weight=None)
                fold_cover.append(c)
            except:
                self.emit(SIGNAL('logcns_c'), "Error en coverage")

            # -----------------------------------------#
            # Ranking loss\n",
            try:
                rl = sklearn.metrics.label_ranking_loss(y_test, y_prob.toarray(), sample_weight=None)
                fold_rank.append(rl)
            except:
                self.emit(SIGNAL('logcns_c'), "Error en ranking loss")
            # -----------------------------------------#
            # Mean average precision
            try:
                m = average_precision_score(y_test, y_prob.toarray(), average='macro', pos_label=1, sample_weight=None)
                fold_prec.append(m)

                m2 = average_precision_score(y_test, y_prob.toarray(), average='micro', pos_label=1, sample_weight=None)
                fold_precm.append(m2)
            except:
                self.emit(SIGNAL('logcns_c'), "Error en average precision score")
            # -----------------------------------------#
            # Micro-average AUC
            try:
                rmi = sklearn.metrics.roc_auc_score(y_test, y_prob.toarray(), average='micro', sample_weight=None,
                                                    max_fpr=None)
                fold_auc.append(rmi)
            except:
                self.emit(SIGNAL('logcns_c'), "Error en roc auc micro")

            # -----------------------------------------#
            # Medidas: sklearn.metrics...(true,predict,..)
        try:
            acc = sklearn.metrics.accuracy_score(y_test, y_score)
            fold_accuracy.append(acc)
        except:
            self.emit(SIGNAL('logcns_c'), "Error en accuracy score")
        # -----------------------------------------#
        try:
            hl = sklearn.metrics.hamming_loss(y_test, y_score)
            fold_hamming.append(hl)
        except:
            self.emit(SIGNAL('logcns_c'), "Error en hamming loss")

    fd = open(outpname, 'a')
    tmp = os.path.basename(str(fname))
    s = os.path.splitext(tmp)[0]

    tstamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fd.write(str(tstamp) + ';')  # marca de tiempo

    cbase = str(cl).split('(')[1]
    cbase = cbase.split('=')[1]

    if not str(cl).split('(')[0] == 'MLkNN':
        fd.write(cbase + ';')  # Clasificador base
    else:
        fd.write('MLkNN' + ';')

    fd.write(parm + ';')  # Params de los metodos

    fd.write(str(s) + ';')  # El nombre del dataset

    fd.write(str(sum(fold_accuracy) / len(fold_accuracy)) + ';')
    fd.write(str(sum(fold_hamming) / len(fold_hamming)) + ';')

    if len(fold_cover) > 0:
        fd.write(str(sum(fold_cover) / len(fold_cover)) + ';')

    if len(fold_rank) > 0:
        fd.write(str(sum(fold_rank) / len(fold_rank)) + ';')

    if len(fold_prec) > 0:
        fd.write(str(sum(fold_prec) / len(fold_prec)) + ';')
        fd.write(str(sum(fold_precm) / len(fold_precm)) + ';')

    if len(fold_auc) > 0:
        fd.write(str(sum(fold_auc) / len(fold_auc)) + ';')

    try:
        d = classification_report(y_test, y_score, digits=20, output_dict=True)
    except:
        self.emit(SIGNAL('logcns_c'), "Error al generar classification report")
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
    self.emit(SIGNAL('infoclassif'), u">>>Consulte el fichero resultante en: " + str(outpname) + '\n')


def getargs(self, metodo):
    if metodo == 'MlKNN':
        # Requiere parámetro k
        k, ok = QInputDialog.getInt(self, u"Parámetro k - MlKNN", u"Introduzca el valor de k para MlKNN: ", 5,
                                    min=1, max=1000)
        if ok:
            result = str(k)
        else:
            return

    else:
        crits = ('gini', 'entropy')
        kern = ('rbf', 'linear', 'poly', 'sigmoid', 'precomputed')
        if metodo == 'kNN':
            # Req n_neighbors
            n_neighbors, ok = QInputDialog.getInt(self, u"Parámetro n_neighbors - kNN", u"Introduzca el valor "
                                                                                        u"de n_neighbors para "
                                                                                        u"kNN: ", 5, min=1, max=1000)
            if ok:
                result = str(n_neighbors)
            else:
                return

        if metodo == 'Random Forests':
            # Req n_estimators, criterion
            n_estimators, ok = QInputDialog.getInt(self, u"Parámetro n_estimators - Random Forests",
                                                   u"Introduzca el valor de n_estimators para Random Forests: ", 10,
                                                   min=1, max=1000)
            if not ok:
                return
            criterion_rf, ok2 = QInputDialog.getItem(self, u"Parámetro criterion - Random Forests",
                                                     u"Seleccione el valor de criterion para Random Forests: ",
                                                     crits, 0, False)
            if ok and ok2:
                result = (str(n_estimators) + ', ' + str(criterion_rf))
            else:
                return

        if metodo == 'SVM':
            # Req kernel, gamma, c (gamma no tiene si kernel es linear)
            kernel, ok = QInputDialog.getItem(self, u"Parámetro kernel - SVM",
                                              u"Seleccione el valor de kernel para SVM: ",
                                              kern, 0, False)
            if not ok:
                return
            C, ok2 = QInputDialog.getDouble(self, u"Parámetro C - SVM", u"Seleccione el valor de C para SVM", 1.0,
                                            decimals=5, min=0.00001, max=10.0)

            if not ok2:
                return
            if kernel == 'poly' or kernel == 'sigmoid' or kernel == 'rbf':
                # Necesitamos gamma
                gamma, ok3 = QInputDialog.getDouble(self, u'Parámetro gamma - SVM',
                                                    u"Introduzca el valor de gamma para SVM: ", 0.0, decimals=5,
                                                    min=0.00001, max=10.0)
                if gamma == 0.0:
                    gamma = 'scale'
            else:
                gamma = 'scale'

            if ok and ok2 or ok3:
                result = str(kernel) + ', ' + str(C) + ', ' + str(gamma)
            else:
                return

        if metodo == 'Decision Tree':
            # criterion
            criterion_dt, ok = QInputDialog.getItem(self, u"Parámetro criterion - Decision Tree",
                                                    u"Seleccione el valor de criterion para Decision Tree: ",
                                                    crits, 0, False)
            if ok:
                result = str(criterion_dt)
            else:
                return
    return result
