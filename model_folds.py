# coding=utf-8
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
import os
import numpy as np
import arff
from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from skmultilearn.model_selection.measures import folds_label_combination_pairs_without_evidence
from skmultilearn.model_selection.measures import example_distribution
from skmultilearn.model_selection.measures import label_combination_distribution
from skmultilearn.model_selection.measures import folds_without_evidence_for_at_least_one_label_combination


def get_filename(self):
    dlg = QFileDialog()
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setFilter("MEKA dataset files (*.arff)")

    if dlg.exec_():
        filenames = dlg.selectedFiles()
        f = open(filenames[0], 'r')
        self.le.setText(filenames[0])
        with f:
            line = f.readline()
            flag = False
            for i in line.split():
                if flag is True:
                    break
                if (i == "-C") or (i == "-c"):
                    flag = True

            if not flag:
                f.close()
                self.emit(SIGNAL('add(QString)'), 'ERROR1')
                # self.contents.setText("Error en el formato de la cabecera del fichero de dataset")
                return
                # sys.exit("Wrong format for the dataset header")
            # else:
            # save filename for posterior execution
        return filenames[0]
    else:
        if self.le.text() is not '':
            return self.le.text()
        else:
            return ''


class Transfomer:
    def transform_to_multiclass(self, y):
        self.label_count = y.shape[1]
        self.unique_combinations = {}
        self.reverse_combinations = []
        self.last_id = 0
        train_vector = []

        for labels_applied in y:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = self.last_id
                self.reverse_combinations.append(labels_applied)
                self.last_id += 1

            train_vector.append(self.unique_combinations[label_string])
        return train_vector


def stratified_folds(n_splits, y):
    t = Transfomer()
    kf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
    folds = [x[1] for x in list(kf.split(np.zeros(y.shape[0]), t.transform_to_multiclass(y)))]
    return folds


def aux_fold(suffix, kfold, X, train_index, X_train, X_test, y_train, y_test, f, sparse, number):

    folds = train_index
    desired_number = (X.shape[0] * (f - 1)) / f
    # Training file

    fp = open(suffix + str(kfold) + '.train', 'w')

    # Save header
    if sparse:
        fp.write('[MULTILABEL, SPARSE]\n')
    else:
        fp.write('[MULTILABEL, DENSE]\n')
    fp.write('$ %d\n' % len(X_train))  # Number of objects
    fp.write('$ %d\n' % len(X_train[0]))  # Number of attributes
    fp.write('$ %d\n' % abs(int(number)))  # Number of labels

    # Data
    for i in range(0, len(X_train)):

        if sparse:
            for j in range(0, len(X_train[i])):
                if X_train[i][j] != '0.0':
                    fp.write(str(j + 1) + ':' + str(X_train[i][j]) + ' ')
                if X_train[i][j] == 'YES':
                    fp.write('1' + ' ')
        else:
            for j in range(0, len(X_train[i])):
                if X_train[i][j] == 'YES':
                    fp.write('1' + ' ')
                elif X_train[i][j] == 'NO':
                    fp.write('0' + ' ')
                else:
                    fp.write(str(X_train[i][j]) + ' ')

        fp.write('[ ')
        for j in range(0, len(y_train[i])):
            if y_train[i][j] == '0.0':
                aux = str(y_train[i][j]).split('.')[0]
                fp.write(str(int(aux)) + ' ')
            else:
                fp.write(str(int(y_train[i][j])) + ' ')
        fp.write(']\n')
    fp.close()

    # Testing file
    fp = open(suffix + str(kfold) + '.test', 'w')

    # Save header
    if sparse:
        fp.write('[MULTILABEL, SPARSE]\n')
    else:
        fp.write('[MULTILABEL, DENSE]\n')
    fp.write('$ %d\n' % len(X_test))
    fp.write('$ %d\n' % len(X_test[0]))
    fp.write('$ %d\n' % abs(int(number)))

    # Data
    for i in range(0, len(X_test)):
        if sparse:
            for j in range(0, len(X_test[i])):
                if X_test[i][j] != '0.0':
                    fp.write(str(j + 1) + ':' + str(X_test[i][j]) + ' ')
                if X_test[i][j] == 'YES':
                    fp.write('1' + ' ')
        else:
            for j in range(0, len(X_test[i])):
                if X_test[i][j] == 'YES':
                    fp.write('1' + ' ')
                elif X_test[i][j] == 'NO':
                    fp.write('0' + ' ')
                else:
                    fp.write(str(X_test[i][j]) + ' ')

        fp.write('[ ')
        for j in range(0, len(y_test[i])):
            if y_test[i][j] == '0.0':
                aux = str(y_test[i][j]).split('.')[0]
                fp.write(str(int(aux)) + ' ')
            else:
                fp.write(str(int(y_test[i][j])) + ' ')
        fp.write(']\n')
    fp.close()
    kfold += 1

    return kfold, folds, desired_number


def exec_fold(self, suffix, kf, X, y, f, sparse, number):
    # print f
    kfold = 0
    folds = []
    desired_number = []
    self.completed = 0

    if not isinstance(kf, list):
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            kfold, fd, d_n = aux_fold(suffix, kfold, X, train_index, X_train, X_test, y_train, y_test, f, sparse, number)
            self.completed += 100/f
            self.emit(SIGNAL("update(int)"), self.completed)
            # self.progress.setValue(self.completed)
            folds.append(fd)
            desired_number.append(d_n)
    else:
        for test_index in kf:
            train_index = [x for x in range(X.shape[0]) if x not in test_index]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            kfold, fd, d_n = aux_fold(suffix, kfold, X, train_index, X_train, X_test, y_train, y_test, f, sparse, number)
            self.completed += 100/f
            self.emit(SIGNAL("update(int)"), self.completed)
            # self.progress.setValue(self.completed)
            folds.append(fd)
            desired_number.append(d_n)

    fp = open(suffix + '.measures', 'w')
    FLZ = folds_label_combination_pairs_without_evidence(y, folds, 1)
    FZ = folds_without_evidence_for_at_least_one_label_combination(y, folds, 1)
    LD = label_combination_distribution(y, folds, 1)
    ED = example_distribution(folds, desired_number)
    fp.write("Label distribution: ")
    fp.write(str(LD) + '\n')
    fp.write("Example distribution: ")
    fp.write(str(ED) + '\n')
    fp.write("Number of fold-label pairs with 0 positive examples, FLZ: ")
    fp.write(str(FLZ) + '\n')
    fp.write("Number of folds that contain at least 1 label with 0 positive examples, FZ: ")
    fp.write(str(FZ) + '\n')
    fp.close()


def gen_folds(self, nfolds, fname, dir, mk1, mk2, mk3):

    if not str(fname).lower().endswith('.arff'):
        self.emit(SIGNAL('add(QString)'), 'ERROR2')
        # sys.exit(u"Formato del dataset no válido, por favor use .arff datasets")
    else:
        dataset = arff.load(open(fname), 'rb')
        data = np.array(dataset['data'])

        fi = open(fname, 'r')
        line = fi.readline()

        flag = False
        for i in line.split():
            if flag is True:
                number = i
                break
            if (i == "-C") or (i == "-c"):
                flag = True

        if not flag:
            fi.close()
            self.emit(SIGNAL('add(QString)'), 'ERROR2')
        if number[-1:] == "'":
            number = number[:-1]
        fi.close()
        nominalIndexArray = []
        nominals = []
        aux = 0
        # from attributes we can get if its nominal
        if int(number) > 0:
            for x in dataset['attributes'][int(number):]:
                if (len(x[1]) > 2) and (x[1] != ("NUMERIC" or "REAL" or "INTEGER" or "STRING")):
                    nominalIndexArray.append(aux)
                    nominals.append(x[1])
                aux += 1
        else:
            for x in dataset['attributes'][:int(number)]:
                if (len(x[1]) > 2) and (x[1] != ("NUMERIC" or "REAL" or "INTEGER" or "STRING")):
                    nominalIndexArray.append(aux)
                    nominals.append(x[1])
                aux += 1

        # Split the data in X and Y
        if int(number) > 0:
            y = data[:, 0:int(number)].astype(int)
            x = data[:, int(number):]
        else:
            y = data[:, int(number):].astype(int)
            x = data[:, :int(number)]

        if len(nominalIndexArray) > 0:
            # Change the nominal attributes to numeric ones
            index = 0
            X = []
            for k in x:
                numericVector = []
                for i in range(0, len(nominalIndexArray)):
                    # Ahora tenemos que crear el vector que le vamos a poner al final de cada
                    checkIfMissing = False
                    for aux in nominals[i]:
                        if aux == k[nominalIndexArray[i]]:
                            # Add 1 to the array
                            checkIfMissing = True
                            numericVector.append(1)
                        else:
                            # Add 0 to the array
                            checkIfMissing = True
                            numericVector.append(0)
                    if checkIfMissing is False:
                        # Add another 1 to the array
                        numericVector.append(1)
                    else:
                        numericVector.append(0)
                auxVector = np.append(k, [numericVector])
                # Substract that nominals values
                auxVector = np.delete(auxVector, nominalIndexArray)
                X.append(auxVector)

            X = np.array(X)
        else:
            X = np.array(x)

        # Sparse or dense?
        sizeofdouble = 8
        sizeofint = 4
        sizeofptr = 8
        dense_size = len(X) * len(X[0]) * sizeofdouble + len(X) * sizeofptr
        # nz = np.count_nonzero(X)
        nz = 0
        for i in range(0, len(X)):
            for j in range(0, len(X[0])):
                if X[i][j] != '0.0':
                    nz += 1
        sparse_size = nz * (sizeofdouble + sizeofint) + 2 * len(X) * sizeofptr + len(X) * sizeofint

        sparse = False if sparse_size >= dense_size else True

        # Ahora incluimos los diferentes métodos para estratificación
        suffix = os.path.basename(str(fname))
        suffix = os.path.splitext(suffix)[0]

        # Crea una carpeta con los k folds del experim y cada estratif

        if nfolds > 0:

            if mk1:
                #tmp = self.checkmt1.text()
                # self.flabel3.setText(u"Generando los folds, método: "+self.checkmt1.text())
                if not os.path.exists(str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_iterative'):
                    os.makedirs(str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_iterative')
                    kf1 = IterativeStratification(n_splits=int(nfolds), order=1)
                    self.emit(SIGNAL('add(QString)'), u">Generando particiones con estratificado iterativo...")
                    exec_fold(self, str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_iterative/' + str(suffix), kf1, X, y, int(nfolds), sparse, number)
                    self.emit(SIGNAL('add(QString)'), u"Hecho!")
                else:
                    self.emit(SIGNAL('add(QString)'), u">Las particiones iterativas solicitadas ya existen")

            if mk2:
                #tmp = self.checkmt2.text()
                # self.flabel3.setText(u"Generando los folds, método: " + self.checkmt2.text())
                if not os.path.exists(str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_random'):
                    os.makedirs(str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_random')
                    kf2 = KFold(n_splits=int(nfolds), shuffle=True)
                    self.emit(SIGNAL('add(QString)'), ">Generando particiones con estratificado aleatorio...")
                    exec_fold(self, str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_random/' + str(suffix), kf2, X, y, int(nfolds), sparse, number)
                    self.emit(SIGNAL('add(QString)'), u"Hecho!")
                else:
                    self.emit(SIGNAL('add(QString)'), u">Las particiones aleatorias solicitadas ya existen")

            if mk3:
                #tmp = self.checkmt3.text()
                # self.flabel3.setText(u"Generando los folds, método: " + self.checkmt3.text())
                if not os.path.exists(str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_labelset'):
                    os.makedirs(str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_labelset')
                    kf3 = stratified_folds(int(nfolds), y)
                    # Para labelset hay q modificar un poco la func
                    self.emit(SIGNAL('add(QString)'), u">Generando particiones con estratificado labelset...")
                    exec_fold(self, str(dir) + '/' + str(suffix) + '/' + str(suffix) + str(nfolds) + '_labelset/' + str(suffix), kf3, X, y, int(nfolds), sparse, number)
                    self.emit(SIGNAL('add(QString)'), u"Hecho!")
                else:
                    self.emit(SIGNAL('add(QString)'), u">Las particiones labelset solicitadas ya existen")

            if (mk1 is False) and (mk2 is False) and (mk3 is False):
                self.emit(SIGNAL('add(QString)'), 'Info1')
                # self.flabel3.setText(u"No se ha seleccionado ningún método de estratificación")
                # self.flabel3.show()
            else:
                self.emit(SIGNAL('add(QString)'), 'Info2')
                # self.flabel3.setText(u"Particiones generadas correctamente, compruebe los .train y .test generados")
        else:
            self.emit(SIGNAL('add(QString)'), 'Info3')
            # self.flabel3.setText(u"No se pueden generar 0 folds, introduzca un valor válido")
            # self.flabel3.show()

    self.emit(SIGNAL('add(QString)'), str(nfolds))
    if mk1:
        self.emit(SIGNAL('add(QString)'), '1')
    if mk2:
        self.emit(SIGNAL('add(QString)'), '2')
    if mk3:
        self.emit(SIGNAL('add(QString)'), '3')