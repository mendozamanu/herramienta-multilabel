# coding=utf-8
import os
from collections import Counter

import arff
import matplotlib.pyplot as plt
import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *

temp = 1


def open_file(self):
    dlg = QFileDialog()

    self.contents.setReadOnly(True)
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setFilter("MEKA dataset files (*.arff)")

    if dlg.exec_():
        filenames = dlg.selectedFiles()
        f = open(filenames[0], 'r')
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
                QMessageBox.warning(self, "Error", "Error en el formato de la cabecera del fichero de dataset")
                self.contents.append("Error en el formato de la cabecera del fichero de dataset")
                return ''

        return filenames[0]
    else:
        return ''


# Operaciones sobre los datasets
def convert(self, fname, dir):
    # Read arff file
    self.emit(SIGNAL('textoinf'), "Analizando el dataset cargado...")

    if not str(fname).lower().endswith('.arff'):
        self.emit(SIGNAL('textoinf'), 'ERROR1')
        return
    else:
        dataset = arff.load(open(fname, 'rb'))
        data = np.array(dataset['data'])

        # We have to get the number of clases from the raw file
        fil = open(fname, "r")
        line = fil.readline()

        flag = False
        for i in line.split():
            if flag is True:
                number = i
                break
            if (i == "-C") or (i == "-c"):
                flag = True

        if not flag:
            fil.close()
            self.emit(SIGNAL('textoinf'), 'ERROR2')

        if number[-1:] == "'":
            number = number[:-1]
        fil.close()
        # Now we have the number stored, knowing that positive means the first attributes and negative the last ones

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

        nz = 0
        for i in range(0, len(X)):
            for j in range(0, len(X[0])):
                if X[i][j] != '0.0':
                    nz += 1
        sparse_size = nz * (sizeofdouble + sizeofint) + 2 * len(X) * sizeofptr + len(X) * sizeofint

        sparse = False if sparse_size >= dense_size else True

        # Use input file as output suffix if no other given
        # suffix = str(fname)[:str(fname).rfind('.')]
        # Complete file

        suffix = os.path.basename(str(fname))
        suffix = os.path.splitext(suffix)[0]

        if not os.path.exists(dir + '/' + suffix + '/'):
            os.makedirs(dir + '/' + suffix + '/')

        fp = open(dir + '/' + suffix + '/' + suffix + '.complete', 'w')

        # Save header
        if sparse:
            fp.write('[MULTILABEL, SPARSE]\n')
        else:
            fp.write('[MULTILABEL, DENSE]\n')
        fp.write('$ %d\n' % len(X))  # Number of objects
        fp.write('$ %d\n' % len(X[0]))  # Number of attributes
        fp.write('$ %d\n' % abs(int(number)))  # Number of labels

        # Data
        for i in range(0, len(X)):
            if sparse:
                for j in range(0, len(X[i])):
                    if X[i][j] != '0.0':
                        fp.write(str(j + 1) + ':' + str(X[i][j]) + ' ')
                    if X[i][j] == 'YES':
                        fp.write('1' + ' ')
            else:
                for j in range(0, len(X[i])):
                    if X[i][j] == 'YES':
                        fp.write('1' + ' ')
                    elif X[i][j] == 'NO':
                        fp.write('0' + ' ')
                    else:
                        fp.write(str(X[i][j]) + ' ')

            fp.write('[ ')
            for j in range(0, len(y[i])):
                if y[i][j] == '0.0':
                    aux = str(y[i][j]).split('.')[0]
                    fp.write(str(int(aux)) + ' ')
                else:
                    fp.write(str(int(y[i][j])) + ' ')
            fp.write(']\n')

        # Save header
        fp.close()

        return dir + '/' + suffix + '/' + suffix + '.complete'


def cardinality(self, df):
    tmp = os.path.basename(str(df))
    dat = os.path.splitext(tmp)[0]

    if not str(df).lower().endswith('.complete'):
        self.emit(SIGNAL('textoinf'), 'ERROR1')
    else:
        datafile = open(df)
        l0 = datafile.readline()
        l0 = l0.split()
        sparse = l0[1]
        if sparse[:-1] == 'SPARSE':
            sparse = True  # The file is in sparse mode
        else:
            sparse = False

        l1 = datafile.readline()
        l2 = datafile.readline()
        l3 = datafile.readline()
        instances = int(l1.split()[1])

        features = int(l2.split()[1])

        labels = int(l3.split()[1])

        l4 = datafile.readline()

        avg = 0
        tmp = 0
        dist = []
        insts = np.zeros(labels, dtype=int)

        nwdfname = str(df)[:str(df).rfind('.')] + "_measures.report"

        self.emit(SIGNAL('textoinf'), "\nMedidas del dataset " + str(dat) + ':')
        fp = open(nwdfname, 'w')
        fp.write("Instances: " + str(instances) + '\n')
        self.emit(SIGNAL('textoinf'), ">Instances: " + str(instances))

        fp.write("Features: " + str(features) + '\n')
        self.emit(SIGNAL('textoinf'), ">Features: " + str(features))

        fp.write("Labels: " + str(labels) + '\n')
        self.emit(SIGNAL('textoinf'), ">Labels: " + str(labels))

        while l4 != "":
            if l4 == ' ':
                pass
            else:
                if not sparse:
                    label = map(int, l4.strip().split()[features + 1:features + 1 + labels])
                    # To remove the '[' ']' from the labels extraction
                    dist.append(''.join(map(str, l4.strip().split()[features + 1:features + 1 + labels])))
                    # en dist tenemos todas las combinacs, luego hacemos el set
                    tmp = sum(label)
                    insts[tmp] += 1
                    avg += sum(label)

                else:
                    # Sparse . find '[' and start reading until ']'
                    label = map(int,
                                l4.strip().split()[l4.strip().split().index('[') + 1:l4.strip().split().index(']')])
                    dist.append(''.join(
                        map(str, l4.strip().split()[l4.strip().split().index('[') + 1:l4.strip().split().index(']')])))
                    tmp = sum(label)
                    insts[tmp] += 1
                    avg += sum(label)

            l4 = datafile.readline()

        un_combs = set(dist)

        fp.write("Cardinality: ")
        card = avg / (instances * 1.0)
        fp.write(str(card) + '\n')
        self.emit(SIGNAL('textoinf'), ">Cardinality: " + str(card))

        fp.write("Density: ")
        fp.write(str(card / (labels * 1.0)) + '\n')
        self.emit(SIGNAL('textoinf'), ">Density: " + str(card / (labels * 1.0)))

        fp.write("Distinct: ")
        fp.write(str(len(un_combs)) + '\n')
        self.emit(SIGNAL('textoinf'), ">Distinct: " + str(len(un_combs)))

        fp.write("Num of instances per label-count (0, 1, 2, ... nlabel)\n")
        for i in range(0, insts.shape[0]):
            fp.write(str(i) + ' ' + str(insts[i]) + '\n')

        fp.write("Labels frequency: \n")

        aux = np.zeros(shape=(labels, 2))

        for i in range(0, labels):
            aux[i] = (sum(int(row[i]) for row in dist), i + 1)

        aux = aux[(-aux[:, 0]).argsort()]

        for s in aux:
            fp.write(str(int(s[1])) + ' ' + str(int(s[0])) + '\n')

        countr = Counter(dist)
        fp.write("Label combinations frequency: \n")
        for value, count in countr.most_common():
            fp.write(str(int(value, 2)) + ' ' + str(count) + '\n')

        datafile.close()
        self.emit(SIGNAL('textoinf'), "\nFichero report creado: " + '\n>>' + str(nwdfname) +
                  " guardado correctamente")
        fp.close()

        return nwdfname, insts


def labfrecplot(insts, name, dir):
    # insts[] is the vector to plot
    tmp = os.path.basename(str(name))
    dat = os.path.splitext(tmp)[0]
    save = str(dir) + '/' + 'tmp/'

    flbs = np.trim_zeros(insts, 'b')
    objects = range(0, flbs.shape[0])
    y_pos = np.arange(len(objects))
    plt.figure(figsize=(15, 9), num='Label frecuency')
    plt.bar(y_pos, flbs, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Instances')
    plt.xlabel('Num of active labels')
    plt.title(dat + ': ' + 'Label frecuency')
    plt.margins(x=0.01)
    for i, j in zip(flbs, y_pos):
        plt.annotate(str(flbs[j]), xy=(j, i + (np.max(flbs) * 0.01)), horizontalalignment='center')

    plt.savefig(save + dat + '_freclbs.png')
    plt.close()


# TODO: mejorar la visualizacion de la grafica que se genera en cardinality...........................................
def label_correlation(y, s):
    """Correlation between labels in a label matrix
    Parameters
    ----------
    y : array-like (n_labels, n_samples)
        Label matrix
    s : float
        Smoothness parameter
    Returns
    -------
    L : array-like (n_labels, n_labels)
        Label correlation matrix
    """
    L = np.zeros(shape=[y.shape[0], y.shape[0]])

    for i in range(0, y.shape[0]):
        yi = sum(y[i, :])
        for j in range(0, y.shape[0]):
            coincidence = 0
            for k in range(0, y.shape[1]):
                if (int(y[i, k]) == int(1)) and (int(y[j, k]) == int(1)):
                    coincidence += 1
            L[i, j] = (coincidence + s) / (yi + 2 * s)

    return L


def coov(self, name, dir, plt1, plt2):
    tmp = os.path.basename(str(name))
    dat = os.path.splitext(tmp)[0]

    save = str(dir) + '/' + 'tmp/'

    if not str(name).lower().endswith('.arff'):
        self.emit(SIGNAL("textoinf"), 'ERROR1')
    else:
        dataset = arff.load(open(name, 'rb'))
        data = np.array(dataset['data'])

        # We have to get the number of clases from the raw file
        file = open(name, "r")
        line = file.readline()

        flag = False
        for i in line.split():
            if flag is True:
                number = i
                break
            if (i == "-C") or (i == "-c"):
                flag = True

        if not flag:
            file.close()
            self.emit(SIGNAL("textoinf"), 'ERROR2')

        if number[-1:] == "'":
            number = number[:-1]
        file.close()
        # Now we have the number stored, knowing that positive means the first attributes and negative the last ones

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

        L = label_correlation(y.transpose(), 0.19)

        global temp

        if temp:
            fp = open(save + dat + '_correlacion.report', 'w')
            self.emit(SIGNAL("textoinf"), "\nFichero report creado: " + '\n' + '>>' + str(save) + str(dat) +
                      '_correlacion.report' + " guardado correctamente" + '\n')

            L.tofile(fp, sep=" ", format='%s')
            fp.close()
            temp = 0

        tri_indx = L[np.tril_indices(L.shape[0], -1)]

        # listas para almac los indices de los intervalos
        l0 = []
        l1 = []
        l2 = []
        l3 = []
        l4 = []
        l5 = []
        l6 = []
        l7 = []
        l8 = []
        l9 = []
        # Contador para el vector de correlacs
        a = 0

        n = L.shape[0]
        cor = np.zeros((n * n) - n)

        for i in range(0, L.shape[0]):
            for j in range(L.shape[1]):
                if j != i:  # Cogemos todos los valores menos la diagonal ya que importa el orden
                    # (etq 1-2 no tiene misma correlac q etiq 2-1)
                    if L[i][j] <= 0.1:  # intervalo 0-0.1
                        l0.append([i, j])
                    if L[i][j] > 0.1 and L[i][j] <= 0.2:  # intervalo 0.1-0.2
                        l1.append([i, j])
                    if L[i][j] > 0.2 and L[i][j] <= 0.3:  # intervalo 0.2-0.3
                        l2.append([i, j])
                    if L[i][j] > 0.3 and L[i][j] <= 0.4:  # intervalo 0.3-0.4
                        l3.append([i, j])
                    if L[i][j] > 0.4 and L[i][j] <= 0.5:  # intervalo 0.4-0.5
                        l4.append([i, j])
                    if L[i][j] > 0.5 and L[i][j] <= 0.6:  # intervalo 0.5-0.6
                        l5.append([i, j])
                    if L[i][j] > 0.6 and L[i][j] <= 0.7:  # intervalo 0.6-0.7
                        l6.append([i, j])
                    if L[i][j] > 0.7 and L[i][j] <= 0.8:  # intervalo 0.7-0.8
                        l7.append([i, j])
                    if L[i][j] > 0.8 and L[i][j] <= 0.9:  # intervalo 0.8-0.9
                        l8.append([i, j])
                    if L[i][j] > 0.9 and L[i][j] <= 1:  # intervalo 0.9-1.0
                        l9.append([i, j])

                    cor[a] = L[i][j]
                    a += 1

        cor = -np.sort(-cor)

        if plt1:
            labelscorrel = [len(l0), len(l1), len(l2), len(l3), len(l4), len(l5), len(l6), len(l7), len(l8), len(l9)]
            objects = (
            '0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1')
            y_pos = np.arange(len(objects))
            plt.figure(figsize=(7, 5), num='Correlation between labels')
            plt.bar(y_pos, labelscorrel, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.xlabel("Correlation interval")
            plt.ylabel('Number of label pairs')
            plt.title(str(dat) + ': ' + 'Correlation between labels')
            for i, j in zip(labelscorrel, y_pos):
                plt.annotate(str(labelscorrel[j]), xy=(j, i + (np.max(labelscorrel) * 0.01)),
                             horizontalalignment='center')

            if os.path.exists(save):
                plt.savefig(save + dat + '_corrlabls.png')
            plt.close()

        if plt2:
            plt.figure(num='Correlation distribution')
            plt.plot(cor)
            plt.axis([0, cor.shape[0], 0, 1.1])
            plt.xlabel('Distinct label pairs')
            plt.ylabel('Correlation')
            plt.title(str(dat) + ': ' + 'Correlation distribution')

            plt.annotate(str("{0:.3f}".format(cor[0])), xy=(0, cor[0] + 0.02))
            plt.annotate(str("{0:.3f}".format(cor[-1])), xy=((n * n) - n - 2, cor[-1] + 0.02))

            if os.path.exists(save):
                plt.savefig(save + dat + '_corrordered.png')
            plt.close()
