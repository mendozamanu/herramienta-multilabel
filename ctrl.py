# coding=utf-8

import model_clasificacion as mc
import model_dataset as md
import model_folds as mf
import model_xml as mx


# Método para enlazar con la apertura del dataset de model_dataset
def eventload(a):
    fname = md.open_file(a)

    return fname


# Método para enlazar con la obtención de las medidas del dataset mediante model_dataset
def measures(a, fname):
    if fname is not '':
        dsc = md.convert(a, fname)
        md.cardinality(a, dsc)


# Método para enlazar con la generación de las gráficas del dataset en model_dataset
def plots(a, fname):
    md.coov(a, fname)


# Método para enlazar con la generación de las particiones en model_folds
def genFolds(a, f):
    mf.gen_folds(a, f)


# Método para enlazar con la obtención de los parámetros de los clasificadores en model_classif
def getargs(a, method):
    args = mc.getargs(a, method)
    return args


# Método para enlazar con la ejecución de la clasificación en model_classif
def exec_class(a, classif, nf, fname):
    mc.classify(a, classif, nf, fname)


# Método para enlazar con la apertura del fichero xml del experimento en model_xml
def loadxml(a):
    fname = mx.load(a)

    return fname


# Método para enlazar con la obtención del directorio de trabajo para el experimento en model_xml
def getsaveDir(a):
    fname = mx.savedir(a)

    return fname


# Métodos para enlazar con la ejecución de las operaciones durante el experimento en model_xml
def execute_dset(a, fname, dir):
    mx.execute_dset(a, fname, dir)


def execute_folds(a, fname, dir):
    mx.execute_folds(a, fname, dir)


def execute_class(a, fname, dir):
    mx.execute_class(a, fname, dir)
