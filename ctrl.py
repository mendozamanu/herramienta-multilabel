import model_clasificacion as mc
import model_dataset as md
import model_folds as mf
import model_xml as mx


def eventload(a):
    fname = md.open_file(a)

    return fname


def measures(a, fname):
    if fname is not '':
        dsc = md.convert(a, fname)
        md.cardinality(a, dsc)


def plots(a, fname):
    md.coov(a, fname)


def eventName(a):
    fname = mf.get_filename(a)

    return fname


def genFolds(a, f):
    mf.gen_folds(a, f)


def getargs(a, method):
    args = mc.getargs(a, method)
    return args


def conf_class(a, classif, nf, fname):
    mc.configure(a, classif, nf, fname)


def exec_class(a, classif, nf, fname):
    mc.classify(a, classif, nf, fname)


def loadxml(a):
    fname = mx.load(a)

    return fname


def getsaveDir(a):
    fname = mx.savedir(a)

    return fname


def execute_dset(a, fname, dir):
    mx.execute_dset(a, fname, dir)


def execute_folds(a, fname, dir):
    mx.execute_folds(a, fname, dir)


def execute_class(a, fname, dir):
    mx.execute_class(a, fname, dir)
