# coding=utf-8
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys, os, numpy as np, time

from sklearn.exceptions import UndefinedMetricWarning
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lxml import etree
import model_dataset as md, model_folds as mf, model_clasificacion as mc
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import warnings

gk = 5
gn_neighbors = 5
gn_estimators = 10
gcriterion_rf = 'gini'
gcriterion_dt = 'gini'
gkernel = 'rbf'
ggamma = 0.0
gC = 1.0


def load(self):
    dlg = QFileDialog()
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setFilter('XML files (*.xml)')
    ok = 0
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        tree = etree.parse(str(filenames[0]))
        print tree.getroot().tag
        if tree.getroot().tag == 'experimento':
            for element in tree.iter():
                if element.tag == 'dataset':
                    ok = 1
            if not ok == 1:
                QMessageBox.about(self, "Aviso", u"El archivo XML cargado no es válido, "
                                                  "revise el contenido o vuelva a generarlo")
                return ''
        else:
            QMessageBox.about(self, "Aviso", u"El archivo XML cargado no es válido, "
                                             "revise el contenido o vuelva a generarlo")
            return ''
        if tree.findall('.//dataset'):
            for name in tree.findall('.//dataset'):
                if name.get('filename') and name.get('op1') and name.get('op2') and name.get('op3') and name.get('op4'):
                    ok = 1
                else:
                    QMessageBox.about(self, "Aviso", u"El archivo XML cargado no es válido, "
                                                     "revise el contenido o vuelva a generarlo")
                    return ''
        if tree.findall('.//estratificado'):
            ok = 1
        if tree.findall('.//metodo'):
            for nam in tree.findall('.//metodo'):
                if nam.get('cbase') and nam.get('method') and nam.get('args'):
                    ok = 1
                else:
                    QMessageBox.about(self, "Aviso", u"El archivo XML cargado no es válido, "
                                                      "revise el contenido o vuelva a generarlo")
                    return ''
        if ok == 1:
            return filenames[0]
        else:
            return ''
    return ''


def savedir(self):
    file = str(QFileDialog.getExistingDirectory(self, 'Select Directory'))
    return file


def execute_dset(self, fname, dir):

    fileds = []
    op1 = []
    op2 = []
    op3 = []
    op4 = []
    dsactive = 0

    if not str(fname) == '':
        tree = etree.parse(str(fname))
        txt = etree.tostring(tree.getroot(), pretty_print=True)

        if tree.findall('dataset'):
            time.sleep(0.2)
            self.emit(SIGNAL('textoinf'), 'Head')
        for name in tree.findall('dataset'):
            if name.get('filename'):
                self.emit(SIGNAL('textoinf'), '>Dataset: ' + str(name.get('filename')))
                fileds.append(str(name.get('filename')))
            if name.get('op1') == 'True':

                self.emit(SIGNAL('textoinf'), 'op1')
                op1.append(str(name.get('op1')))
                dsactive = 1
            else:
                op1.append(str(False))
            if name.get('op2') == 'True':

                self.emit(SIGNAL('textoinf'), 'op2')
                dsactive = 1
                op2.append(str(name.get('op2')))
            else:
                op2.append(str(False))
            if name.get('op3') == 'True':

                self.emit(SIGNAL('textoinf'), 'op3')
                dsactive = 1
                op3.append(str(name.get('op3')))
            else:
                op3.append(str(False))
            if name.get('op4') == 'True':

                self.emit(SIGNAL('textoinf'), 'op4')
                dsactive = 1
                op4.append(str(name.get('op4')))
            else:
                op4.append(str(False))

    if dsactive == 1:
        tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        doc = SimpleDocTemplate(dir + '/' + str(tstamp)+"_plot-report.pdf")
        styles = getSampleStyleSheet()
        parts = []
        save = str(dir) + '/' + 'tmp/'
        print save
        print fileds[0]
        active = 0
        for i in range(0, len(fileds)):
            cnt = 0
            if not op1[i] == 'False':
                cnt += 1
            if not op2[i] == 'False':
                cnt += 1
            if not op3[i] == 'False':
                cnt += 1
            if not op4[i] == 'False':
                cnt += 1

            self.emit(SIGNAL('prog1'), 0)
            tmp = os.path.basename(str(fileds[i]))
            dat = os.path.splitext(tmp)[0]

            if not cnt == 0:
                parts.append(Spacer(1, 0.2 * inch))
                p = Paragraph("Dataset: " + dat, styles["Title"])
                parts.append(Spacer(1, 0.2 * inch))
                parts.append(p)

                self.emit(SIGNAL('textoinf'), 'INFO1')
                prog = 100 / cnt
                if not op1[i] == 'False':

                    active = 1
                    df = md.convert(self, fileds[i], dir)
                    self.emit(SIGNAL('prog1'), prog/2)

                    report, insts = md.cardinality(self, df)
                    self.emit(SIGNAL('prog1'), prog)

                    parts.append(Spacer(1, 0.2 * inch))
                    p = Paragraph(u"Medidas: ", styles["Heading2"])
                    parts.append(p)

                    if os.path.isfile(report):
                        try:
                            fo = open(report, 'r')
                        except:
                            print "Error al leer medidas del dataset"

                        headers = []
                        data = []

                        for o in range(0, 6):
                            try:
                                inline = fo.readline()
                                part = inline.split(': ')
                            except:
                                pass # Derivado del error de arriba

                            if part[0] == 'Instances':
                                headers.append(u"Nº de instancias")
                                data.append(str(part[1]).strip('\n'))

                            if part[0] == 'Features':
                                headers.append(u"Nº de características")
                                data.append(str(part[1]).strip('\n'))

                            if part[0] == 'Labels':
                                headers.append(u"Nº de etiquetas")
                                data.append(str(part[1]).strip('\n'))

                            if part[0] == 'Cardinality':
                                headers.append(u"Cardinalidad")
                                data.append(str(part[1]).strip('\n'))

                            if part[0] == 'Density':
                                headers.append(u"Densidad de etiquetas")
                                data.append(str(part[1]).strip('\n'))

                            if part[0] == 'Distinct':
                                headers.append(u"Nº de combinaciones de etiquetas distintas")
                                data.append(str(part[1]).strip('\n'))

                        parts.append(Spacer(1, 0.4 * inch))
                        tmp = []
                        for l in range(0, 6):
                            if len(headers)>0 and len(data)>0:
                                tmp.append(headers[l:l+1] + data[l:l+1])

                        print tmp

                        t = Table(tmp, rowHeights=(10*mm, 10*mm, 10*mm, 10*mm, 10*mm, 10*mm), colWidths=(70*mm, 60*mm),
                                  style=[('GRID',(0,0),(-1,-1),0.5,colors.black), ('BACKGROUND',(0,0),(0,-1), colors.silver),
                                        ('ALIGN',(0,0),(-1,-1),'CENTER'), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                        ])
                        parts.append(t)
                        parts.append(Spacer(1, 0.2 * inch))

                        parts.append(PageBreak())

                if not op2[i] == 'False':
                    if not os.path.exists(save):
                        os.makedirs(save)
                    prog += prog
                    try:
                        md.coov(self, fileds[i], dir, True, False)
                        self.emit(SIGNAL('prog1'), prog)

                        if os.path.isfile(save + dat + '_corrlabls.png'):
                            parts.append(Spacer(1, 0.2 * inch))
                            p = Paragraph(u"Gráfica de correlación entre etiquetas", styles["Heading2"])
                            parts.append(p)

                            parts.append(Image(save + dat + '_corrlabls.png'))
                            parts.append(PageBreak())
                    except:
                        print u'Error en la generación de gráfica de correlación de etiquetas'

                if not op3[i] == 'False':
                    if not os.path.exists(save):
                        os.makedirs(save)
                    prog += prog
                    try:
                        md.coov(self, fileds[i], dir, False, True)
                        self.emit(SIGNAL('prog1'), prog)

                        if os.path.isfile(save + dat + '_corrordered.png'):
                            parts.append(Spacer(1, 0.2 * inch))
                            p = Paragraph(u"Gráfica de distribucion de la correlación", styles["Heading2"])
                            parts.append(p)

                            parts.append(Image(save + dat + '_corrordered.png'))
                            parts.append(PageBreak())
                    except:
                        print u'Error en la generación de gráfica de distribución de la correlación'

                if not op4[i] == 'False':
                    if not os.path.exists(save):
                        os.makedirs(save)
                    prog += prog
                    try:
                        md.labfrecplot(insts, fileds[i], dir)
                        self.emit(SIGNAL('prog1'), prog)

                        if os.path.isfile(save + dat + '_freclbs.png'):

                            parts.append(Spacer(1, 0.2 * inch))
                            p = Paragraph(u"Gráfica de frecuencia de las etiquetas", styles["Heading2"])
                            parts.append(p)

                            parts.append(Image(save + dat + '_freclbs.png', width=640, height=480, kind='proportional'))
                            parts.append(PageBreak())
                    except:
                        print u'Error en la generación de gráfica de frecuencias de las etiquetas'

        if os.path.exists(save) or active == 1:
            try:
                doc.build(parts)
                print 'docum generado'
                self.emit(SIGNAL('textoinf'), '\nInforme PDF generado, puede consultarlo en: ' + dir + '/' +
                          str(tstamp)+"_plot-report.pdf\n")
            except:
                print 'Error al generar el documento PDF'

    self.emit(SIGNAL('prog1'), 100)
    self.emit(SIGNAL('finished'))


def execute_folds(self, fname, dir):

    tree = etree.parse(str(fname))
    filef = []
    nfls = []
    m1 = []
    m2 = []
    m3 = []
    factive = 0

    if not str(fname) == '':
        print 'estoy entrando... foolds'
        if tree.findall('.//estratificado'):
            time.sleep(1)
            self.emit(SIGNAL('add(QString)'), 'Head')

            for name in tree.iter():
                if name.get('filename'):
                    self.emit(SIGNAL('add(QString)'), '\n>Dataset: ' + str(name.get('filename')))
                    filef.append(str(name.get('filename')))
                    time.sleep(1)
                if name.get('nfolds'):
                    self.emit(SIGNAL('add(QString)'), '>nfolds: ' + str(name.get('nfolds')))
                    factive = 1
                    nfls.append(int(name.get('nfolds')))
                if name.get('m1'):
                    self.emit(SIGNAL('add(QString)'), str(name.get('m1')))
                    factive = 1
                    m1.append(str(name.get('m1')))
                if name.get('m2'):
                    self.emit(SIGNAL('add(QString)'), str(name.get('m2')))
                    factive = 1
                    m2.append(str(name.get('m2')))
                if name.get('m3'):
                    self.emit(SIGNAL('add(QString)'), str(name.get('m3')))
                    factive = 1
                    m3.append(str(name.get('m3')))
        else:
            print 'terminado - paso no activado'
            self.emit(SIGNAL('update(int)'), 100)

    if factive == 1:
        while True:
            if not len(m1) == len(filef):
                m1.append('0')
            else:
                break

        while True:
            if not len(m2) == len(filef):
                m2.append('0')
            else:
                break

        while True:
            if not len(m3) == len(filef):
                m3.append('0')
            else:
                break

        for i in range(0, len(filef)):
            suffix = os.path.basename(str(filef[i]))
            suffix = os.path.splitext(suffix)[0]
            self.emit(SIGNAL('add(QString)'), '\n>Dataset: ' + str(suffix))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=Warning)
                if not m1[i] == '0':
                    self.emit(SIGNAL('add(int)'), 0)
                    try:
                        mf.gen_folds(self, nfls[i], filef[i], dir, True, False, False)
                    except:
                        print "Se ha producido un error en la generación de los folds, método: "+ str(m1[i])

                if not m2[i] == '0':
                    self.emit(SIGNAL('add(int)'), 0)
                    try:
                        mf.gen_folds(self, nfls[i], filef[i], dir, False, True, False)
                    except:
                        print "Se ha producido un error en la generación de los folds, método: "+ str(m2[i])
                if not m3[i] == '0':
                    self.emit(SIGNAL('add(int)'), 0)
                    try:
                        mf.gen_folds(self, nfls[i], filef[i], dir, False, False, True)
                    except:
                        print "Se ha producido un error en la generación de los folds, método: "+ str(m3[i])

    self.emit(SIGNAL('end'))


def execute_class(self, fname, dir):

    clasactive = 0
    fclass = []
    stratif = []
    meths = []
    csbase = []
    nflds = []
    k = []
    n_neighbors = []
    n_estimators = []
    criterion_rf = []
    kernel = []
    C = []
    gamma = []
    criterion_dt = []
    call = []
    parms = []
    estratificado = ''
    meth_count = []
    aux = 0

    if not str(fname) == '':
        tree = etree.parse(str(fname))

        print 'estoy entrando... classif'
        if tree.findall('.//metodo'):
            time.sleep(1)
            self.emit(SIGNAL('infoclassif'), 'Head')
            for name in tree.iter():
                if name.get('filename'):
                    self.emit(SIGNAL('infoclassif'), '\n>Dataset: ' + str(name.get('filename')))
                    fclass.append(str(name.get('filename')))
                    clasactive = 1
                    time.sleep(1)
                    if aux >= 1:
                        meth_count.append(aux)
                        aux=0
                if name.get('nfolds'):
                    self.emit(SIGNAL('infoclassif'), '>nfolds: ' + str(name.get('nfolds')))
                    nflds.append(int(name.get('nfolds')))
                if name.get('m1'):
                    self.emit(SIGNAL('infoclassif'), '\n>Estratificado: ' + str(name.get('m1')))
                    estratificado = 'Iterative'
                if name.get('m2'):
                    self.emit(SIGNAL('infoclassif'), '\n>Estratificado: ' + str(name.get('m2')))
                    estratificado = 'Random'
                if name.get('m3'):
                    self.emit(SIGNAL('infoclassif'), '\n>Estratificado: ' + str(name.get('m3')))
                    estratificado = 'Labelset'
                if name.tag == 'metodo':
                    if name.get('method'):
                        self.emit(SIGNAL('infoclassif'), '>>Algoritmo: ' + str(name.get('method')))
                        meths.append(str(name.get('method')))
                        stratif.append(str(estratificado))
                        clasactive = 1
                        aux+=1
                    if name.get('cbase'):
                        if not name.get('cbase') == '-':
                            self.emit(SIGNAL('infoclassif'), u'>>Clasificador base: ' + str(name.get('cbase')))
                        csbase.append(str(name.get('cbase')))
                        clasactive = 1
                    if name.get('args'):
                        print csbase[(len(csbase) - 1)]
                        if csbase[(-1)] == 'kNN':
                            n = name.get('args')[2:-2]
                            self.emit(SIGNAL('infoclassif'), u'    n_neighbors: ' + str(n))
                            n_neighbors.append(str(n))
                        else:
                            n_neighbors.append('0')
                        if csbase[(-1)] == 'SVM':
                            st = name.get('args')[2:-2].split(',')
                            print st[0], st[1], st[2]
                            self.emit(SIGNAL('infoclassif'), u'    kernel: ' + str(st[0]))
                            self.emit(SIGNAL('infoclassif'), u'    C: ' + str(st[1]))
                            self.emit(SIGNAL('infoclassif'), u'    gamma: ' + str(st[2]))
                            kernel.append(str(st[0]))
                            C.append(str(st[1]))
                            gamma.append(str(st[2]).strip())
                        else:
                            kernel.append('0')
                            C.append('0')
                            gamma.append('0')
                        if meths[(len(meths) - 1)] == 'MlKNN':
                            n = name.get('args')[2:-2]
                            self.emit(SIGNAL('infoclassif'), u'    k: ' + str(n))
                            k.append(str(n))
                        else:
                            k.append('0')
                        if csbase[(-1)] == 'Random Forests':
                            st = name.get('args')[2:-2].split(',')
                            print st[0], st[1]
                            self.emit(SIGNAL('infoclassif'), u'    n_estimators: ' + str(st[0]))
                            self.emit(SIGNAL('infoclassif'), u'    criterion_rf: ' + str(st[1]))
                            n_estimators.append(str(st[0]))
                            criterion_rf.append(str(st[1]).strip())
                        else:
                            n_estimators.append('0')
                            criterion_rf.append('0')
                        if csbase[(-1)] == 'Decision Tree':
                            n = name.get('args')[2:-2]
                            self.emit(SIGNAL('infoclassif'), u'    criterion_dt: ' + str(n))
                            criterion_dt.append(str(n))
                        else:
                            criterion_dt.append('0')
        else:
            print 'terminado - paso no activado'
            self.emit(SIGNAL('progress'), 100)
            self.emit(SIGNAL('infoclassif'), u'\nTerminado\n')

    meth_count.append(aux)  # Le pasamos el contador del ultimo dset ya q no vuelve a entrar para ponerlo
    aux=0
    self.emit(SIGNAL('log'))
    print '---------------------'
    print fclass
    print meth_count
    print stratif
    print csbase
    print meths
    print '----------------------'

    if clasactive == 1:
        print 'estoy para clasificar.....'
        for i in range(0, len(fclass)):
            call = []
            parms = []
            for j in range(0, len(meths)):
                if meths[j] == 'Binary Relevance':
                    if csbase[j] == 'kNN':
                        print n_neighbors[j]
                        call.append(BinaryRelevance(classifier=KNeighborsClassifier(n_neighbors=int(n_neighbors[j])), require_dense=[
                         False, True]))
                        parms.append('n_neighbors= ' + str(n_neighbors[j]))
                    if csbase[j] == 'Random Forests':
                        print n_estimators[j]
                        call.append(BinaryRelevance(classifier=RandomForestClassifier(n_estimators=int(n_estimators[j]), criterion=str(criterion_rf[j])), require_dense=[False, True]))
                        parms.append('n_estimators= ' + str(n_estimators[j]) + ', criterion= ' + str(criterion_rf[j]))
                    if csbase[j] == 'SVM':
                        print gamma[j], C[j], kernel[j]
                        try:
                            gamma[j] = float(gamma[j])
                        except ValueError:
                            print 'no se puede conv a float'  # Será 'scale'
                        print gamma[j]
                        call.append(BinaryRelevance(classifier=SVC(C=float(C[j]), kernel=str(kernel[j]), gamma=gamma[j], probability=True), require_dense=[
                         False, True]))
                        parms.append('C= ' + str(C[j]) + ', kernel= ' + str(kernel[j]) + ', gamma= ' + str(gamma[j]))
                    if csbase[j] == 'Decision Tree':
                        print criterion_dt[j]
                        call.append(BinaryRelevance(classifier=DecisionTreeClassifier(criterion=str(criterion_dt[j])), require_dense=[
                         False, True]))
                        parms.append('criterion = ' + str(criterion_dt[j]))
                if meths[j] == 'Label Powerset':
                    if csbase[j] == 'kNN':
                        print n_neighbors[j]
                        call.append(LabelPowerset(classifier=KNeighborsClassifier(n_neighbors=int(n_neighbors[j])), require_dense=[
                         False, True]))
                        parms.append('n_neighbors= ' + str(n_neighbors[j]))
                    if csbase[j] == 'Random Forests':
                        print n_estimators[j]
                        call.append(LabelPowerset(classifier=RandomForestClassifier(n_estimators=int(n_estimators[j]), criterion=str(criterion_rf[j])), require_dense=[
                         False, True]))
                        parms.append('n_estimators= ' + str(n_estimators[j]) + ', criterion= ' + str(criterion_rf[j]))
                    if csbase[j] == 'SVM':
                        print gamma[j], C[j], kernel[j]
                        call.append(LabelPowerset(classifier=SVC(C=float(C[j]), kernel=str(kernel[j]), gamma=gamma[j], probability=True), require_dense=[
                         False, True]))
                        parms.append('C= ' + str(C[j]) + ', kernel= ' + str(kernel[j]) + ', gamma= ' + str(gamma[j]))
                    if csbase[j] == 'Decision Tree':
                        print criterion_dt[j]
                        call.append(LabelPowerset(classifier=DecisionTreeClassifier(criterion=str(criterion_dt[j])), require_dense=[
                         False, True]))
                        parms.append('criterion = ' + str(criterion_dt[j]))
                if meths[j] == 'Classifier Chain':
                    if csbase[j] == 'kNN':
                        print n_neighbors[j]
                        call.append(ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=int(n_neighbors[j])), require_dense=[
                         False, True]))
                        parms.append('n_neighbors= ' + str(n_neighbors[j]))
                    if csbase[j] == 'Random Forests':
                        print n_estimators[j]
                        call.append(ClassifierChain(classifier=RandomForestClassifier(n_estimators=int(n_estimators[j]), criterion=str(criterion_rf[j])), require_dense=[
                         False, True]))
                        parms.append('n_estimators= ' + str(n_estimators[j]) + ', criterion= ' + str(criterion_rf[j]))
                    if csbase[j] == 'SVM':
                        print gamma[j], C[j], kernel[j]
                        call.append(ClassifierChain(classifier=SVC(C=float(C[j]), kernel=str(kernel[j]), gamma=gamma[j], probability=True), require_dense=[
                         False, True]))
                        parms.append('C= ' + str(C[j]) + ', kernel= ' + str(kernel[j]) + ', gamma= ' + str(gamma[j]))
                    if csbase[j] == 'Decision Tree':
                        print criterion_dt[j]
                        call.append(ClassifierChain(classifier=DecisionTreeClassifier(criterion=str(criterion_dt[j])), require_dense=[
                         False, True]))
                        parms.append('criterion = ' + str(criterion_dt[j]))
                if meths[j] == 'MlKNN':
                    call.append(MLkNN(k=int(k[j])))
                    parms.append('k= ' + str(k[j]))

            #print call
            #print parms
        rn = 0
        for i in range(0, len(fclass)):
            if nflds[i] > 0:
                suffix = os.path.basename(str(fclass[i]))
                suffix = os.path.splitext(suffix)[0]
                self.emit(SIGNAL('infoclassif'), u'\n>Dataset: ' + str(suffix))

                for z in range(rn, rn+meth_count[i]):
                    print '>' + str(call[z]).split('(')[0]
                    print '>' + str(stratif[z])
                    print z
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                            mc.make_classif(self, nflds[i], fclass[i], call[z], parms[z], stratif[z], dir)
                    except Exception as e:
                        print "Se ha producido un error durante la clasificación, método: " + str(call[z]).split('(')[0]
                        print e
                        self.emit(SIGNAL('infoclassif'), u"☒ Se ha producido un error durante "
                                                         u"la clasificación, método: " + str(call[z]).split('(')[0]+
                                  '\nError: ' + str(e))
                rn = meth_count[i]

            else:
                self.emit(SIGNAL('infoclassif'), 'ERROR1')

        self.emit(SIGNAL('infoclassif'), u'\nTerminado\n')
