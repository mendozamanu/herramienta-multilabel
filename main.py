# coding=utf-8
import sys
import os
from PyQt4.QtGui import *
from PyQt4.QtCore import *

import ctrl
from lxml import etree
from collections import defaultdict

# TODO https://stackoverflow.com/questions/3605680/creating-a-simple-xml-file-using-python#3605831 #usamos lxml

# TODO
#  BUGS: 1. al volver para atras cuando abres subventanas (cargar dset, o pedir medidas, se abren 3 veces)
#        2. cuando termina la ejecucion salen 2 errors/warnings sobre QObject::startTumer: Qtimer solo s puede usar en
#        hilos lanzados por QThread.
#        3. Object::disconnect: Unexpected null parameter en el pc de casa
#          ???

root = etree.Element("experimento")
filename = ['',]
file = ''
text = str('')
nfolds = 0
mk1 = False
mk2 = False
mk3 = False
classif = []
dir = './'
xmlname = ''
startxml = False
datasets = []  # Contendra elems de la clase dset, estratif y metodo

proxy = QIdentityProxyModel()


#TODO: no req name al inicializ clase
class dataset:

    def __init__(self):
        self.name = ''
        self.op1 = False
        self.op2 = False
        self.op3 = False
        self.op4 = False
        self.estratif = []  # Contendra objs de la clase estratificado

    def add_name(self, name):
        self.name = name

    def add_op1(self, op):
        self.op1 = op

    def add_op2(self, op):
        self.op2 = op

    def add_op3(self, op):
        self.op3 = op

    def add_op4(self, op):
        self.op4 = op

    def set_estratif(self, strat):
        self.estratif.append(strat)


class stratif:

    def __init__(self, id):
        self.id = id
        self.nfolds = 0
        self.methods = []

    def add_folds(self, fls):
        self.nfolds = fls

    def set_methods(self, met):
        self.methods.append(met)


class metodo:

    def __init__(self):
        self.cbase = ''
        self.method = ''
        self.args = []

    def addcbase(self, cbase):
        self.cbase = cbase

    def addmethod(self, met):
        self.method = met

    def addargs(self, args):
        self.args.append(args)


class GenericThread(QThread):
    def __init__(self, function, *args, **kwargs):
        QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.function(*self.args, **self.kwargs)
        return


class XmlW(QMainWindow):
    def loadXmlW(self):

        self.setCentralWidget(QWidget(self))

        self.threadPool = []

        self.load = QLineEdit() # Para cargar el xml a ejecutar

        #self.load.setText(str(xmlname))
        self.load.setReadOnly(True)
        self.btn1 = QPushButton("Cargar fichero XML")
        self.workingDir = QLineEdit()  # Para cargar el xml a ejecutar
        self.workingDir.setReadOnly(True)
        self.btn2 = QPushButton("Seleccionar directorio de trabajo")
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.progress = QProgressBar()
        self.btn3 = QPushButton("Ejecutar")
        self.btn4 = QPushButton("Inicio")

        self.grid = QGridLayout()
        self.grid.addWidget(self.load, 0, 0)
        self.grid.addWidget(self.btn1, 0, 1)
        self.grid.addWidget(self.workingDir, 1, 0)
        self.grid.addWidget(self.btn2, 1, 1)
        self.grid.addWidget(self.info, 2, 0)
        self.grid.addWidget(self.progress, 3, 0)
        self.grid.addWidget(self.btn3, 2, 1)
        self.grid.addWidget(self.btn4, 3, 1)

        self.centralWidget().setLayout(self.grid)

    def __init__(self, parent=None):
        super(XmlW, self).__init__(parent)
        # layout.addWidget(self.btn1)
        self.setMinimumSize(600, 300)  # Tamanio minimo de ventana
        self.resize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadXmlW()

    def getxmlfile(self):
        global xmlname
        xmlname = ctrl.loadxml(self)
        self.load.setText(str(xmlname))
        self.info.clear()

    def getWorkingDir(self):
        global dir
        dir = ctrl.getsaveDir(self)
        self.workingDir.setText(str(dir))

    def adds(self, info):  # Dataset operations

        if str(info) == 'op1':
            self.info.append('>Operaciones: ' + 'Medidas')
        if str(info) == 'op2':
            self.info.append('>Operaciones: ' + u"Gráfica distribución de la correlación")
        if str(info) == 'op3':
            self.info.append('>Operaciones: ' + u"Gráfica correlación entre etiquetas")
        if str(info) == 'op4':
            self.info.append('>Operaciones: ' + u"Gráfica de frecuencia de las etiquetas")

        if str(info).startswith('\n'):
            self.info.append(info)

        if str(info).startswith('>'):
            self.info.append(info)

        if str(info) == 'Head':
            self.info.append(u"1. Análisis del dataset\nInformación cargada: ")
        #if str(info) == 'INFO1':
            #self.info.append(u"Ejecutando operaciones solicitadas...")
        if str(info) == 'ERROR1':
            QMessageBox.about(self, "Error", "Error en el formato de la cabecera del fichero de dataset")
        if str(info) == 'ERROR2':
            QMessageBox.about(self, "Error", "Error al cargar el dataset, por favor use datasets con formato .arff")

    def add(self, text):  # Folds operations
        # print text
        global nfolds, mk1, mk2, mk3
        if len(text) < 3:
            if int(text) > 3:
                nfolds = int(text)
        if text == 'Head':
            self.info.append(u"\n2. Generar particiones\nInformación cargada:")
        if text == '1':
            mk1 = True
        if text == '2':
            mk2 = True
        if text == '3':
            mk3 = True
        if text == 'ERROR1':
            QMessageBox.about(self, "Error", "Error en el formato de la cabecera del fichero de dataset")
        if text == 'ERROR2':
            QMessageBox.about(self, "Error", "Error al cargar el dataset, por favor use datasets con formato .arff")

        if text == 'Iterative':
            self.info.append(u">Método seleccionado: " + text)

        if text == 'Random':
            self.info.append(u">Método seleccionado: " + text)

        if text == 'Labelset':
            self.info.append(u">Método seleccionado: " + text)

        if text == 'Info1':
            self.info.append(u"No se ha seleccionado ningún método de estratificación")
        if text == 'Info2':
            self.info.append(u"☑Particiones generadas correctamente, compruebe los .train y .test generados")
        if text == 'Info3':
            self.info.append(u"No se pueden generar 0 folds, vuelva a configurar el archivo xml")

        if text == 'Aviso':
            self.info.append(u">Las particiones solicitadas ya existen")

        if str(text).startswith('>') or str(text).startswith('\n') or str(text).endswith('!'):
            self.info.append(text)

    def addc(self, txt):  # Classification operations
        if txt == 'Head':
            self.info.append(u"\n3. Ejecutar clasificación\nInformación cargada:")
        elif txt == 'ERROR1':
            QMessageBox.critical(self, "Error", u"Error al ejecutar la clasificación con 0 folds, "
                                                "configure primero las particiones en el paso anterior")
        elif txt == 'ERROR2':
            QMessageBox.critical(self, "Error",
                                 u"Error al ejecutar la clasificación en el estratificado " +
                                 ", revise las particiones generadas o vuelva a generarlas")
        else:
            self.info.append(txt)

    def prog1(self, progr):
        self.progress.setValue(progr)

    def upd(self, progress):
        self.progress.setValue(progress)

    def nxt(self):
        self.exec_folds()

    def lst(self):
        self.exec_class()

    def execute(self):
        self.info.clear()
        self.exec_ds()
        #self.exec_folds()
        #self.exec_class()

    def exec_ds(self):

        self.threadPool.append(GenericThread(ctrl.execute_dset, self, xmlname, dir))

        # self.threadPool.append(GenericThread(ctrl.plot1, self, op2, fileds, dir))

        self.disconnect(self, SIGNAL("textoinf"), self.adds)
        self.connect(self, SIGNAL("textoinf"), self.adds)
        self.disconnect(self, SIGNAL("finished"), self.nxt)
        self.connect(self, SIGNAL("finished"), self.nxt)
        #self.disconnect(self, SIGNAL("prog1"), self.prog1)
        #self.connect(self, SIGNAL("prog1"), self.prog1)

        self.threadPool[len(self.threadPool) - 1].start()

        # self.threadPool.append(GenericThread(ctrl.plot2, self, op3, fileds, dir))

    def exec_folds(self):

        self.threadPool.append(GenericThread(ctrl.execute_folds, self, xmlname, dir))
        self.disconnect(self, SIGNAL("add(QString)"), self.add)
        self.connect(self, SIGNAL("add(QString)"), self.add)

        self.disconnect(self, SIGNAL("update(int)"), self.upd)
        self.connect(self, SIGNAL("update(int)"), self.upd)

        self.disconnect(self, SIGNAL("end"), self.lst)
        self.connect(self, SIGNAL("end"), self.lst)

        self.threadPool[len(self.threadPool) - 1].start()

    def exec_class(self):

        self.threadPool.append(GenericThread(ctrl.execute_class, self, xmlname, dir))

        self.disconnect(self, SIGNAL("infoclassif"), self.addc)
        self.connect(self, SIGNAL("infoclassif"), self.addc)

        self.disconnect(self, SIGNAL("progress"), self.upd)
        self.connect(self, SIGNAL("progress"), self.upd)

        self.threadPool[len(self.threadPool) - 1].start()


class ClassifW(QMainWindow):

    def loadClass(self):

        self.setCentralWidget(QWidget(self))

        self.threadPool = []
        # self.crono = Crono(self)

        self.label = QLabel(u"Listado de dataset cargados: ")
        self.lst = QListView()
        self.lst2 = QListWidget()

        self.lst.setModel(proxy)

        self.lst.setFixedSize(700, 100)
        self.lst2.setFixedSize(700, 90)

        self.btn2 = QPushButton("Anterior")
        self.btn3 = QPushButton(u"Añadir")
        self.btn4 = QPushButton("Guardar")
        self.btn5 = QPushButton("Siguiente")
        self.methods = QComboBox()
        self.methods.addItems(['Binary Relevance', 'Label Powerset', 'Classifier Chain', 'MlKNN'])
        self.base = QComboBox()
        self.base.addItems(['kNN', 'Random Forests', 'SVM', 'Decision Tree'])
        # self.base.setCurrentIndex(-1)
        self.txt = QLabel()

        self.flabel = QLabel(u"Método: ")
        self.flabel1 = QLabel(u"Clasificador base: ")
        self.flabel2 = QLabel(u"Estratificación: ")
        self.checkmt1 = QCheckBox("Iterative")
        self.checkmt2 = QCheckBox("Random")
        self.checkmt3 = QCheckBox("Labelset")

        # self.progress = QProgressBar(self)

        self.grid = QGridLayout()
        self.grid.setSpacing(2)
        self.grid.addWidget(self.label, 0, 0)
        self.grid.addWidget(self.lst, 1, 0)
        self.grid.addWidget(self.btn2, 4, 1, 1, 14)
        self.grid.addWidget(self.btn3, 5, 1, 1, 14)
        self.grid.addWidget(self.btn4, 6, 1, 1, 14)
        self.grid.addWidget(self.btn5, 9, 1, 1, 14)

        self.grid.addWidget(self.flabel2, 5, 0, Qt.AlignLeft)
        self.grid.addWidget(self.checkmt1, 5, 0, Qt.AlignCenter)
        self.grid.addWidget(self.checkmt2, 6, 0, Qt.AlignCenter)
        self.grid.addWidget(self.checkmt3, 7, 0, Qt.AlignCenter)

        self.grid.addWidget(self.flabel, 3, 0, Qt.AlignLeft)
        self.grid.addWidget(self.flabel1, 3, 0, Qt.AlignCenter)
        self.grid.addWidget(self.methods, 4, 0, Qt.AlignLeft)
        self.grid.addWidget(self.base, 4, 0, Qt.AlignCenter)
        self.grid.addWidget(self.lst2, 8, 0)
        self.grid.addWidget(self.txt, 9, 0)
        # self.grid.addWidget(self.progress, 9, 0)

        #self.txt.setReadOnly(True)
        #self.txt.setText(u"Número de folds realizados anteriormente: " + str(nfolds))
        if nfolds <= 1:
            self.txt.setText(u"Aviso. No se podrá realizar la clasificación si no hay más de 2 particionados realizados")
            self.txt.show()

        self.centralWidget().setLayout(self.grid)

        self.lst.clicked.connect(self.getOperats)

    def __init__(self, parent=None):
        super(ClassifW, self).__init__(parent)
        # layout.addWidget(self.btn1)
        self.setMinimumSize(600, 300)  # Tamanio minimo de ventana
        self.resize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadClass()

    def getOperats(self):
        itms = self.lst.selectedIndexes()
        for SelectedItem in itms:
            if datasets[SelectedItem.row()].estratif is not None:
                self.checkmt1.setEnabled(False)
                self.checkmt2.setEnabled(False)
                self.checkmt3.setEnabled(False)
                self.checkmt1.setChecked(False)
                self.checkmt2.setChecked(False)
                self.checkmt3.setChecked(False)
                for i in range(0, len(datasets[SelectedItem.row()].estratif)):
                    print str(datasets[SelectedItem.row()].estratif[i].id)
                    if str(datasets[SelectedItem.row()].estratif[i].id) == '0':
                        self.checkmt1.setEnabled(True)
                    if str(datasets[SelectedItem.row()].estratif[i].id) == '1':
                        self.checkmt2.setEnabled(True)
                    if str(datasets[SelectedItem.row()].estratif[i].id) == '2':
                        self.checkmt3.setEnabled(True)

    def confmethods(self):
        global classif
        classif.append([str(self.methods.currentText()), str(self.base.currentText())])

        itms = self.lst.selectedIndexes()
        l = []
        for it in itms:
            c = metodo()
            c.addmethod(str(self.methods.currentText()))

            if not self.methods.currentText() == 'MlKNN':
                args = ctrl.getargs(self, str(self.base.currentText()))  # Le psamos los clasif base
                c.addcbase(str(self.base.currentText()))
                # print args
            else:
                args = ctrl.getargs(self, str(self.methods.currentText()))  # Le pasamos mlknn
                c.addcbase('-')

                # print args

            c.addargs(args)

            for i in range(0, len(datasets[it.row()].estratif)):
                if self.checkmt1.isChecked() and str(datasets[it.row()].estratif[i].id) == '0':
                    datasets[it.row()].estratif[i].set_methods(c)

                if self.checkmt2.isChecked() and str(datasets[it.row()].estratif[i].id) == '1':
                    datasets[it.row()].estratif[i].set_methods(c)

                if self.checkmt3.isChecked() and str(datasets[it.row()].estratif[i].id) == '2':
                    datasets[it.row()].estratif[i].set_methods(c)

            # print datasets[it.row()].estratif.methods
            self.lst2.addItem(str(self.methods.currentText()) + ', ' + str(self.base.currentText()) + ' ' + str(args))
            self.txt.setText(u"Añadido correctamente")

    def signalDisable(self):
        if self.methods.currentText() == 'MlKNN':
            self.base.hide()
            self.base.setCurrentIndex(-1)

        else:
            if self.base.currentIndex() == -1:
                self.base.setCurrentIndex(0)
            self.base.show()

    def add(self, text):
        if text == 'ERROR1':
            QMessageBox.critical(self, "Error", u"Error al ejecutar la clasificación con 0 folds, "
                                 "configure primero las particiones en el paso anterior")
        elif text == 'ERROR2':
            QMessageBox.critical(self, "Error",
                                 u"Error al ejecutar la clasificación en el estratificado " +
                                 ", revise las particiones generadas o vuelva a generarlas")
        else:
            self.txt.append(text)

    def getParams(self):
        # Para cada metodo de clasificacion elegido, tras haberlo config -> ejecutar dicha clasificacion
        global classif

        if len(datasets) < 1:
            self.txt.setText(u"Error. No se han añadido datasets")
        else:
            for i in range(0, len(datasets)):
                self.child1 = etree.SubElement(root, "dataset")
                self.child1.set("filename", str(datasets[i].name))
                self.child1.set("op1", str(datasets[i].op1))
                self.child1.set("op2", str(datasets[i].op2))
                self.child1.set("op3", str(datasets[i].op3))
                self.child1.set("op4", str(datasets[i].op4))
                for j in range(0, len(datasets[i].estratif)):
                    self.child1.set("nfolds", str(datasets[i].estratif[j].nfolds))

                    self.child2 = etree.SubElement(self.child1, "estratificado")

                    if str(datasets[i].estratif[j].id) == '0':
                        print 'Iterative'
                        print 'len: ' + str(len(datasets[i].estratif[j].methods))
                        self.child2.set("m1", 'Iterative')
                        for k in range(0, len(datasets[i].estratif[j].methods)):
                            print 'k: ' + str(k)
                            self.child3 = etree.SubElement(self.child2, "metodo")
                            print 'entro metodo - iterat'
                            self.child3.set("cbase", str(datasets[i].estratif[j].methods[k].cbase))
                            self.child3.set("method", str(datasets[i].estratif[j].methods[k].method))
                            self.child3.set("args", str(datasets[i].estratif[j].methods[k].args))

                    if str(datasets[i].estratif[j].id) == '1':
                        self.child2.set("m2", 'Random')
                        print 'Random'
                        print 'len: ' + str(len(datasets[i].estratif[j].methods))
                        for k in range(0, len(datasets[i].estratif[j].methods)):
                            print 'k: ' + str(k)
                            self.child3 = etree.SubElement(self.child2, "metodo")
                            print 'entro metodo- random'
                            self.child3.set("cbase", str(datasets[i].estratif[j].methods[k].cbase))
                            self.child3.set("method", str(datasets[i].estratif[j].methods[k].method))
                            self.child3.set("args", str(datasets[i].estratif[j].methods[k].args))

                    if str(datasets[i].estratif[j].id) == '2':
                        self.child2.set("m3", 'Labelset')
                        print "labelset"
                        for k in range(0, len(datasets[i].estratif[j].methods)):
                            print 'len: ' + str(k)
                            self.child3 = etree.SubElement(self.child2, "metodo")
                            print 'entro metodo-lbset'
                            self.child3.set("cbase", str(datasets[i].estratif[j].methods[k].cbase))
                            self.child3.set("method", str(datasets[i].estratif[j].methods[k].method))
                            self.child3.set("args", str(datasets[i].estratif[j].methods[k].args))

        my_tree = etree.ElementTree(root)

        dlg = QFileDialog().getSaveFileName(self, 'Guardar XML', selectedFilter='XML files (*.xml)')
        global xmlname
        if dlg:
            if not str(dlg).endswith('.xml'):
                dlg = os.path.splitext(str(dlg))[0] + '.xml'

            with open(dlg, 'wb') as f:
                f.write(etree.tostring(my_tree))
                self.txt.setText("Guardado fichero XML, ruta: " + str(dlg))
                xmlname = str(dlg)
                print xmlname
        else:
            print "Cancelled"

        print etree.tostring(root, pretty_print=True)

    #TODO definir señal para progressbar -> podemos definir señales propias solo con nombre, no llamada
    #   ejmplo: en vez de add(int), poner 'progress'
    # https://stackoverflow.com/questions/8649233/threading-it-is-not-safe-to-use-pixmaps-outside-the-gui-thread#8649257
        # self.threadPool[len(self.threadPool)-1].start()
        # ctrl.exec_class(self, classif, nfolds, filename)


class DatasetW(QMainWindow):

    def loadDset(self):
        self.setCentralWidget(QWidget(self))

        self.threadPool = []
        self.le = QLineEdit(file)
        self.le.setReadOnly(True)
        # layout.addWidget(self.le)
        self.btn1 = QPushButton("Cargar dataset")

        self.label = QLabel(u"Listado de dataset cargados: ")
        self.list = QListWidget()  # Al cargar dataset, estos se añadiran a la lista

        self.btn2 = QPushButton("Siguiente")
        self.btn3 = QPushButton("Guardar y ejecutar")
        self.btn4 = QPushButton("Borrar")
        self.btn5 = QPushButton(u"Añadir")
        self.flabel = QLabel("Operaciones: ")
        self.c1 = QCheckBox("Medidas")
        self.c2 = QCheckBox(u"Gráfica distribución de la correlación")
        self.c3 = QCheckBox(u"Gráfica correlación entre etiquetas")
        self.c4 = QCheckBox(u"Gráfica de frecuencia de las etiquetas")
        self.contents = QTextEdit()  # .setReadOnly(True)
        # layout.addWidget(self.contents)

        self.grid1 = QGridLayout()
        self.grid1.setSpacing(5)
        self.grid1.addWidget(self.le, 0, 0)
        self.grid1.addWidget(self.btn1, 0, 1, 1, 14)
        self.grid1.addWidget(self.label, 1, 0)
        self.grid1.addWidget(self.list, 2, 0)
        self.grid1.addWidget(self.btn4, 2, 1, 1, 14)
        self.grid1.addWidget(self.btn5, 4, 1, 1, 14)
        self.grid1.addWidget(self.flabel, 3, 0, Qt.AlignLeft)
        self.grid1.addWidget(self.c1, 4, 0, Qt.AlignLeft)
        self.grid1.addWidget(self.c2, 5, 0, Qt.AlignLeft)
        self.grid1.addWidget(self.c3, 6, 0, Qt.AlignLeft)
        self.grid1.addWidget(self.c4, 7, 0, Qt.AlignLeft)
        self.grid1.addWidget(self.contents, 8, 0)
        self.grid1.addWidget(self.btn3, 5, 1, 1, 14)
        self.grid1.addWidget(self.btn2, 6, 1, 1, 14)

        self.centralWidget().setLayout(self.grid1)
        # self.setCentralWidget(widget)
        self.contents.setText("Datasets seleccionados: ")

        self.list.itemSelectionChanged.connect(self.getOperats)

        self.c4.stateChanged.connect(self.restr)

    def __init__(self, parent=None):
        super(DatasetW, self).__init__(parent)
        # layout.addWidget(self.btn1)
        self.setMinimumSize(600, 300)  # Tamanio minimo de ventana
        self.resize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadDset()

    def add(self, text):
        if text == 'ERROR1':
            QMessageBox.about(self, "Error", "Error al cargar el dataset, por favor use datasets con formato .arff")
        elif text == 'ERROR2':
            QMessageBox.about(self, "Error", "Error en el formato de la cabecera del fichero de dataset")
        else:
            self.contents.append(text)

    def restr(self):
        if self.c4.isChecked() and not self.c1.isChecked():
            self.c1.setChecked(True)
            self.contents.append("Aviso, para calcular la frecuencia de las etiquetas hay que "
                                 "calcular las medidas del dataset")

    def getOperats(self):
        for SelectedItem in self.list.selectedItems():
            self.c1.setChecked(datasets[self.list.row(SelectedItem)].op1)
            self.c2.setChecked(datasets[self.list.row(SelectedItem)].op2)
            self.c3.setChecked(datasets[self.list.row(SelectedItem)].op3)
            self.c4.setChecked(datasets[self.list.row(SelectedItem)].op4)

    def getdataset(self):
        global file
        global text
        file = ctrl.eventload(self)
        filename.append(file)
        exists = self.list.findItems(file, Qt.MatchRegExp)
        if not exists:
            # self.list.addItem(file)
            self.contents.append(file)
            self.c1.setChecked(False)
            self.c2.setChecked(False)
            self.c3.setChecked(False)
            self.c4.setChecked(False)
        else:
            self.contents.append(u'Aviso. El dataset indicado ya se ha cargado')

    def deletedset(self):
        global datasets
        print len(datasets)
        for SelectedItem in self.list.selectedItems():

            item = self.list.takeItem(self.list.row(SelectedItem))
            datasets.pop(self.list.row(SelectedItem))
        print len(datasets)

    def partialsave(self):
        global datasets
        if not file == '':
            exists = self.list.findItems(file, Qt.MatchRegExp)
            if not exists:
                self.list.addItem(file)

                if self.c4.isChecked() and not self.c1.isChecked():
                    self.c1.setChecked(True)
                # print SelectedItem.text()
                ds = dataset()
                ds.add_name(file)
                ds.add_op1(self.c1.isChecked())
                ds.add_op2(self.c2.isChecked())
                ds.add_op3(self.c3.isChecked())
                ds.add_op4(self.c4.isChecked())
                datasets.append(ds)
                # print datasets[0].ops
                print datasets[0].name
            global proxy
            proxy.setSourceModel(self.list.model())

    def plots(self):
        if len(datasets) < 1:
            self.contents.append(u"Aviso. No se puede guardar el XML porque no hay información a guardar, "
                                 u"añada antes uno o varios datasets")
            return

        else:

            for i in range(0, len(datasets)):
                self.child1 = etree.SubElement(root, "dataset")

                self.child1.set("filename", str(datasets[i].name))
                self.child1.set("op1", str(datasets[i].op1))
                self.child1.set("op2", str(datasets[i].op2))
                self.child1.set("op3", str(datasets[i].op3))
                self.child1.set("op4", str(datasets[i].op4))

        my_tree = etree.ElementTree(root)
        global xmlname, startxml
        dlg = QFileDialog().getSaveFileName(self, 'Guardar XML', selectedFilter='XML files (*.xml)')

        if dlg:
            if not str(dlg).endswith('.xml'):
                dlg = os.path.splitext(str(dlg))[0] + '.xml'

            with open(dlg, 'wb') as f:
                f.write(etree.tostring(my_tree))
                self.contents.append("Guardado fichero XML, ruta: " + str(dlg) + '\n')
                xmlname = dlg
            # Move to xmlWindow
            startxml = True
        else:
            print "Cancelled"

        print etree.tostring(my_tree, pretty_print=True)


class FoldsW(QMainWindow):

    def loadFlds(self):
        global file
        if file == '':
            file = filename[-1]

        self.setCentralWidget(QWidget(self))
        self.threadPool = []

        self.label = QLabel(u"Listado de dataset cargados: ")
        self.lst = QListView()

        self.lst.setModel(proxy)

        self.lst.setFixedSize(700, 100)

        self.btn2 = QPushButton("Anterior")
        self.btn3 = QPushButton(u"Añadir")
        self.btn4 = QPushButton("Guardar y ejecutar")
        self.btn5 = QPushButton("Siguiente")

        self.flabel1 = QLabel(u"Número de folds: ")
        self.nlabels = QLineEdit()

        self.onlyInt = QIntValidator()
        self.nlabels.setValidator(self.onlyInt)
        self.nlabels.setText(str(nfolds))

        self.flabel3 = QLabel()
        # self.progress = QProgressBar(self)

        self.flabel2 = QLabel(u"Estratificación: ")
        self.checkmt1 = QCheckBox("Iterative")
        self.checkmt2 = QCheckBox("Random")
        self.checkmt3 = QCheckBox("Labelset")

        self.grid2 = QGridLayout()
        self.grid2.setSpacing(5)
        self.grid2.addWidget(self.label, 0, 0)
        self.grid2.addWidget(self.lst, 1, 0)
        # self.grid1.addWidget(self.contents, 1, 0)

        self.grid2.addWidget(self.flabel1, 2, 0)
        self.grid2.addWidget(self.nlabels, 3, 0, Qt.AlignLeft)

        self.grid2.addWidget(self.flabel2, 2, 0, Qt.AlignCenter)

        self.grid2.addWidget(self.checkmt1, 3, 0, Qt.AlignCenter)
        self.grid2.addWidget(self.checkmt2, 4, 0, Qt.AlignCenter)
        self.grid2.addWidget(self.checkmt3, 5, 0, Qt.AlignCenter)

        self.grid2.addWidget(self.btn2, 5, 1, 1, 18)
        self.grid2.addWidget(self.btn3, 3, 1, 1, 18)
        self.grid2.addWidget(self.btn4, 7, 1, 1, 18)
        self.grid2.addWidget(self.btn5, 8, 1, 1, 18)
        self.grid2.addWidget(self.flabel3, 8, 0, Qt.AlignLeft)
        self.flabel3.hide()
        # self.grid2.addWidget(self.progress, 10, 0)  # , Qt.AlignLeft)
        self.centralWidget().setLayout(self.grid2)
        self.lst.clicked.connect(self.getOperats)

    def __init__(self, parent=None):
        super(FoldsW, self).__init__(parent)
        # layout.addWidget(self.btn1)
        self.setMinimumSize(600, 300)  # Tamanio minimo de ventana
        self.resize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadFlds()

    def getdatasetFname(self):
        global file
        file = ctrl.eventName(self)

    def getOperats(self):
        itms = self.lst.selectedIndexes()
        self.checkmt1.setChecked(False)
        self.checkmt2.setChecked(False)
        self.checkmt3.setChecked(False)
        for SelectedItem in itms:
            if len(datasets[SelectedItem.row()].estratif) > 0:
                self.nlabels.setText(str(datasets[SelectedItem.row()].estratif[SelectedItem.row()].nfolds))
                for i in range(0, len(datasets[SelectedItem.row()].estratif)):
                    # print str(datasets[SelectedItem.row()].estratif[SelectedItem.row()].id)
                    if str(datasets[SelectedItem.row()].estratif[i].id) == '0':
                        x1 = True

                        self.checkmt1.setChecked(x1)
                    if str(datasets[SelectedItem.row()].estratif[i].id) == '1':
                        x2 = True

                        self.checkmt2.setChecked(x2)
                    if str(datasets[SelectedItem.row()].estratif[i].id) == '2':
                        x3 = True

                        self.checkmt3.setChecked(x3)
            else:
                self.checkmt1.setChecked(False)
                self.checkmt2.setChecked(False)
                self.checkmt3.setChecked(False)
                self.nlabels.setText('0')

    def partsave(self):
        itms = self.lst.selectedIndexes()
        c1 = self.checkmt1.isChecked()
        c2 = self.checkmt2.isChecked()
        c3 = self.checkmt3.isChecked()
        for it in itms:
            for i in range(0, 3):
                if c1:
                    f = stratif(0)  # id= 0 iterative
                    if self.nlabels.text() == '0':
                        self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
                        self.flabel3.show()
                    else:
                        f.add_folds(int(self.nlabels.text()))

                        self.flabel3.setText(u"Configuración guardada")
                        self.flabel3.show()

                    datasets[it.row()].set_estratif(f)
                    c1 = False

                if c2:
                    f = stratif(1)  # id= 1 random
                    if self.nlabels.text() == '0':
                        self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
                        self.flabel3.show()
                    else:
                        f.add_folds(int(self.nlabels.text()))

                        self.flabel3.setText(u"Configuración guardada")
                        self.flabel3.show()

                    datasets[it.row()].set_estratif(f)
                    c2 = False

                if c3:
                    f = stratif(2)  # id= 2 labelset
                    if self.nlabels.text() == '0':
                        self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
                        self.flabel3.show()
                    else:
                        f.add_folds(int(self.nlabels.text()))

                        self.flabel3.setText(u"Configuración guardada")
                        self.flabel3.show()

                    datasets[it.row()].set_estratif(f)
                    c3 = False

            print 'len: ' + str(len(datasets[it.row()].estratif))

    def folds(self):

        global file, nfolds
        global xmlname, startxml

        if self.nlabels.text() == '0':
            self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
        else:
            for i in range(0, len(datasets)):
                self.child1 = etree.SubElement(root, "dataset")
                self.child1.set("filename", str(datasets[i].name))
                self.child1.set("op1", str(datasets[i].op1))
                self.child1.set("op2", str(datasets[i].op2))
                self.child1.set("op3", str(datasets[i].op3))
                self.child1.set("op4", str(datasets[i].op4))
                print len(datasets[i].estratif)
                for j in range(0, len(datasets[i].estratif)):
                    self.child2 = etree.SubElement(self.child1, "estratificado")
                    self.child2.set("filename", str(datasets[i].name))
                    self.child2.set("nfolds", str(datasets[i].estratif[i].nfolds))
                    if str(datasets[i].estratif[j].id) == '0':
                        self.child2.set("m1", 'Iterative')
                    if str(datasets[i].estratif[j].id) == '1':
                        self.child2.set("m2", 'Random')
                    if str(datasets[i].estratif[j].id) == '2':
                        self.child2.set("m3", 'Labelset')

            my_tree = etree.ElementTree(root)

        dlg = QFileDialog().getSaveFileName(self, 'Guardar XML', selectedFilter='XML files (*.xml)')

        if dlg:
            if not str(dlg).endswith('.xml'):
                dlg = os.path.splitext(str(dlg))[0] + '.xml'
            if len(datasets) == 0:
                self.flabel3.setText(u"Error. No se han añadido datasets")
                self.flabel3.show()
                return

            if self.nlabels.text() == '0':
                self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
                self.flabel3.show()
            else:
                f = open(dlg, 'wb')
                f.write(etree.tostring(my_tree))
                self.flabel3.setText("Guardado fichero XML, ruta: " + str(dlg))
                self.flabel3.show()
                xmlname = dlg
            startxml = True
        else:
            print "Cancelled"

        print etree.tostring(my_tree, pretty_print=True)
        # TODO : Cerrar y abrir la ventana de XML


# This is the main Window of the aplication
class MainApplication(QMainWindow):
    def loadMain(self):
        self.setCentralWidget(QWidget(self))
        self.btn1 = QPushButton("Cargar y analizar dataset")
        self.btn2 = QPushButton(u"Configurar folds")
        self.btn3 = QPushButton(u"Configurar clasificación")
        self.btn4 = QPushButton("Cargar configuraciones y ejecutar")

        self.btn1.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)
        self.btn2.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)
        self.btn3.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)
        self.btn4.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding)

        self.grid = QGridLayout()
        self.grid.setSpacing(5)
        self.grid.addWidget(self.btn1, 0, 0)
        self.grid.addWidget(self.btn2, 1, 0)
        self.grid.addWidget(self.btn3, 2, 0)
        self.grid.addWidget(self.btn4, 3, 0)

        self.centralWidget().setLayout(self.grid)

        self.DataUI = DatasetW(self)
        self.FoldUI = FoldsW(self)
        self.ClassifUI = ClassifW(self)
        self.xmlUI = XmlW(self)

        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)
        #self.btn4.setEnabled(False)

        self.btn1.clicked.connect(self.startDatasetTab)
        self.btn2.clicked.connect(self.startFoldsTab)
        self.btn3.clicked.connect(self.startClassifTab)
        self.btn4.clicked.connect(self.startxmlTab)

    def __init__(self, parent=None):
        super(MainApplication, self).__init__(parent)
        # layout.addWidget(self.btn1)
        self.setMinimumSize(600, 300)  # Tamanio minimo de ventana
        self.resize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadMain()

    def startDatasetTab(self):

        global text
        global filename
        if len(text) > 1:
            self.DataUI.contents.append(text)
            self.DataUI.le.setText(filename)

        self.DataUI.btn1.clicked.connect(self.DataUI.getdataset)
        self.DataUI.btn2.clicked.connect(self.startFoldsTab)
        self.DataUI.btn3.clicked.connect(self.plot_xml)
        self.DataUI.btn4.clicked.connect(self.DataUI.deletedset)
        self.DataUI.btn5.clicked.connect(self.DataUI.partialsave)

        self.DataUI.show()

    def plot_xml(self):

        self.DataUI.plots()
        if startxml:
            self.DataUI.close()
            self.startxmlTab()

    def startFoldsTab(self):

        self.DataUI.close()
        self.btn2.setEnabled(True)
        # self.FoldUI.btn1.clicked.connect(self.FoldUI.getdatasetFname)
        self.FoldUI.btn2.clicked.connect(self.startDatasetTab)
        self.FoldUI.btn3.clicked.connect(self.FoldUI.partsave)
        self.FoldUI.btn4.clicked.connect(self.fold_xml)
        self.FoldUI.btn5.clicked.connect(self.startClassifTab)

        self.FoldUI.show()

    def fold_xml(self):
        self.FoldUI.folds()
        if startxml:
            self.FoldUI.close()
            self.startxmlTab()

    def startClassifTab(self):

        self.FoldUI.close()
        self.btn3.setEnabled(True)
        self.btn4.setEnabled(True)
        self.ClassifUI.btn2.clicked.connect(self.startFoldsTab)
        self.ClassifUI.btn3.clicked.connect(self.ClassifUI.confmethods)
        self.ClassifUI.methods.activated.connect(self.ClassifUI.signalDisable)
        self.ClassifUI.btn4.clicked.connect(self.ClassifUI.getParams)
        self.ClassifUI.btn5.clicked.connect(self.startxmlTab)
        self.ClassifUI.show()

    def startxmlTab(self):

        if xmlname == '':
            # Guardar las ops antes d camb d ventana
            self.ClassifUI.getParams()

        self.ClassifUI.close()
        self.btn4.setEnabled(True)
        self.xmlUI.load.setText(xmlname)
        self.xmlUI.btn1.clicked.connect(self.xmlUI.getxmlfile)
        self.xmlUI.btn2.clicked.connect(self.xmlUI.getWorkingDir)
        self.xmlUI.btn3.clicked.connect(self.xmlUI.execute)
        self.xmlUI.show()

    def closeEvent(self, event):  # Redefinimos el evento de cierre (pedir confirmacion)

        reply = QMessageBox.question(self, 'Aviso',
                                     u"Está seguro de que desea salir?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)

    ex = MainApplication()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
