# coding=utf-8
import shutil
import sys
import os
from PyQt4.QtGui import *
from PyQt4.QtCore import *

import ctrl
from lxml import etree
from functools import partial
import io

# TODO https://stackoverflow.com/questions/3605680/creating-a-simple-xml-file-using-python#3605831 #usamos lxml

# TODO
#  BUGS:
#        2. cuando termina la ejecucion salen 2 errors/warnings sobre QObject::startTumer: Qtimer solo se puede usar en
#        hilos lanzados por QThread.
#        3. Object::disconnect: Unexpected null parameter en el pc de casa
#        4. Al cancelar la ejecucion del programa porq se pulse volver a otra ventana o cerrar, que termine
#        todo correctamente - creo q done

root = etree.Element("experimento")
file = ''
text = str('')
nfolds = 0
mk1 = False
mk2 = False
mk3 = False
classif = []
dir = './'
wdir = './'
xmlname = ''
startxml = False
datasets = []  # Contendra elems de la clase dset, estratif y metodo
xmlUI = None

dsW = False
fW = False
cW = False
xW = False

p1 = False  # PlotW created flag

executing = False

proxy = QIdentityProxyModel()


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

    def del_estratif(self, idx):
        self.estratif.pop(idx)


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
        self.runs = True

    def __del__(self):
        self.wait()

    def run(self):
        self.starting()
        self.stop()

    def starting(self):
        while self.runs:
            self.function(*self.args, **self.kwargs)
            self.runs = False

    def stop(self):
        self.runs = False


class PlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PlotWindow, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        self.setMaximumSize(700, 900)

        framegm = self.frameGeometry()

        self.move(framegm.bottomLeft())

        self.log("Inicializando ventana PlotWindow...")
        searchdir = str(dir) + '/' + 'tmp/'
        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.connect(self, SIGNAL("del"), self.closeEvent)

        if os.path.exists(searchdir):

            self.scrollArea = QScrollArea(widgetResizable=True)
            self.setCentralWidget(self.scrollArea)
            content_widget = QWidget()
            self.scrollArea.setWidget(content_widget)
            lay = QVBoxLayout(content_widget)
            self.saveall = QPushButton("Guardar todas")

            i = 0

            self.saveP = []
            tosave = []
            self.all = 0
            # QMessageBox.about(self, 'Procesando...', u"Cargando gráficas...")

            for file in os.listdir(searchdir):
                pixmap = QPixmap(os.path.join(searchdir, file))
                if not pixmap.isNull():
                    label = QLabel(pixmap=pixmap.scaled(640, 480, Qt.KeepAspectRatio))

                    lay.addWidget(label)

                    tosave.append(file)
                    tmp1 = QPushButton("Guardar")
                    self.saveP.append(tmp1)
                    lay.addWidget(self.saveP[i])
                    i += 1

            if len(self.saveP) == 0:
                # emitir señal de cerrar ventana
                self.emit(SIGNAL('del'), 'close')
                self.close()

            lay.addWidget(self.saveall)
            self.saveall.clicked.connect(partial(self.save_complete, tosave))

            self.idx = 0

            for p in range(0, len(self.saveP)):
                self.saveP[p].clicked.connect(partial(self.save, tosave[p]))

        else:
            self.log("Cerrando ventana...")
            self.emit(SIGNAL('del'), 'close')
            # self.close()

    def log(self, txt):
        if not os.path.exists(wdir+'/log/'):
            os.makedirs(wdir+'/log/')
        logger = wdir + '/log/dump.log'
        fp = io.open(logger, 'a', encoding='utf-8')
        txt = unicode(txt, 'utf-8')
        fp.write(txt)
        fp.close()

    def closeEvent(self, event):
        global p1
        # Borrar carpeta temporal de gráficas
        if not dir == '' and os.path.isdir(dir + '/tmp/'):
            shutil.rmtree(dir + '/tmp/')

        if event == 'close':
            self.destroy()
        else:
            event.accept()

        p1 = False

        if xmlUI:
            xmlUI.exec_folds()

    def save_complete(self, fnams):
        route = str(QFileDialog.getExistingDirectory(self, 'Select Directory'))

        if route:
            for t in range(0, len(fnams)):
                with open(route + '/' + fnams[t], 'wb') as f:
                    fr = open(dir + '/tmp/' + fnams[t], 'rb')
                    data = fr.read()
                    f.write(data)

                    f.close()
                    fr.close()
            QMessageBox.about(self, "Correcto", u"Gráficas guardadas correctamente en el directorio: " + str(route))
            self.log("Gráficas - guardado correcto")
        else:
            QMessageBox.about(self, "Cancelado", u"Guardado cancelado")
            self.log("Gráficas - guardado cancelado")

    def save(self, fname):

        dlg = QFileDialog().getSaveFileName(self, u'Guardar gráfica', selectedFilter='Image files (*.png)')

        if dlg:
            if not str(dlg).endswith('.png'):
                dlg = os.path.splitext(str(dlg))[0] + '.png'

            with open(dlg, 'wb') as f:
                fr = open(dir + '/tmp/' + fname, 'rb')
                data = fr.read()
                f.write(data)
                QMessageBox.about(self, "OK", u"Gráfica guardada, ruta: " + str(dlg))
                self.log("Gráfica " + str(fname) + " - guardado correcto")

                f.close()
                fr.close()
        else:
            QMessageBox.about(self, "Cancelado", u"Guardado cancelado")


class XmlW(QMainWindow):
    def loadXmlW(self):

        self.setCentralWidget(QWidget(self))

        self.threadPool = []

        self.load = QLineEdit()  # Para cargar el xml a ejecutar

        self.load.setReadOnly(True)
        self.btn1 = QPushButton("Cargar fichero XML")
        self.workingDir = QLineEdit()  # Para cargar el xml a ejecutar
        self.workingDir.setReadOnly(True)
        self.workingDir.setText(wdir)
        self.btn2 = QPushButton("Seleccionar directorio de trabajo")
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.progress = QProgressBar()
        self.btn3 = QPushButton("Ejecutar")
        self.btn4 = QPushButton("Reiniciar")

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

        framegm = self.frameGeometry()
        centerp = QApplication.desktop().screenGeometry().center()
        framegm.moveCenter(centerp)
        self.move(framegm.topLeft())

        self.plots = None

    def __init__(self, parent=None):
        super(XmlW, self).__init__(parent)

        self.setFixedSize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadXmlW()

    def getxmlfile(self):
        global xmlname
        xmlname = ctrl.loadxml(self)
        self.load.setText(str(xmlname))
        self.info.clear()

    def getWorkingDir(self):
        global dir, wdir
        dir = ctrl.getsaveDir(self)
        wdir = dir
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
            if txt == u'\nTerminado\n':
                self.log()

    def prog1(self, progr):
        self.progress.setValue(progr)

    def upd(self, progress):
        self.progress.setValue(progress)

    def logconsole(self, txt):
        if not os.path.exists(wdir + '/log/'):
            os.makedirs(wdir + '/log/')

        logger = wdir + '/log/dump.log'
        fp = io.open(logger, 'a', encoding='utf-8')
        txt = unicode(txt, 'utf-8')
        fp.write(txt)
        fp.close()

    def log(self):
        if not os.path.exists(wdir + '/log/'):
            os.makedirs(wdir + '/log/')

        logger = wdir + '/log/window.log'
        dump = self.info.toPlainText()
        dump = dump.toUtf8()
        fp = open(logger, 'w')
        fp.write(dump)
        fp.close()

    def nxt(self):
        global p1
        if not p1 and os.path.exists(str(dir) + '/' + 'tmp/'):
            self.plots = PlotWindow(self)
            self.plots.show()
            p1 = True
        else:
            if xmlUI:
                xmlUI.exec_folds()

        if not xmlUI:
            logger = wdir + '/log/window.log'
            fp = open(logger, 'a')
            fp.write("\n>Se ha cancelado la ejecución")
            fp.close()
        else:
            self.log()

        # self.exec_folds()

    def closeEvent(self, event):

        if self.plots:
            self.plots.close()
        global xmlUI, xW, xmlname, dir, p1
        xmlUI = None
        xW = False
        p1 = False # No vuelve a cargar la ventana si cerramos con (X) la ventana - p1 a true
        xmlname = ''
        dir = './'
        self.btn3.setEnabled(True)

    def lst(self):
        if not xmlUI:
            logger = wdir + '/log/window.log'
            fp = open(logger, 'a')
            fp.write("\n>Se ha cancelado la ejecución")
            fp.close()
        else:
            self.log()
            self.exec_class()
            self.log()
            self.btn3.setEnabled(True)

    def execute(self):
        self.info.clear()
        if not xmlname == '':
            self.btn3.setEnabled(False)
            global executing
            executing = True
            self.exec_ds()
        else:
            pass

    def exec_ds(self):

        self.threadPool.append(GenericThread(ctrl.execute_dset, self, xmlname, dir))

        self.disconnect(self, SIGNAL("textoinf"), self.adds)
        self.connect(self, SIGNAL("textoinf"), self.adds)

        self.disconnect(self, SIGNAL("finished"), self.nxt)
        self.connect(self, SIGNAL("finished"), self.nxt)

        self.disconnect(self, SIGNAL("prog1"), self.upd)
        self.connect(self, SIGNAL("prog1"), self.upd)

        self.disconnect(self, SIGNAL("logcns_ds"), self.logconsole)
        self.connect(self, SIGNAL("logcns_ds"), self.logconsole)

        self.threadPool[len(self.threadPool) - 1].start()

    def exec_folds(self):

        self.progress.setValue(0)
        self.threadPool.append(GenericThread(ctrl.execute_folds, self, xmlname, dir))
        self.disconnect(self, SIGNAL("add(QString)"), self.add)
        self.connect(self, SIGNAL("add(QString)"), self.add)

        self.disconnect(self, SIGNAL("update(int)"), self.upd)
        self.connect(self, SIGNAL("update(int)"), self.upd)

        self.disconnect(self, SIGNAL("end"), self.lst)
        self.connect(self, SIGNAL("end"), self.lst)

        self.disconnect(self, SIGNAL("logcns_f"), self.logconsole)
        self.connect(self, SIGNAL("logcns_f"), self.logconsole)

        self.threadPool[len(self.threadPool) - 1].start()

    def exec_class(self):

        self.progress.setValue(0)
        self.threadPool.append(GenericThread(ctrl.execute_class, self, xmlname, dir))

        self.disconnect(self, SIGNAL("infoclassif"), self.addc)
        self.connect(self, SIGNAL("infoclassif"), self.addc)

        self.disconnect(self, SIGNAL("progress"), self.upd)
        self.connect(self, SIGNAL("progress"), self.upd)

        self.disconnect(self, SIGNAL("log"), self.log)
        self.connect(self, SIGNAL("log"), self.log)

        self.disconnect(self, SIGNAL("logcns_c"), self.logconsole)
        self.connect(self, SIGNAL("logcns_c"), self.logconsole)

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

        self.lst.setFixedSize(700, 90)
        self.lst2.setFixedSize(700, 90)

        self.btn2 = QPushButton("Reiniciar")
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

        self.grid = QGridLayout()
        self.grid.setSpacing(2)
        self.grid.addWidget(self.label, 0, 0)
        self.grid.addWidget(self.lst, 1, 0)
        self.grid.addWidget(self.btn2, 10, 1, 1, 14)
        self.grid.addWidget(self.btn3, 5, 1, 1, 14)
        self.grid.addWidget(self.btn4, 6, 1, 1, 14)
        self.grid.addWidget(self.btn5, 8, 1, 1, 14)

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

        if nfolds <= 1:
            self.txt.setText(u"Aviso. No se podrá realizar la clasificación "
                             u"si no hay más de 2 particionados realizados")
            self.txt.show()

        self.centralWidget().setLayout(self.grid)

        framegm = self.frameGeometry()
        centerp = QApplication.desktop().screenGeometry().center()
        framegm.moveCenter(centerp)
        self.move(framegm.topLeft())

        self.btn3.setEnabled(False)

        self.lst.clicked.connect(self.getOperats)
        self.checkmt1.clicked.connect(self.enable)
        self.checkmt2.clicked.connect(self.enable)
        self.checkmt3.clicked.connect(self.enable)

    def __init__(self, parent=None):
        super(ClassifW, self).__init__(parent)

        self.setFixedSize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadClass()

    def enable(self):
        if self.checkmt1.isChecked() or self.checkmt2.isChecked() or self.checkmt3.isChecked():
            self.btn3.setEnabled(True)
        else:
            self.btn3.setEnabled(False)

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
                    # print str(datasets[SelectedItem.row()].estratif[i].id)
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
            dname = str(datasets[it.row()].name)
            dname = os.path.basename(dname)
            dname = os.path.splitext(dname)[0]
            self.lst2.addItem(dname + ': ' + str(self.methods.currentText()) + ', '
                              + str(self.base.currentText()) + ' ' + str(args))

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

    def log(self, text):
        logger = wdir + '/dump.log'
        fp = io.open(logger, 'a', encoding='utf-8')
        text = unicode(text, 'utf-8')
        fp.write(text)
        fp.close()

    def getXml(self):
        # Para cada metodo de clasificacion elegido, tras haberlo config -> ejecutar dicha clasificacion
        global classif, root
        root = etree.Element("experimento")

        for i in range(0, len(datasets)):
            p = len(datasets[i].estratif)
            if any(datasets[i].estratif[0 % p].methods) or any(datasets[i].estratif[1 % p].methods) or \
                    any(datasets[i].estratif[2 % p].methods):
                pass
            else:
                reply = QMessageBox.question(self, "Aviso", u"Debería añadir al menos un "
                                             u"método de clasificación para cada dataset. ¿Desea continuar "
                                             u"de todos modos?", QMessageBox.Yes |
                                             QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    pass
                else:
                    return 'error'
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

                        self.child2.set("m1", 'Iterative')
                        for k in range(0, len(datasets[i].estratif[j].methods)):

                            self.child3 = etree.SubElement(self.child2, "metodo")
                            self.child3.set("cbase", str(datasets[i].estratif[j].methods[k].cbase))
                            self.child3.set("method", str(datasets[i].estratif[j].methods[k].method))
                            self.child3.set("args", str(datasets[i].estratif[j].methods[k].args))

                    if str(datasets[i].estratif[j].id) == '1':
                        self.child2.set("m2", 'Random')
                        for k in range(0, len(datasets[i].estratif[j].methods)):
                            self.child3 = etree.SubElement(self.child2, "metodo")
                            self.child3.set("cbase", str(datasets[i].estratif[j].methods[k].cbase))
                            self.child3.set("method", str(datasets[i].estratif[j].methods[k].method))
                            self.child3.set("args", str(datasets[i].estratif[j].methods[k].args))

                    if str(datasets[i].estratif[j].id) == '2':
                        self.child2.set("m3", 'Labelset')
                        for k in range(0, len(datasets[i].estratif[j].methods)):
                            self.child3 = etree.SubElement(self.child2, "metodo")
                            self.child3.set("cbase", str(datasets[i].estratif[j].methods[k].cbase))
                            self.child3.set("method", str(datasets[i].estratif[j].methods[k].method))
                            self.child3.set("args", str(datasets[i].estratif[j].methods[k].args))

            my_tree = etree.ElementTree(root)

            dlg = QFileDialog().getSaveFileName(self, 'Guardar XML', selectedFilter='XML files (*.xml)')

            global xmlname, startxml

        if dlg:
            if not str(dlg).endswith('.xml'):
                dlg = os.path.splitext(str(dlg))[0] + '.xml'

            with open(dlg, 'wb') as f:
                f.write(etree.tostring(my_tree))
                self.txt.setText("Guardado fichero XML, ruta: " + str(dlg))
                xmlname = str(dlg)
                self.log("Guardado xml: " +str(xmlname) + '\n')
                # print etree.tostring(root, pretty_print=True)

                reply = QMessageBox.question(self, u'Información',
                                             u"Fichero guardado, ¿desea pasar a ejecutar el experimento?",
                                             QMessageBox.Yes |
                                             QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    startxml = True
                else:
                    pass

        else:
            return 'error'

    # Definir señal para progressbar -> podemos definir señales propias solo con nombre, no llamada
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
        self.btn1 = QPushButton("Cargar dataset")

        self.label = QLabel(u"Listado de dataset cargados: ")
        self.list = QListWidget()  # Al cargar dataset, estos se añadiran a la lista

        self.btn6 = QPushButton("Reiniciar")
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
        self.grid1.addWidget(self.btn6, 8, 1, 1, 14)

        self.centralWidget().setLayout(self.grid1)

        framegm = self.frameGeometry()
        centerp = QApplication.desktop().screenGeometry().center()
        framegm.moveCenter(centerp)
        self.move(framegm.topLeft())
        # self.setCentralWidget(widget)
        self.contents.setText("Datasets seleccionados: ")

        self.list.itemSelectionChanged.connect(self.getOperats)

        self.c4.stateChanged.connect(self.restr)
        # Por defecto desactivados los botones para selecc ops sobre dataset hasta q se cargue uno
        self.c1.setEnabled(False)
        self.c2.setEnabled(False)
        self.c3.setEnabled(False)
        self.c4.setEnabled(False)

    def __init__(self, parent=None):
        super(DatasetW, self).__init__(parent)

        self.setFixedSize(850, 400)  # Tamanio por defecto de ventana

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

    # Función para controlar la restricción de la casilla 4 en la ventana de configurar el dataset
    def restr(self):
        if self.c4.isChecked() and not self.c1.isChecked():
            self.c1.setChecked(True)
            self.contents.append("Aviso, para calcular la frecuencia de las etiquetas hay que "
                                 "calcular las medidas del dataset")

    def closeEvent(self, evnt):
        global file, dsW
        file = ''
        while self.grid1.count():
            item = self.grid1.takeAt(0)
            item.widget().deleteLater()
        dsW = False

    def getOperats(self):
        for SelectedItem in self.list.selectedItems():
            self.c1.setChecked(datasets[self.list.row(SelectedItem)].op1)
            self.c2.setChecked(datasets[self.list.row(SelectedItem)].op2)
            self.c3.setChecked(datasets[self.list.row(SelectedItem)].op3)
            self.c4.setChecked(datasets[self.list.row(SelectedItem)].op4)

    def getdataset(self):
        global file
        global text

        file = self.le.text()
        if file == '':
            file = ctrl.eventload(self)
            if not file == '':  # Se ha cancelado la operac
                exists = self.list.findItems(file, Qt.MatchRegExp)
                if not exists:
                    # self.list.addItem(file)
                    self.contents.append(file)
                    self.c1.setEnabled(True)
                    self.c2.setEnabled(True)
                    self.c3.setEnabled(True)
                    self.c4.setEnabled(True)

                    self.c1.setChecked(False)
                    self.c2.setChecked(False)
                    self.c3.setChecked(False)
                    self.c4.setChecked(False)

            else:
                self.contents.append(u"Operación cancelada")
        else:
            reply = QMessageBox.question(self, 'Aviso',
                                         u"Ya se ha cargado un dataset, ¿desea cargar otro sin guardar el actual?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if reply == QMessageBox.Yes:
                file = ctrl.eventload(self)
                self.contents.undo()
                self.contents.append(file)
                self.c1.setChecked(False)
                self.c2.setChecked(False)
                self.c3.setChecked(False)
                self.c4.setChecked(False)

    def deletedset(self):
        global datasets
        for SelectedItem in self.list.selectedItems():
            item = self.list.takeItem(self.list.row(SelectedItem))
            datasets.pop(self.list.row(SelectedItem))

        if self.list.count() == 0:
            self.c1.setChecked(False)
            self.c2.setChecked(False)
            self.c3.setChecked(False)
            self.c4.setChecked(False)

    def partialsave(self):
        global datasets, file
        for selected in self.list.selectedItems():
            datasets[self.list.row(selected)].op1 = self.c1.isChecked()
            datasets[self.list.row(selected)].op2 = self.c2.isChecked()
            datasets[self.list.row(selected)].op3 = self.c3.isChecked()
            datasets[self.list.row(selected)].op4 = self.c4.isChecked()

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

                file = ''
                self.le.setText('')
                self.contents.append(u">Añadido correctamente")
                self.c1.setChecked(False)
                self.c2.setChecked(False)
                self.c3.setChecked(False)
                self.c4.setChecked(False)

            global proxy
            proxy.setSourceModel(self.list.model())
        else:
            self.contents.append(u"Aviso, no se ha seleccionado dataset.")

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
            b1 = True

        #print etree.tostring(my_tree, pretty_print=True)


class FoldsW(QMainWindow):

    def loadFlds(self):

        self.setCentralWidget(QWidget(self))
        self.threadPool = []

        self.label = QLabel(u"Listado de dataset cargados: ")
        self.lst = QListView()

        self.lst.setModel(proxy)

        self.lst.setFixedSize(700, 100)

        self.btn2 = QPushButton("Reiniciar")
        self.btn1 = QPushButton(u"Añadir a todos")
        self.btn3 = QPushButton(u"Añadir")
        self.btn4 = QPushButton("Guardar y ejecutar")
        self.btn5 = QPushButton("Siguiente")

        self.flabel1 = QLabel(u"Número de folds: ")
        self.flabel = QLabel()  # hidden
        self.nlabels = QLineEdit()

        self.onlyInt = QIntValidator(0, 100)
        self.nlabels.setValidator(self.onlyInt)
        self.nlabels.setText(str(nfolds))

        self.flabel3 = QLabel()

        self.flabel2 = QLabel(u"Estratificación: ")
        self.checkmt1 = QCheckBox("Iterative")
        self.checkmt2 = QCheckBox("Random")
        self.checkmt3 = QCheckBox("Labelset")

        self.grid2 = QGridLayout()
        self.grid2.setSpacing(5)
        self.grid2.addWidget(self.label, 0, 0)
        self.grid2.addWidget(self.lst, 1, 0)

        self.grid2.addWidget(self.flabel1, 2, 0)
        self.grid2.addWidget(self.nlabels, 3, 0, Qt.AlignLeft)

        self.grid2.addWidget(self.flabel2, 2, 0, Qt.AlignCenter)

        self.grid2.addWidget(self.checkmt1, 3, 0, Qt.AlignCenter)
        self.grid2.addWidget(self.checkmt2, 4, 0, Qt.AlignCenter)
        self.grid2.addWidget(self.checkmt3, 5, 0, Qt.AlignCenter)

        self.grid2.addWidget(self.btn1, 2, 1, 1, 18)
        self.grid2.addWidget(self.btn3, 3, 1, 1, 18)
        self.grid2.addWidget(self.btn4, 4, 1, 1, 18)
        self.grid2.addWidget(self.btn5, 5, 1, 1, 18)
        self.grid2.addWidget(self.btn2, 7, 1, 1, 18)
        self.grid2.addWidget(self.flabel, 6, 0)
        self.grid2.addWidget(self.flabel3, 7, 0, Qt.AlignLeft)
        self.flabel3.show()

        self.centralWidget().setLayout(self.grid2)

        framegm = self.frameGeometry()
        centerp = QApplication.desktop().screenGeometry().center()
        framegm.moveCenter(centerp)
        self.move(framegm.topLeft())

        self.lst.clicked.connect(self.getOperats)

    def __init__(self, parent=None):
        super(FoldsW, self).__init__(parent)

        self.setFixedSize(850, 400)  # Tamanio por defecto de ventana

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadFlds()

    def getOperats(self):
        itms = self.lst.selectedIndexes()
        self.checkmt1.setChecked(False)
        self.checkmt2.setChecked(False)
        self.checkmt3.setChecked(False)

        for SelectedItem in itms:
            if len(datasets[SelectedItem.row()].estratif) > 0:
                for i in range(0, len(datasets[SelectedItem.row()].estratif)):
                    self.nlabels.setText(str(datasets[SelectedItem.row()].estratif[i].nfolds))
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

    def partsave(self, mode):  # Pulsar boton Añadir (mode -> añadir (1)/ añadir todos (2)
        if mode == 2:
            self.lst.setSelectionMode(QAbstractItemView.MultiSelection)
            self.lst.selectAll()

        itms = self.lst.selectedIndexes()

        c1 = self.checkmt1.isChecked()
        c2 = self.checkmt2.isChecked()
        c3 = self.checkmt3.isChecked()
        global nfolds
        if not c1 and not c2 and not c3:
            self.flabel3.setText(u"Aviso. No se ha seleccionado ningún método...")
            return

        if len(itms) >= 1:
            for d in itms:
                for l in reversed(range(0, len(datasets[d.row()].estratif))):

                    datasets[d.row()].del_estratif(l)
                    # Esto reseteará los estratifs si volvemos a añadir sobre uno ya establecido
            for it in itms:
                c1 = self.checkmt1.isChecked()
                c2 = self.checkmt2.isChecked()
                c3 = self.checkmt3.isChecked()

                for i in range(0, 3):
                    if c1:
                        f = stratif(0)  # id= 0 iterative
                        if self.nlabels.text() == '0':
                            self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
                            self.flabel3.show()
                        else:
                            f.add_folds(int(self.nlabels.text()))
                            nfolds = int(self.nlabels.text())

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
                            nfolds = int(self.nlabels.text())

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
                            nfolds = int(self.nlabels.text())

                            self.flabel3.setText(u"Configuración guardada")
                            self.flabel3.show()

                            datasets[it.row()].set_estratif(f)
                            c3 = False

        else:

            self.flabel3.setText(u"Aviso. No se ha seleccionado ningún dataset")
            self.flabel3.show()
        #c1 = False
        #c2 = False
        #c3 = False
        self.lst.setSelectionMode(QAbstractItemView.SingleSelection)

    def folds(self):

        global file, nfolds
        global xmlname, startxml
        global root

        # Si llegamos a esta ventana, no se ha guardado el experim asi q podemos resetearlo para evitar
        # elems no deseados
        root = etree.Element("experimento")

        for i in range(0, len(datasets)):
            if not len(datasets[i].estratif) > 0:
                QMessageBox.about(self, "Aviso", u"Aviso: No se han añadido estratificados "
                                                 u"a uno o varios datasets")
                return
        if self.nlabels.text() == '0':
            self.flabel3.setText("Aviso. No se pueden ejecutar particiones con 0 folds...")
            return
        else:
            for i in range(0, len(datasets)):
                if len(datasets[i].estratif) > 0:
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
                            self.child2.set("m1", 'Iterative')
                        if str(datasets[i].estratif[j].id) == '1':
                            self.child2.set("m2", 'Random')
                        if str(datasets[i].estratif[j].id) == '2':
                            self.child2.set("m3", 'Labelset')
                else:
                    self.flabel3.setText(u"Error. No se ha añadido ningún estratificado")
                    return

            my_tree = etree.ElementTree(root)

        dlg = QFileDialog().getSaveFileName(self, 'Guardar XML', selectedFilter='XML files (*.xml)')

        if dlg:
            if not str(dlg).endswith('.xml'):
                dlg = os.path.splitext(str(dlg))[0] + '.xml'
            if len(datasets) == 0:
                self.flabel3.setText(u"Error. No se han añadido datasets")
                self.flabel3.show()
                return

            for i in range(0, len(datasets)):
                for j in range(0, len(datasets[i].estratif)):
                    if str(datasets[i].estratif[j].nfolds) == '0':
                        self.flabel3.setText("Aviso (error 0 folds), no se ha podido guardar el fichero XML")
                        self.flabel3.show()
                        return

            f = open(dlg, 'wb')
            f.write(etree.tostring(my_tree))
            self.flabel3.setText("Guardado fichero XML, ruta: " + str(dlg))
            self.flabel3.show()
            xmlname = dlg
            #print etree.tostring(my_tree, pretty_print=True)
            startxml = True
        else:
            pass
            #print "Cancelled"


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

        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)
        # self.btn4.setEnabled(False)

        self.btn1.clicked.connect(self.startDatasetTab)
        self.btn2.clicked.connect(self.startFoldsTab)
        self.btn3.clicked.connect(self.startClassifTab)
        self.btn4.clicked.connect(self.startxmlTab)

    def __init__(self, parent=None):
        super(MainApplication, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.CustomizeWindowHint)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowSystemMenuHint & Qt.WindowCloseButtonHint
                            & Qt.WindowMinimizeButtonHint)

        self.setFixedSize(850, 400)  # Tamanio por defecto de ventana

        framegm = self.frameGeometry()
        # centerp = QApplication.desktop().screenGeometry().center()
        # framegm.moveCenter(centerp)
        self.move(framegm.topLeft())

        self.setWindowTitle(u"Herramienta para el estudio del problema "
                            u"de desequilibrio en problemas de clasificación multietiqueta")

        self.loadMain()

    def startDatasetTab(self):

        global dsW

        if not dsW:
            self.DataUI = DatasetW(self)
            dsW = True
        if dsW:

            self.DataUI.btn1.clicked.connect(self.DataUI.getdataset)
            self.DataUI.btn2.clicked.connect(self.startFoldsTab)
            self.DataUI.btn3.clicked.connect(self.plot_xml)
            self.DataUI.btn4.clicked.connect(self.DataUI.deletedset)
            self.DataUI.btn5.clicked.connect(self.DataUI.partialsave)
            self.DataUI.btn6.clicked.connect(self.restart)

            if not self.DataUI.isVisible():
                self.DataUI.show()
                # print self.DataUI.isVisible()
        if fW:
            self.FoldUI.close()
        if xW:
            xmlUI.close()

    def plot_xml(self):

        self.DataUI.plots()

        if startxml:
            self.DataUI.close()
            self.startxmlTab()

    def startFoldsTab(self):

        global fW

        warn = 0
        if len(datasets) < 1:
            QMessageBox.about(self.DataUI, "Aviso", u"No se ha guardado ningún dataset, "
                                                    u"añada al menos uno para pasar a la siguiente ventana")
            return

        for i in range(0, len(datasets)):
            if not datasets[i].op1 and not datasets[i].op2 and not datasets[i].op3 \
                    and not datasets[i].op4:
                warn = 1
        if warn == 1:
            reply = QMessageBox.question(self.DataUI, 'Aviso',
                                         u"No se han añadido medidas a uno o varios datasets,"
                                         u" ¿quiere continuar de todas formas?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

            if reply == QMessageBox.No:
                return
            else:
                pass

        if not fW:
            self.FoldUI = FoldsW(self)
            fW = True
        if fW:
            self.DataUI.hide()
            self.btn1.setEnabled(False)
            self.btn2.setEnabled(True)

            self.FoldUI.btn2.clicked.connect(self.restart)
            self.FoldUI.btn1.clicked.connect(partial(self.FoldUI.partsave, 2))
            self.FoldUI.btn3.clicked.connect(partial(self.FoldUI.partsave, 1))
            self.FoldUI.btn4.clicked.connect(self.fold_xml)
            self.FoldUI.btn5.clicked.connect(self.startClassifTab)

            self.FoldUI.show()
        if cW:
            self.ClassifUI.close()
        if xW:
            xmlUI.close()

    def fold_xml(self):

        self.FoldUI.folds()

        if startxml:
            self.FoldUI.close()
            self.startxmlTab()

    def startClassifTab(self):

        global cW

        for i in range(0, len(datasets)):
            if not len(datasets[i].estratif) > 0:
                QMessageBox.about(self.FoldUI, "Aviso", u"Aviso: No se han añadido estratificados "
                                                        u"a uno o varios datasets")
                return

        if not cW:
            self.ClassifUI = ClassifW(self)
            cW = True
        if cW:
            self.FoldUI.hide()
            self.btn1.setEnabled(False)
            self.btn2.setEnabled(False)
            self.btn3.setEnabled(True)
            self.btn4.setEnabled(True)

            self.ClassifUI.btn2.clicked.connect(self.restart)
            self.ClassifUI.btn3.clicked.connect(self.ClassifUI.confmethods)
            self.ClassifUI.methods.activated.connect(self.ClassifUI.signalDisable)
            self.ClassifUI.btn4.clicked.connect(self.clasifxml)
            self.ClassifUI.btn5.clicked.connect(self.startxmlTab)
            self.ClassifUI.show()
        if xW:
            xmlUI.close()

    def clasifxml(self):
        self.ClassifUI.getXml()

        if startxml:
            self.ClassifUI.close()
            cw = False
            self.startxmlTab()

    def startxmlTab(self):

        global xW, dsW, fW, cW
        if not xW:
            global xmlUI
            xmlUI = XmlW(self)
            xW = True

        if xmlname == '' and cW:
            # Guardar las ops antes d camb d ventana
            ret = self.ClassifUI.getXml()
            if ret == 'error':
                return
        if dsW:
            self.DataUI.close()
            dsW = False
        if fW:
            self.FoldUI.close()
            fw = False
        if cW:
            self.ClassifUI.close()
            cW = False

        if xW:
            self.btn1.setEnabled(False)
            self.btn2.setEnabled(False)
            self.btn3.setEnabled(False)
            self.btn4.setEnabled(True)
            xmlUI.load.setText(xmlname)
            xmlUI.btn1.clicked.connect(xmlUI.getxmlfile)
            xmlUI.btn2.clicked.connect(xmlUI.getWorkingDir)
            xmlUI.btn3.clicked.connect(xmlUI.execute)
            xmlUI.btn4.clicked.connect(self.restart)
            xmlUI.show()

    def restart(self):
        global file, text, nfolds, classif, dir, xmlname, datasets, xmlUI, dsW, fW, cW, xW, p1

        if not dir == '' and os.path.isdir(dir + '/tmp/'):  # Control de errores aplicado para que esto sea posible
            shutil.rmtree(dir + '/tmp/')

        if dsW:
            # La ventana de Dataset está/ha estado abierta
            self.DataUI.close()

        if fW:
            self.FoldUI.close()

        if xW:
            for i in range(0, len(xmlUI.threadPool)):
                xmlUI.threadPool[i].stop()
            xmlUI.btn3.setEnabled(True)
            xmlUI.close()

        if cW:
            self.ClassifUI.close()

        self.btn1.setEnabled(True)
        self.btn2.setEnabled(False)
        self.btn3.setEnabled(False)
        self.btn4.setEnabled(True)

        file = ''
        text = str('')
        nfolds = 0
        classif = []
        dir = './'
        xmlname = ''
        datasets = []  # Contendra elems de la clase dset, estratif y metodo
        xmlUI = None

        dsW = False
        fW = False
        cW = False
        xW = False

        p1 = False

    def closeEvent(self, event):  # Redefinimos el evento de cierre (pedir confirmacion)

        reply = QMessageBox.question(self, 'Aviso',
                                     u"¿Está seguro de que desea salir?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            self.restart()
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
