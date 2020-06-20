# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fer.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(762, 639)
        MainWindow.setStyleSheet("background-color:rgb(255,255,128)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Image_Show = QtWidgets.QLabel(self.centralwidget)
        self.Image_Show.setGeometry(QtCore.QRect(30, 20, 521, 620))
        self.Image_Show.setStyleSheet("border-radius:30px;")
        self.Image_Show.setText("")
        self.Image_Show.setObjectName("Image_Show")
        #####自己+++
        self.movie = QtGui.QMovie('./fer_ui/img_start4.gif') 
        self.Image_Show.setMovie(self.movie)
        self.Image_Show.setAlignment(QtCore.Qt.AlignCenter)
        self.movie.start()
        self.messagebox = QtWidgets.QMessageBox()
        self.messagebox.setStyleSheet(
        "QMessageBox{ background-color:rgb(255,255,128)}"
        "QPushButton { color: white; background-color:rgb(125,125,255) }"
        )
        #####
        self.Button_Load = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Load.setGeometry(QtCore.QRect(580, 70, 151, 231))
        font = QtGui.QFont()
        font.setFamily("Bookman Old Style")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.Button_Load.setFont(font)
        self.Button_Load.setAutoFillBackground(False)
        self.Button_Load.setStyleSheet("border-radius:30px;\n"
"background-color:rgb(125,125,255);\n"
"color:white;")
        self.Button_Load.setObjectName("Button_Load")
        self.Button_Predict = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Predict.setGeometry(QtCore.QRect(580, 330, 151, 231))
        font = QtGui.QFont()
        font.setFamily("Bookman Old Style")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.Button_Predict.setFont(font)
        self.Button_Predict.setStyleSheet("border-radius:30px;\n"
"background-color:rgb(125,125,255);\n"
"color:white;")
        self.Button_Predict.setObjectName("Button_Predict")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "FER"))
        self.Button_Load.setText(_translate("MainWindow", "Load "))
        self.Button_Predict.setText(_translate("MainWindow", "Predict"))
        self.messagebox.setWindowTitle(_translate("MainWindow", "INFO")) ## 自己++

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

