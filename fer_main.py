import sys
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from fer_ui import Ui_MainWindow
from dlib_picture import start 
import numpy as np
import time

class mywindow(QtWidgets.QMainWindow):
    ## override the init function
    def __init__(self, parent = None):
        super(QMainWindow, self).__init__(parent)  ## inherit
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("FER ! ")
        self.linkEvent()
        self.show()
        

    def linkEvent(self):## add link between model and view
        
        self.ui.Button_Load.clicked.connect(lambda :self.bt_load_image())
        self.ui.Button_Predict.clicked.connect(lambda:self.bt_predict())
        return

    
    def bt_load_image(self):#bt_load_imgae event 觸發,打開文件夾選取檔案,並檢查是否有讀取,若無則顯示no file!
        
        filename=QFileDialog.getOpenFileName(self,"choose picture","./",'Image Files(*.png *.jpg)')
        self.filename_=filename[0]
        if self.filename_=="":
            #print("No file !")
            self.ui.messagebox.setText('No file !')
            self.ui.messagebox.show()
            return None
        
        self.cvImg=cv2.imread(self.filename_)
        self.cvImg_gray=cv2.imread(self.filename_,0)
        #self.cvImg=cv2.imdecode(np.fromfile(self.filename_,dtype=np.uint8),-1)
        self.cvImg=cv2.resize(self.cvImg, (450,650))
        self.cvImg_gray=cv2.resize(self.cvImg_gray, (450,650))
        #顯示圖片
        self.display_img_on_label(self.cvImg)

    
    def bt_predict(self):#bt_predict event 觸發
        
        try: ###設定如果圖片為空則先讀取圖片######
            self.cvImg
        except Exception as e:
            print("Img Warning: {}".format(e))
            self.ui.messagebox.setText("Choose picture first ! ")
            self.ui.messagebox.show()
            return None
        start_time= time.time()
        self.predict_img,self.info =start(self.cvImg,self.cvImg_gray)
        print(time.time()-start_time)
        messagelist=['No face detected !']
        if self.info in messagelist:####是否有偵測到圖片中的人臉
            self.ui.messagebox.setText(self.info)
            self.ui.messagebox.show()
        else:
            self.display_img_on_label(self.predict_img)

    
    def display_img_on_label(self,cvimg):#將圖片顯示在label上
        
        #圖片通道依灰階還彩色,轉成rgb
        if len(cvimg.shape) < 3 or cvimg.shape[2] == 1: #灰階圖
            qimg = cv2.cvtColor(cvimg, cv2.COLOR_GRAY2RGB)
        else: #彩圖
            qimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

        #image_height, image_width, image_depth = qimg.shape
        image_height, image_width, image_depth = cvimg.shape
        qimg = QImage(qimg.data, image_width, image_height,image_width * image_depth,QImage.Format_RGB888)
        self.ui.Image_Show.setPixmap(QPixmap.fromImage(qimg))

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()