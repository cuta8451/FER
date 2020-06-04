# -*- coding: utf-8 -*-
import sys
import os
import numpy as np 
import pandas as pd
import time
import cv2

#keras
from keras.utils import to_categorical

# fer2013 dataset:
# Training       28709
# PrivateTest     3589
# PublicTest      3589 total 35887


#讀取資料，
def read_csv(path_,fname_):
    
    #dir_path = 'D:/python/fer2013_/data'
    file_= os.path.join(path_,fname_).replace('\\','/')
    data = pd.read_csv(file_)
    num_of_instances = len(data) #獲取數據集的數量
    print('number of instances',num_of_instances)
    
    #提取pixels,emotions,usages該columns的全部值
    pixels = data[" pixels"] 
    emotions = data['emotion']
    usages = data[" Usage"]
    
    return pixels,emotions,usages

#將訓練、測試集分開
def seperate_data(pixels_,emotions_,usages_):

    num_classes = 7   #表情有七類
    x_train,y_train,x_test,y_test,y_test_label,x_vali,y_vali= [],[],[],[],[],[],[]
    
    for emotion,img,usage in zip(emotions_,pixels_,usages_):    #同時遍歷多個數組或列表時，可用zip()函數進行遍歷
    
        emotion_one = to_categorical(emotion,num_classes)   # 將七類標籤轉成獨熱向量編碼one-hot encoding
        val = img.split(' ')
        pixels = np.array(val,'float32')
        
        if(usage == 'Training'): #or usage == 'PrivateTest'
            x_train.append(pixels)
            y_train.append(emotion_one)
        elif(usage == 'PrivateTest'):
            x_test.append(pixels)
            y_test.append(emotion_one)
            y_test_label.append(emotion)
        else:
            x_vali.append(pixels)
            y_vali.append(emotion_one)
        
    return x_train,y_train,x_test,y_test,y_test_label,x_vali,y_vali

#將原圖48*48調整成224*224,符合VGGFACE的模型輸入大小
def resize_Img(train_):

    pixels_resize=[]
    for num_ in range(train_.shape[0]):
    
        if(num_ % 1000 == 0):
            print('now resize 48 --> 224 number',num_)

        pixels_resize.append(cv2.resize(train_[num_], (224, 224), interpolation=cv2.INTER_LINEAR))
    
    #(224,224)→(224,224,3)，也就是每個灰階像素copy三次到三個channels中
    #stacked_img = np.stack((pixels_resize,)*3, axis=-1)
    pixels_=np.array(pixels_resize,'float32')
    #pixels_resize= np.dstack((pixels_,) * 3)
    return pixels_resize

#x/y_train, x/y_test 轉換成numpy數組格式，並呼叫resize_Img()調成VGGface的224*224大小,方便後續處理
def transfer_numpy(xtrain_,ytrain_,xtest_,ytest_,xvali_,yvali_):
    
    #Training Set轉np.array → 48*48變成224*224
    x_train = np.array(xtrain_)
    x_train = x_train.reshape(-1,48,48)#,1
    #x_train= (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
    #x_train_255 = x_train/255
    x_train = np.stack((x_train,)*3, axis=-1)#變成3channel
    #x_train_resize=resize_Img(x_train)
    y_train = np.array(ytrain_)
    
    #Testing Set轉np.array → 48*48變成224*224
    x_test = np.array(xtest_)
    x_test = x_test.reshape(-1,48,48)#,1
    #x_test= (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
    #x_test_255 = x_test/255
    x_test = np.stack((x_test,)*3, axis=-1)
    #x_test_resize=resize_Img(x_test)
    y_test = np.array(ytest_)
    #cv2.imshow("original",x_train[2050].astype(np.uint8))
    #cv2.imshow("resize",x_train_resize[2050].astype(np.uint8))
    #cv2.waitKey(0)

    x_vali = np.array(xvali_)
    x_vali = x_vali.reshape(-1,48,48)#,1
    #x_vali= (x_vali-np.min(x_vali))/(np.max(x_vali)-np.min(x_vali))
    #x_vali_255 = x_vali/255
    x_vali = np.stack((x_vali,)*3, axis=-1)
    #x_vali_resize=resize_Img(x_vali)
    y_vali = np.array(yvali_)
    return x_train,y_train,x_test,y_test,x_vali,y_vali#x_train_resize,y_train,x_test_resize,y_test
    
def main():
    dir_path = 'D:/wendy/fer2013_/data'
    pixels,emotions,usages=read_csv(dir_path,'icml_face_data.csv')
    x_train,y_train,x_test,y_test,y_test_label,x_vali,y_vali=seperate_data(pixels,emotions,usages)
    x_train,y_train,x_test,y_test,x_vali,y_vali=transfer_numpy(x_train,y_train,x_test,y_test,x_vali,y_vali)
    
    #存成npz檔
    np.savez('./fer2013_/npz/vali_ch3_label_pri.npz',x_train,y_train,x_test,y_test,y_test_label,x_vali,y_vali) # arr_0:train_pixels / arr_1:train_emotion / arr_2:test_pixels / arr_3:test_emotion  
    
    return

if __name__=="__main__":
    start=time.clock()
    main()
    end=time.clock()
    print("total spend:",(end-start))