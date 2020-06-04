import numpy as np
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#VGG16
from keras.applications.vgg16 import VGG16

#VGG_Face
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten,Dense,Dropout
from keras.optimizers import Adam


from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger,ReduceLROnPlateau,TensorBoard
from sklearn.metrics import confusion_matrix,classification_report
from keras.models import load_model
import itertools
import matplotlib.pyplot as plt 
import cv2
import time
import os

def read_Data():
    #ferdata_vali_ch3_label
    data= np.load('./fer2013_/npz/vali_ch3_label_pri.npz')#arr_0:train_pixels / arr_1:train_emotion / arr_2:test_pixels / arr_3:test_emotion  
    #trainPixels,trainEmotion,testPixels,testEmotion=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
    #trainPixels,trainEmotion,testPixels,testEmotion,test_label = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3'],data['arr_4']
    trainPixels,trainEmotion,testPixels,testEmotion,test_label,valiPixels,valiEmotion = data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3'],data['arr_4'],data['arr_5'],data['arr_6']
    #cv2.imshow("try",testPixels[2050].astype(np.uint8))
    #cv2.waitKey(0)
    #print('255 shape',trainPixels.shape,trainEmotion.shape,testPixels.shape,testEmotion.shape,test_label.shape,valiPixels.shape,valiEmotion.shape)#,test_label.shape
    return trainPixels,trainEmotion,testPixels,testEmotion,test_label,valiPixels,valiEmotion#,test_label,valiPixels,valiEmotion 

def callbacks(batch,lr):
    #callbacks
    #EarlyStopping:當指定的評量數據(acc、loss)停止進步,則中斷訓練。
    #ModelCheckpoint:在指定時刻,將神經網路存起來。
    #ReduceLROnPlateau:當指定的評量數據(acc、loss)停止進步,則降低LR。
    #CSVLogger:將每個訓練週期的評量數據,存到本地端

    early_stopping_=EarlyStopping(monitor='val_loss',min_delta=0,patience=20,verbose=1,mode='auto',baseline=None)
    model_checkpoint_=ModelCheckpoint(filepath='./fer2013_/weights/vgg16_frozen_batch{0:d}_lr{1:f}'.format(batch,lr)+'_{epoch:02d}-{val_acc:.4f}.h5',monitor='val_acc',save_best_only=True,verbose=1)
    csv_logger_=CSVLogger(filename='./fer2013_/weights/vgg16_frozen_batch{0:d}_lr{1:f}_log.csv'.format(batch,lr),separator=',',append=False)
    ReduceLROnPlateau_=ReduceLROnPlateau(monitor='val_loss',factor=0.4,patience=4,verbose=1,min_delta=1e-6)
    tensorboard_=TensorBoard(log_dir='./fer2013_/weights', histogram_freq=1)

    return early_stopping_,model_checkpoint_,csv_logger_,ReduceLROnPlateau_,tensorboard_#,ReduceLROnPlateau_,tensorboard_

def VGG16_structure(): #VGG16架構
    # Convolution Features

    base_model = VGG16(weights='imagenet', include_top=False,input_shape=(48,48,3))
    last_layer = base_model.get_layer('block5_pool').output#block5_pool

    # 新分類器 
    x = Flatten(name='flatten')(last_layer)
    #x=  Dropout(0.35)(x)
    x = Dense(512, activation='relu', name='fc6')(x)
    #x=  Dropout(0.35)(x)
    #x = Dense(256, activation='relu', name='fc7')(x) #later try 256
    #x=  Dropout(0.35)(x)
    out = Dense(7, activation='softmax', name='fc7')(x)
    model = Model(base_model.input, out)
    # 首先，我们只训练顶部的几层（随机初始化的层）
    # 锁住所有 InceptionV3 的卷积层
    
    #凍結權重
    #base_model.trainable=False
    
    for layer in base_model.layers:
        layer.trainable=False
    
    model.summary()

    return model

def VGGface_structure(): #VGGFACE架構
    # Convolution Features

    base_model = VGGFace(include_top=False, input_shape=(48, 48, 3), pooling='avg',weights='vggface')
    last_layer = base_model.get_layer('pool5').output#

    # 新分類器 
    x = Flatten(name='flatten')(last_layer)
    x = Dense(512, activation='relu', name='fc6')(x)
    #x=  Dropout(0.25)(x)
    #x = Dense(512, activation='relu', name='fc7')(x) #later try 256
    #x=  Dropout(0.25)(x)
    out = Dense(7, activation='softmax', name='fc7')(x)
    model = Model(base_model.input, out)
    
    #base_model.trainable=False
    
    #凍結權重
    for layer in base_model.layers:
        layer.trainable=False
    
    model.summary()

    return model

def image_generator(trainpixels_,trainemotion_,valipixels_,valiemotion_,testpixels_,testemotion_,batchsize_,epochs_,learningrate_):
    #呼叫callbacks
    early_stopping,model_checkpoint,csv_logger,reducelr,tensorboard=callbacks(batchsize_,learningrate_)#,reducelr
    fclist=[early_stopping,reducelr,csv_logger,model_checkpoint]
    callbacklist=[early_stopping,csv_logger,model_checkpoint,reducelr] #early_stopping,model_checkpoint,csv_logger,reducelr
    
    #Generator  
    train=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
    trn_gen=train.flow(x=trainpixels_,y=trainemotion_,batch_size=batchsize_)#,save_to_dir='./fer2013_/pic',save_format='jpg'
    vali = ImageDataGenerator(rescale=1./255)#
    vali_gen=vali.flow(x=valipixels_,y=valiemotion_,batch_size=batchsize_)
    test = ImageDataGenerator(rescale=1./255)#
    test_gen=test.flow(x=testpixels_,y=testemotion_,batch_size=batchsize_)#,save_to_dir='./fer2013_/pic',save_format='jpg'
    

    model=VGG16_structure()#VGGface_structure()
    model.compile(loss = 'categorical_crossentropy',optimizer = Adam(lr=learningrate_),metrics=['accuracy'])
    
    #單純train FC layer
    model.fit_generator(trn_gen,steps_per_epoch=len(trainpixels_)//batchsize_,validation_data=vali_gen,validation_steps=len(valipixels_)//batchsize_,epochs=epochs_,callbacks=fclist)#
    #model.save('./fer2013_/weights/vggface_batch{0:d}_lr{1:f}.h5'.format(batchsize_,learningrate_))
    '''
    #block3_conv1是VGG16， conv3_1是VGGFace
    unfreeze=['conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7']
    #unfreeze = ['block3_conv1','block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1','block5_conv2','block5_conv3','fc6','fc7']
    for layer in model.layers:
        if layer.name in unfreeze:
            layer.trainable=True
        else:
            layer.trainable=False
    
    #unfreeze train
    #model.compile(loss = 'categorical_crossentropy',optimizer = Adam(lr=learningrate_),metrics=['accuracy'])
    
    history=model.fit_generator(trn_gen,steps_per_epoch=len(trainpixels_)//batchsize_,validation_data=vali_gen,validation_steps=len(valipixels_)//batchsize_,epochs=epochs_,callbacks=callbacklist)#
    '''
    model.save('./fer2013_/weights/vgg16_frozen_{0:d}_lr{1:f}.h5'.format(batchsize_,learningrate_))

    test_score = model.evaluate_generator(test_gen, steps=len(test_gen), verbose=0)
    testLoss=test_score[0]
    testAcc=100*test_score[1]
    '''
    print('vgg16_batch{0:d}_lr{1:f}_Loss_ACC \n'.format(batchsize_,learningrate_))
    print('Test loss:', test_score[0])
    print('Test accuracy:', 100*test_score[1])
    '''
    return history,testLoss,testAcc
    
def plot_AccLoss(history_,batch,lr):#畫出accracy 跟 loss

    #plot accuracy and loss of each epochs
    print(history_.history.keys())
    fig, (ax1, ax2)= plt.subplots(nrows=2,ncols=1,sharex=False,sharey=False,figsize=(10,10),constrained_layout=True) #constrained_layout自動調整圖片之間的間距
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    acc=ax1.plot(history_.history['acc'])
    vacc=ax1.plot(history_.history['val_acc'])
    ax1.set_title('model accuracy')

    # summarize history for loss 
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    loss=ax2.plot(history_.history['loss']) 
    vloss=ax2.plot(history_.history['val_loss'])
    ax2.set_title('model loss')
    
    #fig.suptitle('Accuracy and Loss of each epochs')
    fig.legend([acc,vacc,loss,vloss],labels=['Acc','Val_Acc','Loss','Val_Loss'],loc='upper right',borderaxespad=0.1) #所有子圖的圖例

    #plt.subplots_adjust(wspace=0.5) #調整子圖之間的寬距
    fig.savefig('./fer2013_/weights/vgg16_frozen_batch{0:d}_lr{1:f}.png'.format(batch,lr))
    #fig.show()

    return

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def predict(testPixels,testEmotion,test_label,batchsize_,model_path):#預測並劃出混淆矩陣
    model=load_model(model_path)
    test = ImageDataGenerator(rescale=1./255)#
    test_gen=test.flow(x=testPixels,y=testEmotion,batch_size=batchsize_,shuffle=False)
    predict=model.predict_generator(test_gen,steps=(len(testPixels)//batchsize_)+1)
    predict_class = [np.argmax(pro) for pro in predict]

    #Confusion Matrix
    cm=confusion_matrix(test_label,predict_class)
    target_names=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    #各類的precision recall f1-score
    print(classification_report(test_label,predict_class,target_names=target_names))
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cm, classes=target_names,normalize=False)

def main():
    batchsize_=[16]#8,16,32,64,128
    epochs=300
    lr=[1e-4]#1e-3,5e-4,    1e-4,5e-5,1e-5
    #trainPixels,trai1nEmotion,testPixels,testEmotion = read_Data()
    #trainPixels,trainEmotion,testPixels,testEmotion,test_label = read_Data()
    trainPixels,trainEmotion,testPixels,testEmotion,test_label,valiPixels,valiEmotion= read_Data()
    
    test_loss=[]
    test_acc=[]
    
    for batchsize in batchsize_:
        for lr_ in lr:
            history,testloss,testacc=image_generator(trainPixels,trainEmotion,valiPixels,valiEmotion,testPixels,testEmotion,batchsize,epochs,lr_)
            test_loss.append(testloss)
            test_acc.append(testacc)
            plot_AccLoss(history,batchsize,lr_)
    print(test_loss,test_acc)
    '''
    #把acc loss存入txt檔
    loss=str(test_loss[0])+', '+str(test_loss[1])+', '+str(test_loss[2])+', '+str(test_loss[3])+', '+str(test_loss[4])+'\n'+str(test_acc[0])+', '+str(test_acc[1])+', '+str(test_acc[2])+', '+str(test_acc[3])+', '+str(test_acc[4])
    with open('vggface.txt','w') as f:
        f.write(loss)
    '''
    
    '''
    #predict CF
    model_path='./fer2013_/weights/vgg16_face_batch(0602)/vgg16_private_16_lr0.000100.h5'
    predict(testPixels,testEmotion,test_label,batchsize_[0],model_path)
    '''
    return


if __name__ == "__main__":
    start=time.clock()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" #因為gpu1顯體不足,所以改換成gpu3去跑
    main()
    end=time.clock()
    print("total spend:",(end-start))
