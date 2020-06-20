import dlib
import cv2
import time
from keras.models import load_model
import numpy as np

def show_emotion(frame,x,y,w,h,emotions,prob,predicted_e,show_high):
    font = cv2.FONT_ITALIC#cv2.FONT_HERSHEY_SIMPLEX
    #Drawing rectangle and showing output values on frame
    cv2.rectangle(frame, (x, y), (x + w, y + h),(125,125, 255),2)
    text_distance=15
    if show_high:
        cv2.putText(frame,predicted_e,(x,y+h+text_distance),font,0.5,(255,255,255),1,cv2.LINE_AA)
        #cv2.putText(frame,predicted_e,(x,y+h+text_distance),font,0.5,(0, 0, 255),1,cv2.LINE_AA)
    else:
        #Angry 
        for index in range(7):
            cv2.putText(frame, emotions[index] + ':', (x, y + h+text_distance), font,0.5, (255,255,255), 1, cv2.LINE_AA)
            text_distance+=15
                
        #bar 92%
        y0 = 8
        for score in prob.astype('int'):
            cv2.rectangle(frame,(x+75, y + h+y0), (x+75 + score,y + h+y0+ 9),(128, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(score) + '% ',(x+ 80 +score, y + h+ y0 + 9),font, 0.5, (255,255,255),1, cv2.LINE_AA)
            y0 += 15

def preprocess(gray,x,y,w,h,num):
    cropped_face = gray[y:y + h,x:x + w]
    test_image = cv2.resize(cropped_face, (48, 48))
    #cv2.imshow('cropped {0:d}'.format(num), cropped_face)
    
    ####pre-processing
    test_image = test_image.astype("float") / 255.0
    test_image=np.asarray(test_image)
    #print('(48,48):',test_image.shape)
    test_image = np.stack((test_image,)*3, axis=-1)
    test_image = np.expand_dims(test_image, axis=0)
    #print('(1,48,48,3):',test_image.shape)

    return test_image

def predict_emotion(frame,gray,detector,model):

    
    #Dictionary for emotion recognition model output and emotions
    emotions = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
    
    # apply face detection (hog)
    faces_hog = detector(gray,1)
    try: ####是否有偵測到圖片中的人臉
        faces_hog[0]
    except Exception as e:
        print("face count Warning: {}".format(e))
        info = 'No face detected !' 
        return frame,info

    num=1
    # loop over detected faces
    for face in faces_hog:
        x = abs(face.left())  ##有遇過出現負值的現象 所以加abs
        y = abs(face.top())
        w = abs(face.right()) - x
        h = abs(face.bottom()) - y
        test_image= preprocess(gray,x,y,w,h,num)
        num+=1
        # Probablities of all classes
        #Finding class probability takes approx 0.05 seconds
        #start_time = time.time()
        probab = model.predict(test_image)[0] * 100 #每個情緒類別預測百分比
                
        #Finding label from probabilities
        #Class having highest probability considered output label
        label = np.argmax(probab) #取百分比最大的情緒 (3)
        probab_predicted = int(probab[label]) #最高機率的情緒值 (99)
        predicted_emotion = emotions[label] #情緒數字轉成英文 (3→Happy)
            
        #show the result on the picture
        show_emotion(frame,x,y,w,h,emotions,probab,predicted_emotion,show_high=False)
    
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    # cv2.destroyWindow()
    info='OK'
    return frame,info

def start(img,gray):

    face_detector = dlib.get_frontal_face_detector() 
    emotion_model = load_model('./fer_ui/vgg16_batch16.h5')#vgg16_batch16_lr0.000100.h5
    frame,info=predict_emotion(img,gray,face_detector,emotion_model)

    return frame,info
'''
if __name__ == '__main__':
    main()
'''