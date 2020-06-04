import dlib
import cv2
import time
from keras.models import load_model
import numpy as np

def show_emotion(frame,x,y,w,h,emotions,prob,predicted_e,show_high):
    font = cv2.FONT_ITALIC#cv2.FONT_HERSHEY_SIMPLEX
    #Drawing rectangle and showing output values on frame
    cv2.rectangle(frame, (x, y), (x + w, y + h),(155,155, 0),2)
    text_distance=15
    if show_high:#predicted_e
        cv2.putText(frame," ",(x,y+h+text_distance),font,0.5,(0,0,255),1,cv2.LINE_AA)
        #cv2.putText(frame,predicted_e,(x,y+h+text_distance),font,0.5,(0, 0, 255),1,cv2.LINE_AA)
    else:
        #Angry 
        for index in range(7):
            cv2.putText(frame, emotions[index] + ':', (x, y + h+text_distance), font,0.5, (0, 0, 255), 1, cv2.LINE_AA)
            text_distance+=15
                
        #bar 92%
        y0 = 8
        for score in prob.astype('int'):
            cv2.rectangle(frame,(x+75, y + h+y0), (x+75 + score,y + h+y0+ 9),(0, 255, 255), cv2.FILLED)
            cv2.putText(frame, str(score) + '% ',(x+ 80 +score, y + h+ y0 + 9),font, 0.5, (0, 0, 255),1, cv2.LINE_AA)
            y0 += 15

def preprocess(gray,x,y,w,h,num):
    cropped_face = gray[y:y + h,x:x + w]
    test_image = cv2.resize(cropped_face, (48, 48))
    #cv2.imshow('cropped {0:d}'.format(num), cropped_face)
    #pre-processing
    test_image = test_image.astype("float") / 255.0
    test_image=np.asarray(test_image)
    print('(48,48)：',test_image.shape)
    test_image = np.stack((test_image,)*3, axis=-1)
    test_image = np.expand_dims(test_image, axis=0)
    print('(1,48,48,3)：',test_image.shape)

    return test_image

def predict_emotion(img_path,detector,model):

    
    #Dictionary for emotion recognition model output and emotions
    emotions = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

    # load input image
    #frame = cv2.imread(img_path)
    #gray = cv2.imread(img_path,0)
    frame = cv2.resize(cv2.imread(img_path),(550,750))
    gray = cv2.resize(cv2.imread(img_path,0),(550,750))

    if frame is None:
        print("Could not read input image")
        exit()
    
    # apply face detection (hog)
    faces_hog = detector(gray,1)
    num=1
    # loop over detected faces
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        test_image= preprocess(gray,x,y,w,h,num)
        num+=1
        # Probablities of all classes
        #Finding class probability takes approx 0.05 seconds
        start_time = time.time()
        probab = model.predict(test_image)[0] * 100
                
        #Finding label from probabilities
        #Class having highest probability considered output label
        label = np.argmax(probab)
        probab_predicted = int(probab[label]) #最高機率的情緒數字
        predicted_emotion = emotions[label] #情緒數字轉成英文
            
        #show the result on the picture
        show_emotion(frame,x,y,w,h,emotions,probab,predicted_emotion,show_high=True)
    

    return frame

def main():
    #Creating objects for face and emotiction detection 
    img_path = './fer2013_/detection/me_happy.jpg'#me_sad_3.jpg
    # initialize hog + svm based face detector
    face_detector = dlib.get_frontal_face_detector() 
    emotion_model = load_model('./fer2013_/weights/vggface_batch16_lr0.000050.h5')#vgg16_batch16_lr0.000100.h5
    frame=predict_emotion(img_path,face_detector,emotion_model)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyWindow()
if __name__ == '__main__':
    main()