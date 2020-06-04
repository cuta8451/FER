###### tags: `github`
# Facial Emotion Recognition
## 檔案說明(file description)
* [preprocessing.py](https://github.com/cuta8451/FER/blob/master/preprocessing.py) 
    * 預處理
* [trans_model.py](https://github.com/cuta8451/FER/blob/master/trans_model.py)
    * 模型訓練
* [dlib_picture.py](https://github.com/cuta8451/FER/blob/master/dlib_picture.py)
    * 實際圖片中的人臉情緒偵測

## 資料集說明(dataset description)
* 使用的是Kaggle上的FER2013資料集：Challenges in Representation Learning: Facial Expression Recognition
* 7種情緒分別是憤怒(Anger)，厭惡(Disgust)，恐懼(Fear)，快樂(Happy)，悲傷(Sad)，驚訝(Surprise)和中性(Neutral)，總共35887筆資料，圖片為48*48像素灰階圖。
* ![](https://i.imgur.com/SvnZAEE.png)
* ![](https://i.imgur.com/ePqq4Cm.png)

## 預處理方法 (preprocessing)
* 調整圖片大小
* 特徵縮放(0~1)
* 資料擴充:幾何
* 資料擴充GAN(ing)
## 模型 (model)
* CNN
    * VGG16
    * VGGFACE
    * MoblieNet(待做)
## 結果 (result)

* Accuracy & Loss
    * ![](https://i.imgur.com/6X5MJQ5.png)
    * ![](https://i.imgur.com/GodCBbP.png)

* Confusion Matrix
    * ![](https://i.imgur.com/VyQVZie.png)
    * ![](https://i.imgur.com/9b0NQhb.png)

* Image Testing
    * ![](https://i.imgur.com/7ZDdCyv.png)
    * ![](https://i.imgur.com/Yjib91r.png)
    * 



