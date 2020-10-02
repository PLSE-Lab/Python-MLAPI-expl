#!/usr/bin/env python
# coding: utf-8

# **If you found this helpful, please *upvote* this kerel and the associated dataset.**

# Feel free to use my "DeepFakes" dataset. It includes 11 chunks of training data(only include cropped faces) and I will add new ones once in a while. **BEFORE YOU USE THE DATASET ATACHED TO THIS KERNEL, YOU HAVE TO AGREE THE DEEPFAKE COMPETITION RULE.**

# Edit: 
# 1. Change CNN More Similar To MesoNet
# 2. Add more data
# 3. Did some hyperparameter tuning

# **Install MTCNN**

# In[ ]:


get_ipython().system('pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl')


# **Import Libraries**

# In[ ]:


import pandas as pd
import keras
import os
import numpy as np
from sklearn.metrics import log_loss
from keras import Sequential
from keras.layers import *
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from mtcnn import MTCNN
from tqdm.notebook import tqdm


# # Load Train Data

# In[ ]:


df_train0 = pd.read_json('../input/deepfakes/metadata0.json')
df_train1 = pd.read_json('../input/deepfakes/metadata1.json')
df_train2 = pd.read_json('../input/deepfakes/metadata2.json')
df_train3 = pd.read_json('../input/deepfakes/metadata3.json')
df_train4 = pd.read_json('../input/deepfakes/metadata4.json')
df_train5 = pd.read_json('../input/deepfakes/metadata5.json')
df_train6 = pd.read_json('../input/deepfakes/metadata6.json')
df_train7 = pd.read_json('../input/deepfakes/metadata7.json')
df_train8 = pd.read_json('../input/deepfakes/metadata8.json')
df_train9 = pd.read_json('../input/deepfakes/metadata9.json')
df_train10 = pd.read_json('../input/deepfakes/metadata10.json')
df_train11 = pd.read_json('../input/deepfakes/metadata11.json')
df_train12 = pd.read_json('../input/deepfakes/metadata12.json')
df_train13 = pd.read_json('../input/deepfakes/metadata13.json')
df_train14 = pd.read_json('../input/deepfakes/metadata14.json')
df_train15 = pd.read_json('../input/deepfakes/metadata15.json')
df_train16 = pd.read_json('../input/deepfakes/metadata16.json')
df_train17 = pd.read_json('../input/deepfakes/metadata17.json')
df_train18 = pd.read_json('../input/deepfakes/metadata18.json')
df_train19 = pd.read_json('../input/deepfakes/metadata19.json')
LABELS = ['REAL','FAKE']
df_trains = [df_train0 ,df_train1, df_train2, df_train3, df_train4,
             df_train5, df_train6, df_train7, df_train8, df_train9,
             df_train10, df_train11, df_train12, df_train13, df_train14,
             df_train15, df_train16,df_train17,df_train18,df_train19]
nums = list(range(len(df_trains)))


# In[ ]:


from tqdm import tqdm_notebook
def read_image(num,name):
    num=str(num)
    if len(num)==2:
        path='../input/deepfakes/DeepFake'+num+'/DeepFake'+num+'/' + x.replace('.mp4', '') + '.jpg'
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    else:
        path='../input/deepfakes/DeepFake0'+num+'/DeepFake0'+num+'/' + x.replace('.mp4', '') + '.jpg'
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
X = []
y = []
for df_train,num in tqdm_notebook(zip(df_trains,nums),total=len(df_trains)):
    images = list(df_train.columns.values)
    for x in images:
        try:
            X.append(read_image(num,x))
            y.append(LABELS.index(df_train[x]['label']))
        except Exception as err:
            print(x)


# # Apply Underbalancing Techinique

# In[ ]:


print('There are '+str(y.count(1))+' fake samples')
print('There are '+str(y.count(0))+' real samples')


# The data is not balanced. We are going to use the undersampling technique.

# In[ ]:


import random
real=[]
fake=[]
for m,n in zip(X,y):
    if n==0:
        real.append(m)
    else:
        fake.append(m)
fake=random.sample(fake,len(real))
X,y=[],[]
for x in real:
    X.append(x)
    y.append(0)
for x in fake:
    X.append(x)
    y.append(1)


# In[ ]:


print('There are '+str(y.count(1))+' fake samples')
print('There are '+str(y.count(0))+' real samples')


# Now, the data is balanced.

# In[ ]:


train_X,val_X,train_y,val_y = train_test_split(X, y, test_size=0.15,shuffle=True)


# # Define Model

# In[ ]:


def define_model():
    model = Sequential(
        [
            Conv2D(8, (3, 3), padding="same", activation = 'elu', input_shape=(92, 92,3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(8, (5, 5), padding="same", activation = 'elu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(16, (5, 5), padding="same", activation = 'elu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(16, (5, 5), padding="same", activation = 'elu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(16,activation='relu'),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=5e-5))
    model.summary()
    return model


# This model is a modified version of MesoNet

# # Train Model

# In[ ]:


model=define_model()
model.fit([train_X],[train_y],epochs=7)


# In[ ]:


model.fit([train_X],[train_y],epochs=7)


# **Check Validation Log Loss**

# In[ ]:


answer=[LABELS[n] for n in val_y]
pred=np.random.random(len(val_X))
print('random loss: ' + str(log_loss(answer,pred.clip(0.0001,0.99999))))
pred=np.array([1 for _ in range(len(val_X))])
print('1 loss: ' + str(log_loss(answer,pred)))
pred=np.array([0 for _ in range(len(val_X))])
print('0 loss: ' + str(log_loss(answer,pred)))
pred=np.array([0.5 for _ in range(len(val_X))])
print('0.5 loss: ' + str(log_loss(answer,pred)))


# In[ ]:


pred=model.predict([val_X])
print('model loss: '+str(log_loss(answer,pred.clip(0.1,0.9))))


# This is not looking good. The model learned to try to be closer to 0.5.

# # Save Model

# In[ ]:


model.save('model.h5')


# # Make submission

# In[ ]:


test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'
filenames=os.listdir(test_dir)
test_video_files = [test_dir + x for x in filenames]
detector = MTCNN()
def detect_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final = []
    detected_faces_raw = detector.detect_faces(img)
    if detected_faces_raw == []:
        print('no faces found, skip to next frame')
        return []
    for x in detected_faces_raw:
        x, y, w, h = x['box']
        final.append([x, y, w, h])
    return final
def crop(img, x, y, w, h):
    x -= 40
    y -= 40
    w += 40
    h += 40
    if x < 0:
        x = 0
    if y <= 0:
        y = 0
    return cv2.cvtColor(cv2.resize(img[y: y + h, x: x + w], (92, 92)), cv2.COLOR_BGR2RGB)
def detect_video(video):
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    while True:
        ret, frame = cap.read()
        bounding_box = detect_face(frame)
        if bounding_box == []:
            continue
        x, y, w, h = bounding_box[0]
        return crop(frame, x, y, w, h)
test_X = []
for video in tqdm(test_video_files):
    test_X.append(detect_video(video))


# In[ ]:


df_test=pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')
pred=model.predict([test_X]).clip(0.1,0.9)
df_test['label']=pred
df_test['filename']=filenames


# In[ ]:


df_test.head()


# In[ ]:


df_test.to_csv('submission.csv',index=False)


# # Further Work
# 1. Do some more hyperparamater tuning
# 2. Train on the whole video(and maybe also sound)
# 3. Try LSTM-CNN
# 4. K Folds(I will try it later when I upload more data)
