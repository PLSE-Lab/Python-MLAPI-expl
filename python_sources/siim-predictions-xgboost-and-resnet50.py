#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # TRAIN

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
train.head()


# In[ ]:


train.sort_values('image_name',ascending=True,inplace=True)
train.head()


# In[ ]:


train.info()


# In[ ]:


train['sex'].fillna(train['sex'].mode()[0],inplace=True)
train['age_approx'].fillna(train['age_approx'].mean(),inplace=True)
train['anatom_site_general_challenge'].fillna(train['anatom_site_general_challenge'].mode()[0],inplace=True)


# In[ ]:


train.anatom_site_general_challenge.value_counts()


# In[ ]:


corr = train.corr()
corr.style.background_gradient(cmap='inferno')


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train.sex)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train.anatom_site_general_challenge)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train.diagnosis)


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(y="benign_malignant",data=train)


# # Deep Learning

# In[ ]:


train.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
labels=lab.fit_transform(train.benign_malignant)


# In[ ]:


import glob
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
image_array=[]
l=[]
i=0
for img in tqdm(glob.glob("../input/siim-isic-melanoma-classification/jpeg/train/*.jpg")):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    image_array.append(np.array(size_image))
    i=i+1
    if i==1000:
        break


# In[ ]:


len(image_array)


# In[ ]:


data=np.array(image_array)
labels=labels[:1000]


# In[ ]:


len(labels),np.unique(labels)


# In[ ]:


import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,10))
ax=figure.add_subplot(121)
ax.imshow(data[0])
bx=figure.add_subplot(122)
bx.imshow(data[60])
plt.show()


# In[ ]:


np.save("Cells",data)
np.save("labels",labels)


# In[ ]:


Cells=np.load("Cells.npy")
labels=np.load("labels.npy")


# In[ ]:


s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]


# In[ ]:


num_classes=len(np.unique(labels))
len_data=len(Cells)
num_classes


# In[ ]:


x_train,x_test=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)


# In[ ]:


y_train,y_test=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]


# In[ ]:


import keras
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)


# In[ ]:


from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50


# In[ ]:


base_model = ResNet50(weights='imagenet',include_top=False, input_shape=(50,50,3))
x = base_model.output
x = Flatten()(x)
x=Dense(500, activation='relu')(x)
x=Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1,validation_split=0.33,callbacks=[checkpoint])


# In[ ]:


scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])


# In[ ]:


test_image_array=[]
for img in tqdm(glob.glob("../input/siim-isic-melanoma-classification/jpeg/test/*.jpg")):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    test_image_array.append(np.array(size_image))
np.save("test",test_image_array)


# In[ ]:


ts_array=np.load("../input/siimpred/test.npy")
p=np.argmax(model.predict(ts_array),axis=1)


# In[ ]:


a=lab.inverse_transform(p)
a=pd.DataFrame(a)


# In[ ]:


a.to_csv("benign_malignant.csv",index=False)


# # Machine Learning

# In[ ]:


from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
train['benign_malignant']=la.fit_transform(train['benign_malignant'])
train['diagnosis']=la.fit_transform(train['diagnosis'])
train['anatom_site_general_challenge']=la.fit_transform(train['anatom_site_general_challenge'])
train['sex']=la.fit_transform(train['sex'])


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


x=train.drop(['image_name','patient_id','target','diagnosis'],axis=1)
y=train['target']


# In[ ]:


from imblearn.over_sampling import SMOTE
sm=SMOTE()
x,y=sm.fit_sample(x,y)


# # Machine Learning

# In[ ]:


from sklearn.model_selection import train_test_split,KFold,cross_val_score
xr,xt,yr,yt=train_test_split(x,y)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor,XGBRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor,LGBMRegressor


# In[ ]:


model=XGBClassifier(n_estimators=1000)
model.fit(x,y)
# kfold=KFold(n_splits=10)
# res=cross_val_score(model,x,y,cv=kfold)
# res.mean()*100


# In[ ]:


yp=model.predict(xt)


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(r2_score(yt,yp))
print(mean_absolute_error(yt,yp))
print(mean_squared_error(yt,yp))


# In[ ]:


sample=pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")
sample.head()


# # TEST

# In[ ]:


test=pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
test.head()


# In[ ]:


test['sex'].fillna(test['sex'].mode()[0],inplace=True)
test['age_approx'].fillna(test['age_approx'].mean(),inplace=True)
test['anatom_site_general_challenge'].fillna(test['anatom_site_general_challenge'].mode()[0],inplace=True)


# In[ ]:


test['anatom_site_general_challenge']=la.fit_transform(test['anatom_site_general_challenge'])
test['sex']=la.fit_transform(test['sex'])


# In[ ]:


test.head()


# In[ ]:


test=pd.concat([test,a],axis=1)
test.head()


# In[ ]:


test.columns=[                   'image_name',                    'patient_id',
                                 'sex',                    'age_approx',
       'anatom_site_general_challenge',                               'benign_malignant']


# In[ ]:


test['benign_malignant']=la.fit_transform(test['benign_malignant'])


# In[ ]:


test.info()


# In[ ]:


x1=test.drop(['image_name','patient_id'],axis=1)
yp1=model.predict(x1)
yp1=pd.DataFrame(yp1)


# In[ ]:


test=pd.concat([test,yp1],axis=1)
test.head()


# In[ ]:


test.columns=[                   'image_name',                    'patient_id',
                                 'sex',                    'age_approx',
       'anatom_site_general_challenge',              'benign_malignant',
                                     'target']


# In[ ]:


test.to_csv('sub.csv',columns=['image_name','target'],index=False)

