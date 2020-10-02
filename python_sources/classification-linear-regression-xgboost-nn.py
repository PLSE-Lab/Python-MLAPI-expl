#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from skimage import color


# In[ ]:


import pandas as pd
import re
import random
import numpy as np
import os
import keras
get_ipython().system('pip install metrics')
import metrics
import keras.backend as k
from cv2 import imread,resize
from time import time
#from scipy.misc import imread
from sklearn.metrics import accuracy_score,normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
df_train=pd.read_csv('/content/final_train.csv')
df_test=pd.read_csv('/content/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_test.columns


# In[ ]:


print(len(df_train['anatom_site_general_challenge'].unique()))


# In[ ]:


print(len(df_train['sex'].unique()))


# In[ ]:


np.random.seed(1234)
boxplot = df_train.boxplot(column=['age_approx'])


# In[ ]:


Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1


# In[ ]:


IQR


# In[ ]:


df_train[df_train['age_approx']==0]


# In[ ]:


for i in df_train.columns:
  column=df_train[i]
  print(i,column.isna().sum())


# In[ ]:


df_train['target'][df_train['sex'].isna()]


# In[ ]:


df_train['target'][df_train['age_approx'].isna()]


# In[ ]:


def max_count(df,col_1):
  return max(set(df[col_1]), key = list(df[col_1]).count) #.unique()


# In[ ]:


max(set(df_train['target'][df_train['anatom_site_general_challenge'].isna()]), key = list(df_train['target'][df_train['anatom_site_general_challenge'].isna()]).count) #.unique()


# In[ ]:


max(set(df_train['target']), key = list(df_train['target']).count)


# In[ ]:


anatom_mx=max_count(df_train,'anatom_site_general_challenge')


# In[ ]:


sex_mx=max_count(df_train,'sex')


# In[ ]:


#index = df_train['age_approx'].index[df_train['age_approx'].apply(np.isnan)]


# In[ ]:


def replace_val(df_train,column,val):
  index = df_train[column].index[df_train[column].isna()]
  for Index in index:
    df_train[column][Index]=val
    df_train[column][Index]=val
  return df_train


# In[ ]:


sex_le = LabelEncoder()
anatom_le=LabelEncoder()


# In[ ]:




def preprocess(df_train):
  global sex_le
  global anatom_le
  df_train=replace_val(df_train,'anatom_site_general_challenge',anatom_mx)
  df_train=replace_val(df_train,'sex',sex_mx)
  q_low = df_train["age_approx"].quantile(0.01)
  q_hi  = df_train["age_approx"].quantile(0.99)

  df_filtered = df_train[(df_train["age_approx"] < q_hi) & (df_train["age_approx"] > q_low)]
  
  try:
      df_filtered=df_filtered.drop(['diagnosis','benign_malignant'],axis=1)
      sex_le.fit(df_filtered['sex'])
      anatom_le.fit(df_filtered['anatom_site_general_challenge'])
      df_filtered['sex']=sex_le.transform(df_filtered['sex'])
      df_filtered['anatom_site_general_challenge']=anatom_le.transform(df_filtered['anatom_site_general_challenge'])
  except Exception as e:
      print(e)
      df_filtered['sex']=sex_le.transform(df_filtered['sex'])
      df_filtered['anatom_site_general_challenge']=anatom_le.transform(df_filtered['anatom_site_general_challenge'])
  return df_filtered


# In[ ]:


df_train1=preprocess(df_train)


# In[ ]:


df_train1.head()


# In[ ]:


import cv2
import glob
import random


# In[ ]:


len(list(df_train.index[df_train["target"]==0]))


# In[ ]:




def get_images(directory,df_train):
    Images=[]
    label=0
    os.chdir(directory)
    indx1=list(df_train["image_name"].index[df_train["target"]==1][:584])
    indx1.extend(list(df_train["image_name"].index[df_train["target"]==0])[:2000])
    indexes=random.sample(indx1,len(indx1))
    #print(type(indexes))
    print(len(indexes))
    for image_file in indexes:
        image=imread(df_train1["image_name"][image_file]+".jpg")
        image = color.rgb2gray(image) 
        image=cv2.resize(image,(150,150))
        Images.append(image)
        label=label+1
        if label%100==0:
            print(label)
            print(np.array(Images).shape)
    return Images


# In[ ]:


Images=get_images("/kaggle/input/siim-isic-melanoma-classification/jpeg/train",df_train1)


# In[ ]:


#Images=np.array(Images)
#Images.shape


# In[ ]:


len(Images)
for i in range(len(Images)):
    Images[i]=Images[i].astype('float32')


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


train_x = np.stack(Images)
train_x /= 255.0
train_x = train_x.reshape(-1, 22500).astype('float32')


# In[ ]:


train_x.shape


# In[ ]:


km = KMeans(n_jobs=-1, n_clusters=2, n_init=20)
km.fit(train_x)
#pred = km.predict(val_x)


# In[ ]:


km


# In[ ]:


import pickle
os.chdir("/kaggle/working")


# In[ ]:



pickle.dump(km, open("kmeans_cluster.pkl", "wb"))


# In[ ]:



kmeans = pickle.load(open("/kaggle/input/kmeans-model/kmeans_cluster1.pkl", "rb"))


# In[ ]:


km=kmeans


# In[ ]:


from IPython.display import FileLink
os.chdir("/kaggle/working")
FileLink("kmeans_cluster.pkl")


# In[ ]:


index = ['Row'+str(i) for i in range(1, len(train_x)+1)]

df = pd.DataFrame(train_x, index=index)


# In[ ]:


df.to_csv("Images.csv",index=False)


# In[ ]:


len(df)


# In[ ]:


len(df_train1)


# In[ ]:


df_train.index


# In[ ]:


df_train2=df_train1.reset_index()


# In[ ]:


len(df_train2)


# In[ ]:


df_train2.index


# In[ ]:


Images=[]


# In[ ]:



def get_images_cluster(directory,df):
    global Images
    label=0
    os.chdir(directory)
    #print(len(indexes))
    for image_file in range(len(Images),len(df)):
        try:
            os.chdir(directory)
            image=imread(df["image_name"][image_file]+".jpg")
            image = color.rgb2gray(image) 
            image=cv2.resize(image,(150,150))
            image=[image]
            test = np.stack(image)
            test /= 255.0
            test = test.reshape(-1, 22500).astype('float32')
            pred = km.predict(test)
            Images.append(pred[0])
            label=label+1
            if label%100==0:
                print(label)
                os.chdir("/kaggle/working")
                with open("prediction.txt","w") as f:
                    for item in Images:
                        f.write("{}\n".format(item))
        except Exception as e:
            print(e)
            Images.append(0)
    return Images


# In[ ]:


Images=get_images_cluster("/kaggle/input/siim-isic-melanoma-classification/jpeg/train",df_train2)


# In[ ]:


Images


# In[ ]:


df_train2["cluster"]=Images
os.chdir("/kaggle/working")
df_train2.to_csv("final_train.csv",index=False)


# In[ ]:


df_test1=preprocess(df_test)


# In[ ]:


df_test1.head()


# In[ ]:


df_test2=df_test1.reset_index()


# In[ ]:


df_test2.index


# In[ ]:


Images=get_images_cluster("/kaggle/input/siim-isic-melanoma-classification/jpeg/test",df_test2)


# In[ ]:


#os.chdir("/kaggle/input/siim-isic-melanoma-classification/jpeg/test")
Images


# In[ ]:


df_test1["cluster"]=Images
os.chdir("/kaggle/working")
df_test1.to_csv("final_test.csv",index=False)


# In[ ]:


test.shape


# In[ ]:


import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)


# In[ ]:


get_ipython().system('pip install catboost')
from catboost import CatBoostClassifier


# In[ ]:


import xgboost as xgb
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import re


# In[ ]:


#len(df_train['patient_id'].unique())
df_train=pd.read_csv('/content/final_train.csv')
df_test=pd.read_csv('/content/final_test1.csv')


# In[ ]:


type(df_train['sex'][0])


# In[ ]:


patient=LabelEncoder()
vals=list(df_train['patient_id'])
vals.extend(df_test['patient_id'])
patient.fit(vals)


df_train['patient_id']=patient.transform(df_train['patient_id'])


# In[ ]:


'''patient_se=LabelEncoder()
vals1=list(df_test['sex'])
vals1.extend(df_test['sex'])
patient_se.fit(vals1)


df_train['sex']=patient_se.transform(df_train['sex'])'''


# In[ ]:


'''patient_an=LabelEncoder()
vals2=list(df_train['anatom_site_general_challenge'])
vals2.extend(df_test['anatom_site_general_challenge'])
patient_an.fit(vals2)


df_train['anatom_site_general_challenge']=patient_an.transform(df_train['anatom_site_general_challenge'])'''


# In[ ]:


df_test['patient_id']=patient.transform(df_test['patient_id'])
#df_test['sex']=patient_se.transform(df_test['sex'])
#df_test['anatom_site_general_challenge']=patient_an.transform(df_test['anatom_site_general_challenge'])


# In[ ]:


type(df_test['patient_id'][0])


# In[ ]:


df_test.head()


# In[ ]:


import seaborn as sns
sns.distplot(df_train['cluster'])


# In[ ]:


Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'cluster']]
#X["A"] = X["A"] / X["A"].max()'''
seed = 7
test_size = 0.33
Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'cluster']]
X_train=X
y_train=Y
X_test=df_test[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'cluster']]


# In[ ]:



train_DMatrix = xgb.DMatrix(X_train, label= y_train)
test_DMatrix = xgb.DMatrix(X_test)
param = {
    'booster':'gbtree', 
    'eta': 0.3,
    'num_class': 2,
    'max_depth': 100
}

epochs = 100
clf = xgb.XGBClassifier(n_estimators=2000, 
                        max_depth=8, 
                        objective='multi:softprob',
                        seed=0,  
                        nthread=-1, 
                        learning_rate=0.15, 
                        num_class = 2, 
                        scale_pos_weight = (len(X_train)/584))
clf.fit(X_train, y_train)
#clf.predict_proba(X_test)[:,1]
# clf.predict(x_test)


# In[ ]:



df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=target
df.head()
df.to_csv('Submission_xgboost_cluster.csv',index=False)
#sub_tabular = sub.copy()


# In[ ]:


Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge']]
#X["A"] = X["A"] / X["A"].max()'''
seed = 7
test_size = 0.33
Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge']]
X_train=X
y_train=Y
X_test=df_test[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge']]
train_DMatrix = xgb.DMatrix(X_train, label= y_train)
test_DMatrix = xgb.DMatrix(X_test)
param = {
    'booster':'gbtree', 
    'eta': 0.3,
    'num_class': 2,
    'max_depth': 100
}

epochs = 100
clf = xgb.XGBClassifier(n_estimators=2000, 
                        max_depth=8, 
                        objective='multi:softprob',
                        seed=0,  
                        nthread=-1, 
                        learning_rate=0.15, 
                        num_class = 2, 
                        scale_pos_weight = (len(X_train)/584))
clf.fit(X_train, y_train)
#clf.predict_proba(X_test)[:,1]
# clf.predict(x_test)

df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=target
df.head()
df.to_csv('Submission_xgboost_.csv',index=False)
#sub_tabular = sub.copy()


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0).fit(X_train,y_train)
clf.predict(X_test)
df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=target
df.head()
df.to_csv('Submission_Logistic_.csv',index=False)
df.head()


# In[ ]:


#prediction of the test set
#importing keras and other required library's
#importing and splitting data into training and testing 
import keras
from sklearn.model_selection import train_test_split

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
#creating new neural network model
classifier=Sequential()
classifier.add(Dense(output_dim=100,init='uniform',activation='relu',input_dim=X.shape[1]))
classifier.add(Dropout(p=0.2))


classifier.add(Dense(output_dim=50,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])




#trining or fitting the model
classifier.fit(X_train,y_train,batch_size=100,nb_epoch=100)
y_pred = classifier.predict(X_test)
y_pred


# In[ ]:


df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=y_pred
df.to_csv('Submission_NN.csv',index=False)
df.head()


# In[ ]:




