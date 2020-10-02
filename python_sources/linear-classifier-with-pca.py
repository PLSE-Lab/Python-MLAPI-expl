#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import RBFSampler,Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import os


# In[ ]:


#https://github.com/radekosmulski/whale/blob/master/oversample.ipynb

df = pd.read_csv('../input/train.csv')
im_count = df[df.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
df = df.join(im_count, on='Id')
val_fns = set(df.sample(frac=1)[(df.Id != 'new_whale') & (df.sighting_count > 1)].groupby('Id').first().Image)


# In[ ]:


df_val = df[df.Image.isin(val_fns)]
train_df = df[~df.Image.isin(val_fns)]
# df_train_with_val = df


# In[ ]:


# res = None
# sample_to = 2

# for grp in df_train.groupby('Id'):
#     n = grp[1].shape[0]
#     additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)
#     rows = pd.concat((grp[1], additional_rows))
    
#     if res is None: res = rows
#     else: res = pd.concat((res, rows))


# In[ ]:


#https://www.kaggle.com/pestipeti/keras-cnn-starter

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 1))
    count = 0
    
    for fig in data['Image']:
        
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100), color_mode='grayscale')
        x = image.img_to_array(img)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig, end='\r')
        count += 1
    
    return X_train


# In[ ]:


X_train = prepareImages(df, df.shape[0], "train")
X_train.shape


# In[ ]:


X_test = prepareImages(df_val, df_val.shape[0], "train")
X_test.shape


# In[ ]:


enc = OneHotEncoder()
enc.fit(df.Id.values.reshape(-1,1))
y_train = enc.transform(df.Id.values.reshape(-1,1))
y_test = enc.transform(df_val.Id.values.reshape(-1,1))
n_classes = len(df.Id.unique())
label_encoder = LabelEncoder()
label_encoder.fit(df.Id.values)


# In[ ]:


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
scaler = MinMaxScaler(copy=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


pca = PCA(n_components=1000, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# In[ ]:


inputs = Input(shape=(1000,))
x = Dense(n_classes, activation='softmax')(inputs)
model = Model(inputs=inputs, outputs=x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()


# In[ ]:


model.fit(X_train_pca, y_train, validation_data=(X_test_pca, y_test), epochs=3, batch_size=20)


# In[ ]:



test = os.listdir("../input/test/")
test_df = pd.DataFrame(test, columns=['Image'])
test_df['Id'] = ''
Test_img = prepareImages(test_df, test_df.shape[0], "test")
Test_img.shape
Test_img = Test_img.reshape(Test_img.shape[0], -1)
Test_img = scaler.fit_transform(Test_img)
Test_img = pca.transform(Test_img)


# In[ ]:


predictions = model.predict(Test_img, batch_size=1)


# In[ ]:


for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


# In[ ]:


test_df.head()
test_df.to_csv('submission.csv', index=False)


# In[ ]:




