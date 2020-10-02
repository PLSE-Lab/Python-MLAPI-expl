#!/usr/bin/env python
# coding: utf-8

# **Python based Kernel for Amazon**
# 
# We see there are various labels associated with each image / chip showing the phenomenon occurring in Amazon rain forest 
# 
# These 'Class Labels'have following parts:
# 
#  1. 'Atmospheric Condition' and always exist
#     --> *'clear', 'partly_cloudy', 'cloudy', and 'haze'*     
# 
#  2. 'More Common Labels' that may or may not exist
#    --> *'primary', 'water', 'habitation', 'agriculture', 'road', 'cultivation', 'bare_ground'*
# 
#  3. 'Less Common Labels' that rarely exist
#     --> *slash_burn, selective_logging, 'blooming', 'conventional_mining', 'artisinal_mining', 'blow_down*
# 
# **We treat 'labels' as a corpus of documents. Then we convert this corpus into a feature matrix using CountVectorizer() which is then converted to a pandas dataframe.** This should give a nice manageble set of data corresponding to the images which can be analysed.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.plotly as py
import plotly.tools as tls
import cv2
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from sklearn import preprocessing


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')
print(df.head())


# Use CountVectorizer to transform the corpus of documents (i.e. the labels associated with the images) into matrix form. Then create bar graph of the labels vs their count.

# In[ ]:


labels = np.array(df['tags'])

vect = CountVectorizer()
vect.fit(labels)
vect.get_feature_names()

labels_dtm = vect.transform(labels)
df_labels = pd.DataFrame(labels_dtm.toarray(), columns = vect.get_feature_names())

# create a dict to collect total values of each class of label
amazon_condition = {}
for col in df_labels.columns.values:
    z = df_labels.groupby([col])[col].count().astype(int)
    amazon_condition[col] = 0
    for i, j in enumerate(z):
        if i != 0:
            amazon_condition[col] += j
amazon_condition_labels = [x for x in amazon_condition.keys()]
amazon_condition_values = [x for x in amazon_condition.values()]

print(amazon_condition_labels)
print(amazon_condition_values)


# In[ ]:


#from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import fbeta_score
from PIL import Image
from PIL import ImageStat
import glob

def extract_features(path):
    features = []
    image_stat = ImageStat.Stat(Image.open(path))
    features += image_stat.sum
    features += image_stat.mean
    features += image_stat.rms
    features += image_stat.var
    features += image_stat.stddev
    img = cv2.imread(path)
    cv2img = cv2.imread(path,0)
    features += list(cv2.calcHist([cv2img],[0],None,[256],[0,256]).flatten())
    mean, std = cv2.meanStdDev(img)
    features += list(mean)
    features += list(std)
    return features


# In[ ]:


from tqdm import tqdm
from time import time

X_train = pd.DataFrame()
input_path = '../input/'
df['path'] = df['image_name'].map(lambda x: input_path + 'train-jpg/' + x + '.jpg')

f_list = []

t0 = time()
for i in df['path']:
    f = np.array(extract_features(i)).astype(int)
    f_list.append(f)

print("done in %0.3fs" % (time() - t0))    


# In[ ]:


f_list_arr = np.array(f_list)
X_train = pd.DataFrame(f_list_arr)
#normalize the X_train scale
#X_train = preprocessing.scale(X_train)
for i in X_train.columns.values:
    X_train[i] = X_train[i]/max(X_train[i])
    
print(type(X_train))
print(X_train.head())


# In[ ]:


print(type(X_train))
print(type(df_labels))
Y_train = np.array(df_labels)
print(Y_train[:5])
print(len(X_train[0]))
pad = np.zeros((len(X_train[0]),42))
print(pad.shape)
X_train = np.hstack((X_train,pad))
print(X_train.shape)


# In[ ]:


print(df_labels.head())


# In[ ]:


#print(test_images)


# In[ ]:


test_images = []
X_test = []
test_images = glob.glob(input_path + 'test-jpg-v2/*')
X_test = pd.DataFrame([[x.split('/')[3].replace('.jpg',''),x] for x in test_images])
print(X_test[:5])

X_test.columns = ['image_name','path']
print(X_test[:5])
print(X_test.shape, type(X_test))


# In[ ]:


ftr_list_arr=[]
ftr_list = []
pad=[]

t0 = time()
for i in X_test['path']:
    ftr = np.array(extract_features(i)).astype(int)
    ftr_list.append(ftr)
print("done in %0.3fs" % (time() - t0))    

ftr_list_arr = np.array(ftr_list)


# In[ ]:


print(ftr_list_arr.shape)
print(len(ftr_list_arr))
print(type(ftr_list_arr))


# In[ ]:


pad = np.zeros((len(ftr_list_arr),42))
print(pad.shape)
ftr_list_arr = np.hstack((ftr_list_arr,pad))
test_pred = pd.DataFrame(ftr_list_arr)

print(test_pred.shape)
#print(test_pred.head())

#normalize the test_pred scale
#test_pred = preprocessing.scale(test_pred)

for i in test_pred.columns.values:
    test_pred[i] = test_pred[i]/max(X_train[i])


# In[ ]:


X_train_to_csv = pd.DataFrame(X_train)
Y_train_to_csv = pd.DataFrame(Y_train)

X_train_to_csv.to_csv('X_train.csv')
Y_train_to_csv.to_csv('Y_train.csv')
test_pred.to_csv('pred.csv')


# In[ ]:


print(check_output(["ls"]).decode("utf8"))
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(test_pred.shape)


# In[ ]:


X_val = X_train[32000:]
X_train = X_train[:32000]
Y_val = Y_train[32000:]
Y_train = Y_train[:32000]


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

print(test_pred.shape)


# In[ ]:


from keras import backend as K
img_rows = 18 
img_cols = 18

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #X_test_k = X_test_k.reshape(X_test_k.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    #X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    #X_test_k = X_test_k.reshape(X_test_k.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# In[ ]:


print(Y_train.shape)
print(Y_val.shape)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
print(X_train.shape)
print(X_val.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


batch_size = 128
nb_classes = 17
#nb_epoch = 5
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 18, 18
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]
    
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, 
          validation_data=(X_val, Y_val), shuffle=True)

score = model.evaluate(X_val, Y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


#print(score)
#model.predict_proba(X_val[:100])
print(X_val[:2])
class_predictions = []
class_predictions = model.predict_classes(X_val[:2], batch_size=32, verbose=1)
print(class_predictions.astype(float))


# In[ ]:


p_test = model.predict(X_val, verbose=1)
#p_test [p_test >0.24] = 1
#p_test [p_test < 1] = 0
print(p_test.shape)
print(p_test[:10])
print(Y_val[:10])

print


# In[ ]:


from sklearn.metrics import fbeta_score

#p_valid = model.predict(X_val, batch_size=128)
p_proba = model.predict_proba(X_val, batch_size=128)
#print(y_valid)
#print(p_valid)
#print(fbeta_score(Y_val, np.array(p_valid) > 0.2, beta=2, average='samples'))


# In[ ]:


print(X_val[:2])
print(p_proba[:10])


# In[ ]:


print(type(test_pred))
predset = np.array(test_pred)
predset = predset.reshape(predset.shape[0],18,18,1)


# In[ ]:


print(predset.shape)


# In[ ]:


p_test = model.predict(predset, batch_size = 128, verbose=2)


# In[ ]:


print(p_proba[:1])
print(Y_val[:1])


# In[ ]:


#result [result >0.24] = 1
#result [result < 1] = 0
#print(result[:5])


# In[ ]:


#print(type(result))
#result_df = pd.DataFrame(result)
#result_df.columns = df_labels.columns.values
#print(result_df.head())
#tags = []

#for i,j in enumerate(np.array(result_df)):
#    temp_tags = []
    #print(temp_tags)
#    for c, col in enumerate(result_df.columns.values):
#        if j[c] == 1:
#            temp_tags.append(col)
#    tags.append(temp_tags)

#tags1 = []    

#for x in tags:
#    st = ''    
#    for y in x:
#        st += y + ' '
#    tags1.append(st[:(len(st)-1)])         


# In[ ]:


#print(len(tags), type(tags))
#print(tags[:10])
#print(len(tags1), type(tags1))
#print(tags1[:10])
#X_test['tags'] = tags1

#X_test[:10]
#print(X_test.columns.values)


# In[ ]:


#X_test[['image_name','tags']].to_csv('submission_amazon_02.csv', index=False)


# In[ ]:


#pdForCSV = pd.DataFrame()
#pdForCSV['image_name'] = X_test.image_name.values
#pdForCSV['tags'] = preds
#pdForCSV.to_csv('2017_05_01_XGB_submission.csv', index=False)

