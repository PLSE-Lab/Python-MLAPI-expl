#!/usr/bin/env python
# coding: utf-8

#  # About this notebook
#  
# This notebook presents a Knaive model and is a supporting notebook to the my main notebook here, which presents complete detialed explanation and dicussions abobut this competitiom:
# https://www.kaggle.com/tanulsingh077/steganalysis-complete-understanding-and-model ---> Original author (TANUL SINGH)
# 
# To understand how I came up with this idea and how everything works please refer above
# 
# **<span style="color: red;">If you like my effort and contribution please show token of gratitude by upvoting my kernels</span>**
# # Basic Idea
# Yesterday Kaggle launched the steganalysis competition. I found it very interesting as I have never heard of something like this. It instilled the spy fantasies within me. From yesterday I have been reading research papers on approaching this problem using deep learning but before I tried **SRNET** I want to do something of own. After a lot of thinking I came up with this :-<br><br>
# * In this we are suppose to predict whether the test images are hiding some information or not but we dont have labels for the train images we just clean images and same images encoded using different algos, the main point is creating labels then we can approach this as regression problem . I thought Since in steganography in images, any technique involves changes in pixel values, a very knaive way to get labels would be to flatten the RGB images(both encoded and normal) into a vector and then find cosine dissimilarity between the two vectors, since the encoded value contains a hidden information its vector will differ from the main vector and hence we will have a non zero value of cosine dissimilarity.
# * Then we Label the images as: Cover_images as 0 , JMIPOD images as similarity between cover and JMIPOD images , JUNIWARD images as similarity between cover and JUNIWARD images, UERD images as similarity between cover and UERD images
# * After creating our labels , we will stack all the images and labels in a datafame and then approach this as simple regression problem
# 
# Update : One flaw that I found in the above approach was that labels to be predicted should be between zero and one, whereas our regression problem predicts values of any range and values can also be negative , so I passed out cosine dissmilarity through sigmoid function to get a probability between zero and one

# In[ ]:


# PRELIMINARIES
import os
import skimage.io as sk
import matplotlib.pyplot as plt
from scipy import spatial
from tqdm import tqdm
from PIL import Image
from random import shuffle


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
train_imageids = pd.Series(os.listdir(BASE_PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)
test_imageids = pd.Series(os.listdir(BASE_PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)


# In[ ]:


cover_images_path = pd.Series(BASE_PATH + '/Cover/' + train_imageids ).sort_values(ascending=True)
JMIPOD_images_path = pd.Series(BASE_PATH + '/JMiPOD/'+train_imageids).sort_values(ascending=True)
JUNIWARD_images_path = pd.Series(BASE_PATH + '/JUNIWARD/'+train_imageids).sort_values(ascending=True)
UERD_images_path = pd.Series(BASE_PATH + '/UERD/'+train_imageids).sort_values(ascending=True)
test_images_path = pd.Series(BASE_PATH + '/Test/'+test_imageids).sort_values(ascending=True)
ss = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


# In[ ]:


final=[]
def create_labels(cover,jmipod,juniward,uerd,image_id):
    image = sk.imread(cover)
    jmipodimg = sk.imread(jmipod)
    juniward = sk.imread(juniward)
    uerd = sk.imread(uerd)
    
    vec1 = np.reshape(image,(512*512*3))
    vec2 = np.reshape(jmipodimg,(512*512*3))
    vec3 = np.reshape(juniward,(512*512*3))
    vec4 = np.reshape(uerd,(512*512*3))
    
    cos1 = spatial.distance.cosine(vec1,vec2)
    cos2 = spatial.distance.cosine(vec1,vec3)
    cos3 = spatial.distance.cosine(vec1,vec4)
    
    final.append({'image_id':image_id,'jmipod':cos1,'juniward':cos2,'uerd':cos3})


# I will be taking only the first 30k images , although you can all of the images if you want

# In[ ]:


for k in tqdm(range(30000)):
    create_labels(cover_images_path[k],JMIPOD_images_path[k],JUNIWARD_images_path[k],UERD_images_path[k],train_imageids[k])


# In[ ]:


train_temp = pd.DataFrame(final)
train_temp.head()


# Adding softmax to our dissimilarity to get Probabilities

# In[ ]:


def sigmoid(X):
   return 1/(1+np.exp(-X))


# In[ ]:


train_temp['jmipod'] = train_temp['jmipod'].apply(lambda x:sigmoid(x))
train_temp['juniward'] = train_temp['juniward'].apply(lambda x:sigmoid(x))
train_temp['uerd'] = train_temp['uerd'].apply(lambda x:sigmoid(x))


# In[ ]:


train_temp.head()


# In[ ]:


IMG_SIZE = 300
def load_training_data():
  train_data = []
  data_paths = [cover_images_path,JUNIWARD_images_path,JMIPOD_images_path,UERD_images_path]
  labels = [np.zeros(train_temp.shape[0]),train_temp['juniward'],train_temp['jmipod'],train_temp['uerd']]
  for i,image_path in enumerate(data_paths):
    for j,img in enumerate(image_path[:10000]):
        label = labels[i][j]
        img = Image.open(img)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        train_data.append([np.array(img), label])
        
  shuffle(train_data)
  return train_data


# In[ ]:


def load_test_data():
    test_data = []
    for img in test_images_path:
        img = Image.open(img)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        test_data.append([np.array(img)])
            
    return test_data


# * I have used only top 44,000 images because if I run on more than that it exceeds memory allocation
# 
# * Now that we have a data loader , we can now load the data and check if everything is working

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = load_training_data()')


# ## Sanity Check

# In[ ]:


len(train)


# In[ ]:


plt.imshow(train[115][0], cmap = 'gist_gray')


# In[ ]:


trainImages = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train])


# # Model
# 
# I am using 5 Conv2D layers with max pooling and batch norm for my baseline

# In[ ]:


#PRELIMINARIES
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization


# In[ ]:


model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))


# In[ ]:


model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mean_squared_error'])


# In[ ]:


print(model.summary())


# In[ ]:


model.fit(trainImages, trainLabels, batch_size = 100, epochs = 3, verbose = 1)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = load_test_data()\ntestImages = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict = model.predict(testImages,batch_size=100)')


# In[ ]:


ss['Label'] = predict


# In[ ]:


ss.to_csv('submission.csv',index=False)


# In[ ]:


train_temp.to_csv('train.csv',index=False)


# In[ ]:




