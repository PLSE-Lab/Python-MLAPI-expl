#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import seaborn as sns
import cv2
from random import randint
import shutil


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
       # print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:



os.getcwd()
#os.chdir('/Users/Aron/Kaggle/plant_pathology')
local_dir = '/Users/Aron/Kaggle/plant_pathology/plant-pathology-2020-fgvc7'
kaggle_dir = '../input/plant-pathology-2020-fgvc7/'

sample_submission = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")

train.head()


# In[ ]:


#There are 4 categories.  check to make sure there is a fatir representation of each of the 4 categories.

# since each image can only be represented in each column once, 
# the mean of the columns are the percentage each column is of the data.
print(train.sum())
pcts = train.mean()
pcts.plot(kind = 'bar')


# Kaggle dosent allow me to write to the input filee, os Iwill copy tht e images over to the working folder to manipulate.  I could probably do this another way, but, Im using the tools I know without looking too much up.
# 
# 

# In[ ]:


#copies all of the files in the 
#make sure the directory is gone before it copies evertything
shutil.rmtree('/kaggle/working/images')
#os.remove("/kaggle/working/images")
#copies all of the files in the 
shutil.copytree(kaggle_dir+'images/', '/kaggle/working/images')
#filelist = [ f for f in os.listdir(kaggle_dir+'/images') if f.startswith("Train_") ]

#os.mkdir('/kaggle/working/images/')
#for f in filelist:
#    shutil.copyfile(kaggle_dir+'images/'+filelist, '/kaggle/working/images/'+filelist)
    


# In[ ]:





# In[ ]:



multi_disease = train[train['multiple_diseases'] == 1]
#grab the multiple disease examples
multi_list = multi_disease['image_id']
#grab the id's from the above rows
multi_disease_list = multi_list.values.tolist() 
#converte to a list



#define the new image number
img_num = len(train)-1
df_new_name = []

for i in range(5):
    for img_name in multi_disease_list:
        new_name = 'Train_'+ str(img_num) #call the new name for the image
        img = cv2.imread(kaggle_dir+'/images/'+ img_name + '.jpg') 
        # load the image that matches the name in the list
        cv2.imwrite('/kaggle/working/images/'+ new_name +'.jpg', img) # write the image with the new name to the 
        dat = {
            'image_id': new_name,
            'healthy' : 0,
            'multiple_diseases' : 1,
            'rust' : 0,
            'scab' : 0
            }
        #write the line for the dataframe
        df_new_name.append(dat)
        img_num += 1 # do this last to incriment before next loop

df = pd.DataFrame(df_new_name)

train = pd.concat([train, df])


# In[ ]:





# In[ ]:


new_name


# In[ ]:


path1= '/kaggle/working/images/Train_2100.jpg'
img=mpimg.imread(path1)
imgplot = plt.imshow(img)


# In[ ]:


print(train.sum())
pcts = train.mean()
pcts.plot(kind = 'bar')


# In[ ]:





# In[ ]:



#The sample submission file, the training data and testing labels are read in.  ther eare 1821 training and testing images in the dataset.Next is to look at the distribution of the training set ategories.  

#There are 4 categories.  check to make sure there is a fatir representation of each of the 4 categories.

# since each image can only be represented in each column once, 
# the mean of the columns are the percentage each column is of the data.
print(train.sum())
pcts = train.mean()
pcts.plot(kind = 'bar')


#We can see that the multiple disease column is the least represented in the data. 


#lets check out a couple of the images from the training set.

#Check image size
im = cv2.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
print(type(im))
print(im.shape)
print(type(im.shape))

#now resize and look at them

img_size = 273
train_image = []
for name in train['image_id']:
    path= '/kaggle/working/images/'+name+'.jpg'
    img=cv2.imread(path)
    image=cv2.resize(img,(img_size,img_size))
    train_image.append(image)


fig, axis = plt.subplots(1, 4, figsize=(41, 27))
for i in range(4):
    axis[i].set_axis_off()
    axis[i].imshow(train_image[1000+i])
    # randint(0,1500)+
    #generate different pictures from the test data.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Now check some test images

test_image = []
for name in test['image_id']:
    path='../input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'
    img=cv2.imread(path)
    image=cv2.resize(img,(img_size, img_size))
    test_image.append(image)

fig, axis = plt.subplots(1, 4, figsize=(21, 47))
for i in range(4):
    axis[i].set_axis_off()
    axis[i].imshow(test_image[i])
    # randint(0,1500)+
    #generate different pictures from the test data.


# In[ ]:



#shape the training images to work for keras.
x_train = np.asarray(train_image, dtype=np.float32)
x_train = x_train/255

x_train.shape

x_test = np.asarray(test_image, dtype=np.float32)
x_test = x_test/255

x_test.shape

type(train)


#grab the labels for the training images.
y = train.iloc[:,1:5]
# turn the labels into an arrray
y_train = np.array(y.values, dtype='float32')

#Check that the shape is correct.  1821 rows, 4 columns
#and check the value of the first row. should be floats
print(y_train.shape,y_train[0])


# After the miserable performance of the previous, model, I atarted looking in to how to resample the training data. I knew what I needed to do, but was having some difficulty knowing how to cde it specifically.  The bar chart shws that the multiple disease class is so infrequent, only making up 91 of the images in the data.  One way I could deal with it is multiply the size of the class by 5 or 6 to increase it to 455 - 546 instances. I will probably try this to see how it works, as well as read up on why this simple method should not be done. I will see if I can use the image generator, where it rotates, zooms, and transposes the images to help with the issue of having 5 or 6 copies of a multiple disease training instance.  
# 
# 

# In[ ]:


91*6


# In[ ]:


import keras
from keras.models import Model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer, Input
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# resplit and shape the data again. the data to get a clean set.
x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                  y_train, 
                                                  test_size = 0.20, 
                                                  random_state = 403 )


# In[ ]:


#use the sequential model

cnn = Sequential()
I = Input(shape=(img_size, img_size, 3))
#add aconvolutional layer
cnn.add(Conv2D(25, kernel_size = (5,5), strides = (1,1), 
               padding = 'same',
               activation = 'relu',
               input_shape = (img_size, img_size, 3)
              )
       )

cnn.add(Conv2D(75, kernel_size = (5,5), strides = (1,1),
               padding = 'same',
               activation = 'relu',
               input_shape = (img_size, img_size, 3)
              )
       )
cnn.add(MaxPool2D(pool_size=(4,4)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(150, kernel_size = (3,3), strides = (1,1),
               padding = 'same',
               activation = 'relu',
               input_shape = (img_size, img_size, 3)
              )
       )
cnn.add(MaxPool2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(300, activation='relu'))
cnn.add(Dropout(0.35))
cnn.add(Dense(150, activation='relu'))
cnn.add(Dropout(0.25))
# output layer
cnn.add(Dense(4, activation='softmax'))

optimizer_1 = Adam(lr=0.001)
optimizer_2 = SGD(lr=0.01)

cnn.compile(loss="categorical_crossentropy", optimizer=optimizer_1, metrics=["accuracy"])


# In[ ]:


cnn.fit(x_train,
        y_train,
        batch_size = 32,
        epochs = 50,
        validation_data = (x_val,y_val))


# In[ ]:


#resNetModel.save('plant_pathology_resnet50.h5')
predict= cnn.predict(x_test)


# In[ ]:


prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)
for row in range(test.shape[0]):
    for col in range(4):
        if predict[row][col] == max(predict[row]):
            prediction[row][col] = 1
        else:
            prediction[row][col] = 0
prediction = pd.DataFrame(prediction)
prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
df = pd.concat([test.image_id, prediction], axis = 1)
df.to_csv('submission.csv', index = False)
from IPython.display import FileLink
FileLink(r'submission.csv')


#   

# In[ ]:


predict[3
       ]


# In[ ]:


predict = cnn.predict(x_test)


# In[ ]:


# predict using the trained model

predict= cnn.predict(x_test)


# In[ ]:


#convert the probabilties to binary predictions by taking the max of the row and turning to 1
prediction = np.ndarray(shape = (test.shape[0],4), dtype = np.float32)
for row in range(test.shape[0]):
    for col in range(4):
        if predict[row][col] == max(predict[row]):
            prediction[row][col] = 1
        else:
            prediction[row][col] = 0

#convert it to a data frame and add the image names. 


# In[ ]:


prediction = pd.DataFrame(prediction)
prediction.columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
df = pd.concat([test.image_id, prediction], axis = 1)

df.head()
df.tail()


# In[ ]:


#Turn into a submission file
df.to_csv('submission.csv', index = False)


# In[ ]:


os.getcwd()


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')

