#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Quick, Draw! Mini Project 
# 
# The purpose of the miniproject is to classify the "Quick Draw!" dataset using a constructed CNN architecture. 
# 
# Below an overview of the content of this notebook can be found.

# 
# ### Overview
# 
# - The Data
# - The Network
# - Breakdown of Results
# 
# 
# 

# In[ ]:


#setup
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import ast
import json
from PIL import Image, ImageDraw 
path_train = '../input/train_simplified/'


# ## 1. The Data 
# The train data is stored in a zip file containing a csv file for each class. There is a total of 340 classes. 
# ### Load the Data
# In the following small code snippet the csv file for the class "cat" is loaded. Here only the first 10 rows are loaded into a pandas DataFrame just to show the structure of the data. As can be seen each sample is described by 6 parameters; a country code, the drawing, a key id, whether it was recognised or not, a timestamp, and the label (a word).

# In[ ]:


cat=pd.read_csv('../input/train_simplified/cat.csv',nrows=10)
cat


# ### a. How many images does the dataset consist of?

# In[ ]:


#load 
classSamples=[]
filenames=[]
for filename in os.listdir(path_train):
    filenames.append(filename)
    temp=pd.read_csv(os.path.join(path_train,filename),usecols=['word'])
    classSamples.append(temp.shape[0])
print("The dataset consists of ",sum(classSamples)," images.")


# ### b. How many classes? How many images per class?

# In[ ]:


minSamples=min(classSamples)
print("There are ",len(filenames)," classes, and there are between ",minSamples," and ",max(classSamples)," samples per class.")


# ### c.  Show 20 sample images from at least 8 different classes
# Below 20 images from 10 randomly selected classes are shown.

# In[ ]:


display_samples=pd.DataFrame()
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/cat.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/owl.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/dog.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/ant.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/alarm clock.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/key.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/trumpet.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/donut.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/feather.csv',usecols=['drawing', 'word'],nrows=2))
display_samples=display_samples.append(pd.read_csv('../input/train_simplified/frog.csv',usecols=['drawing', 'word'],nrows=2))


# In[ ]:


#display_samples['drawing'] = display_samples['drawing'].apply(ast.literal_eval)


# In[ ]:


display_samples['drawing'] = display_samples['drawing'].apply(json.loads)


# In[ ]:


figrows=4
figcols=5
fig, axs = plt.subplots(nrows=figrows, ncols=figcols, sharex=True, sharey=True, figsize=(16, 10))
for i, drawing in enumerate(display_samples.drawing):
    ax = axs[i // figcols, i % figcols]
    for x, y in drawing:
        ax.set_title(display_samples.word.iloc[i])
        ax.plot(x, -np.array(y), lw=3)
    ax.axis('off')
plt.show()


# ### d. Consider if/how the data distribution will affect training of a classifier.
# 
# If the network training the model is presented with more samples from a certain class than other, the model is likely to be more specialised in the traits of the given class  and thus perform better when classifying that specific class. If there is a significant imbalance in the amount of samples from the different classes presented during training the training of recognition of some classes might overrule the training of other classes resulting in a poor perfomance when classifying the overruled classes.

# ## 2. The Network
# The intend is to create a Convolutional Neural Network which trains on the data in an image form rather than the "stroke" form in the provided dataset. 
# 
# In the following code snippet the csv file for each class is loaded. From each file a pandas DataFrame is extracted and appended to a collective DataFrame, which will serve as the training data containing datapoints from all the classes.  The kernel dies if all data about all samples is loaded, therefore the amount of data kept in structures has ben minimised. In order minimise memory use only the columns "Drawing" and "Word", which is the label, and in some cases "Recognized" will be used. Furthermore, only 500 samples from each class is used. 

# In[ ]:


get_ipython().run_line_magic('reset', '-f')


# In[ ]:


#setup
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import graphviz

from tqdm import tqdm
import ast
import cv2
from glob import glob
from PIL import Image, ImageDraw 
from dask import bag
from numpy import array
from numpy import argmax
from sklearn.utils import shuffle
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
path_train = '../input/train_simplified/'

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#import pydotplus as pydot


# A function is defined for converting from strokes to an image

# In[ ]:


#Definitions of functions
#imheight, imwidth = 32, 32 
imheight, imwidth = 64, 64  
# function for converting from strokes into image
def draw_it(strokes):
    image = Image.new("P", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in ast.literal_eval(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.
print("Functions defined")


# Data is loaded and converted

# In[ ]:


train_data=pd.DataFrame()
samples_per_class=500
for filename in os.listdir(path_train):
    #filenames.append(filename)
    #temp=pd.read_csv(os.path.join(path_train,filename),usecols=['drawing','word'],nrows=number_of_samples_per_class)
    temp=pd.read_csv(os.path.join(path_train,filename),usecols=['countrycode','drawing', 'recognized','word'],nrows=samples_per_class*5//4)
    temp=temp[temp.recognized == True].head(samples_per_class)
    temp=temp.loc[:,['countrycode','drawing','word']]
    train_data=train_data.append(temp)
print("dataload done")
featureVector = []
n_cores = mp.cpu_count()
pool = mp.Pool(processes=n_cores//2)
featureVector = pool.map_async(draw_it,train_data.drawing).get()
pool.close()
pool.join()
print("FINISHED parallel for train data")
train_data['drawing'] =featureVector

del featureVector
del temp


# In[ ]:


## for showing examples
#print(train_data.word.iloc[3])
#plt.imshow(train_data.drawing.iloc[3])


# Splitting in train and test sets and encoding labels and countries 

# In[ ]:


#making sure the dataset is randomly mixed before splittiong
train_data=shuffle(shuffle(train_data,random_state=33),random_state=27)
label_encoder=LabelEncoder()
onehot_encoder=OneHotEncoder(sparse=False)
train_data.countrycode=train_data.countrycode.fillna('0')
unique_countries=train_data.countrycode.unique()
num_countries=unique_countries.shape[0]

country_encoder=LabelEncoder()
country_encoder.fit(unique_countries)


temp_label=label_encoder.fit_transform(train_data.word)
forOneHot = temp_label.reshape(len(temp_label), 1)
#print(forOneHot)
label=onehot_encoder.fit_transform(forOneHot)
#print(label)
#train_X,validation_X,train_y,validation_y = train_test_split(train_data,label,stratify = label,test_size=0.20)
train_X,validation_X,train_y,validation_y=train_test_split(train_data, label, test_size=0.2, random_state=13)
#train_x=np.stack(np.array(train_X.drawing))
#train_x=train_X.drawing.tolist()
#validation_image=validation_X.drawing

del forOneHot
del temp_label
del train_data


# converting split data to the right format

# In[ ]:


train_x=np.array(train_X.drawing.tolist())
validation_x=np.array(validation_X.drawing.tolist())
train_x = train_x.reshape(train_x.shape[0], imheight, imwidth, 1)
validation_x = validation_x.reshape(validation_x.shape[0], imheight, imwidth, 1)
validation_cc=country_encoder.transform(validation_X.countrycode)
del validation_X
del train_X


# Creating the structure of the network

# In[ ]:


batch_size = 150
epochs = 50
input_shape = train_x[0].shape
#input_shape = train_image.iloc[0].shape
print(input_shape)
num_classes = 340
    
model = Sequential()
model.add(Conv2D(12, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_last',
                     padding = "same"))

model.add(Conv2D(12, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_last',
                     padding = "same"))
#model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
    #model.add(Dropout(0.25))
model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_last',
                     padding = "same"))

model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape,
                     data_format='channels_last',
                     padding = "same"))
model.add(BatchNormalization())
model.add(Conv2D(32,
                     kernel_size=(5, 5),
                     activation='relu',
                     padding = "same",
                     data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
    
model.add(Conv2D(64,
                     kernel_size=(5, 5),
                     activation='relu',
                     padding = "same",
                     data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
    
model.add(Conv2D(64,
                     kernel_size=(5, 5),
                     activation='relu',
                     padding = "same",
                     data_format='channels_last'))
    
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax',))
    
learningrate = 1e-2
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy',
                  optimizer=adagrad,
                  metrics=['accuracy'])
model.summary()
print("Structure made")


# Training the model

# In[ ]:


history = model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_split=0.1)


# Evaluation of training.

# In[ ]:


"""
#Could be used for plotting the CNN structure but only wors sometimes
from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
"""


# In[ ]:


print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
  
plt.show()

fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'], loc='lower right')
  
plt.show()


# Classify test dataset

# In[ ]:


score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)


# Examination of accuracies

# In[ ]:


Pred_label=model.predict(validation_x)
Predict_label=Pred_label.dot(onehot_encoder.active_features_).astype(int)
Num_real_y=validation_y.dot(onehot_encoder.active_features_).astype(int)


# In[ ]:


Class_perform=np.zeros([num_classes,2])
Class_Acc=np.zeros([num_classes])
for i in range(Num_real_y.shape[0]):
    if Predict_label[i]==Num_real_y[i]:
        Class_perform[Num_real_y[i],0]+=1
    Class_perform[Num_real_y[i],1]+=1
    for k in range(num_classes):
        if Class_perform[k,1]>0:
            Class_Acc[k]=Class_perform[k,0]/Class_perform[k,1]
indices=Class_Acc.argsort()[::-1][:5]
print("Top 5 classified classes: ",label_encoder.inverse_transform(indices))
print("Accuracy of top 5 classified classes: ",Class_Acc[indices])
print("Sample amount of top 5 classified classes: ",Class_perform[indices,1])
print("Worst Class :",label_encoder.inverse_transform(Class_Acc.argsort()[0]))
print("Accuracy of worst class :",Class_Acc[Class_Acc.argsort()[0]])


# In[ ]:


Country_perform=np.zeros([num_countries,2])
Country_Acc=np.zeros([num_countries])
for i in range(Num_real_y.shape[0]):
    if Predict_label[i]==Num_real_y[i]:
        Country_perform[validation_cc[i],0]+=1
    Country_perform[validation_cc[i],1]+=1
    for k in range(num_countries):
        if Country_perform[k,1]>0:
            Country_Acc[k]=Country_perform[k,0]/Country_perform[k,1]
indicesC=Country_Acc.argsort()[::-1][:5]
print("Top 5 classified countries: ",country_encoder.inverse_transform(indicesC))
print ("Accuracy of top 5 classified countries: ",Country_Acc[indicesC])
print("Sample amount from top 5 classified countries: ",Country_perform[indicesC,1])
print("Worst country :",country_encoder.inverse_transform(Country_Acc.argsort()[1]))
print("Accuracy of worst country :",Country_Acc[Country_Acc.argsort()[1]])


# The results can vary from run to run with the same settings. This might be due to the fact that there seemingly is no random seed used for when the train data is shuffled inbetween epochs. 

# 
