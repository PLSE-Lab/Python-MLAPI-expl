#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import tensorflow as tf
get_ipython().system('pip install seaborn')
import seaborn as sns

from sklearn.model_selection import  train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.applications import DenseNet121
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


img_size = 512   # setting our image size to a standard value 

get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# # **LOADING DATA**

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# # ** EXPLORATORY DATA ANALYSIS **

# From the information given the 'disagnois' column gives us a rating for each image for the severity of diabetic retinopathy on a scale of 0 to 4
# 
# 0 - No DR 
# 
# 1 - Mild
# 
# 2 - Moderate
# 
# 3 - Severe
# 
# 4 - Proliferative DR

# Lets establish our x and y variable for this dataset

# In[ ]:


x = train_df['id_code']
y = train_df['diagnosis']


# To avoid any bias, we need to reset the data structure each time we run the cell above. We can do this using the shuffle object in the pandas dataframe

# In[ ]:


seed = 50  # defining the number of seeds in our shuffling 
x, y = shuffle(x, y, random_state=seed)


# ## Splitting Our Training & Testing Datasets

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state =seed)


# Now we need to do a count to see how many of each type of severity of diabetic retinopothy we have in our training dataset

# In[ ]:


train_df['diagnosis'].value_counts()


# It seems that the majority of the training set is made up of patients that have no DR. Overall, we can see a decrease in counts as we goes up in the the severity of DR
# 
# To better visualize this lets do a simple bar graph visualization 

# In[ ]:


train_df['diagnosis'].value_counts().plot.bar()

plt.title('Visualization of DR Trainig Dataset')
plt.xlabel('DR Diagnosis Type')
plt.ylabel('Patient Count')


# # **DISPLAYING SOME CORRECTED SAMPLE IMAGES**

# The purpose of this function is to display a set of grayscale converted and cropped samples images from the given dataset 

# In[ ]:


def display_train_images(df, col=3, row=3):
    
    tol = 30  # initializing our tolerence level for cropping 

    matrix = col*row
    fig = plt.figure(figsize=(10, 10)) #definingthe size of our inline plot from matplotlib 
    
    for i in range(matrix):
        
        image_path = df.loc[i,'id_code']     # assigning the id_code of each eye image to a certain path
        image_id = df.loc[i,'diagnosis']
        
        image = cv2.imread(f'../input/train_images/{image_path}.png')
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting images from a RGB format to a grayscale format 
        
        mask = img>tol
        img = img[np.ix_(mask.any(1),mask.any(0))]       # converting the images into an adjusted or cropped image
        
        img = cv2.resize(img, (img_size, img_size))   # resizing our images all to a single size 
        
        img=cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), img_size/10),-4, 128) # creating a guassian blur from the grayscale image 
        
        fig.add_subplot(row, col, i+1)
        
        plt.title(image_id)
        plt.axis('off')
        plt.imshow(img,cmap='gray')
        
    plt.tight_layout()

    return df


# ## Displaying Our Training Images

# In[ ]:


train_df = display_train_images(train_df)


# ## Displaying Our Testing Images 

# In[ ]:


def display_test_images(df, col=3, row=3):
    
    tol = 30  # initializing our tolerence level for cropping 

    matrix = col*row
    fig = plt.figure(figsize=(10, 10)) #definingthe size of our inline plot from matplotlib 
    
    for i in range(matrix):
        
        image_path = df.loc[i,'id_code']     # assigning the id_code of each eye image to a certain path
        image = cv2.imread(f'../input/test_images/{image_path}.png')
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting images from a RGB format to a grayscale format 
        
        mask = img>tol
        img = img[np.ix_(mask.any(1),mask.any(0))]       # converting the images into an adjusted or cropped image
        
        img = cv2.resize(img, (img_size, img_size))   # resizing our images all to a single size 
        
        
        fig.add_subplot(row, col, i+1)
        
        plt.axis('off')
        plt.imshow(img,cmap='gray')
        
    plt.tight_layout()


# In[ ]:


display_test_images(test_df)


# As we can see fromthe figure above, the top of each image in the subset shows the type of diagnosis for each patient (from the 0-4 scale) and each image in now converted into a grayscale image, cropped to 512x512 pixel image
# 
# Since our tesing images do not have a diagnosis, the numbers are missing from the top of each figure in the subplot 

# ## Creating An Image Info Generator 

# In[ ]:


BATCH_SIZE = 32

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )


# # **MODEL TRAINING - DENSENET 121**

# ### Creating A Call back for QWK

# In[ ]:


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


# In[ ]:


densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)


# ### Model Architecture Design 

# In[ ]:


def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='sigmoid'))
    
    # Compiling the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.00005),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model()
model.summary()


# In[ ]:


kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)


# In[ ]:




