#!/usr/bin/env python
# coding: utf-8

# Image Classificaation using SVM is very efficient way of modelling and very rarely used algorithm for image processing and modelling..!!!!

# Tips for using SVM for image classification
# 
# * You should have image data in 2D rather than 4D (as SVM training model accepts dim <=2 so we need to convert the image data to 2D which i'll be showing later on in this notebook).
# 
# * SVM algorithm is to be used when their is shortage of data in our dataset .
# 
# * If we have good amount of image data so, we look further for CNN model.
# 

# # INFO OF DATASET...!!

# The Dataset is named as 'Color Classification' created by Aydin Ayanzadeh. we are provided with images of different color set with labels of color name such as red,blue,etc link :- https://www.kaggle.com/ayanzadeh93/color-classification

# **Importing the dataset**

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
        os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# importing basic Packages..!!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os
from tqdm import tqdm


# now,we have provided data directory to DATADIR variable and labels of color set to CATEGORIES variable for further use.

# In[ ]:


DATADIR = '../input/color-classification/ColorClassification'
CATEGORIES = ['orange','Violet','red','Blue','Green','Black','Brown','White']
IMG_SIZE=100


# Ex. of an sample image is shown below
# 

# In[ ]:


for category in CATEGORIES:
    path=os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break


# performing preprocessing steps...::
# 

# In[ ]:


training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()            


# In[ ]:


print(len(training_data))


# storing trainig length for further use.

# In[ ]:


lenofimage = len(training_data)


# for image to be trained we have to convert the image to a array form so,that our model can train on it...!!

# and X should be of type (training_data_length , -1) because SVM takes 2D input to train

# In[ ]:


X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(lenofimage,-1)
##X = tf.keras.utils.normalize(X, axis = 1)


# In[ ]:


X.shape


# **flattening the array**

# In[ ]:


X = X/255.0


# Ex. of flattened array...

# In[ ]:


X[1]


# note : y should be in array form compulsory.
# 

# In[ ]:


y=np.array(y)


# In[ ]:


y.shape


# Now we are ready with our dependent and independent features, now its time for data modelling
# 
# applying train_test_split on our data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


# **fitting our data in SVM model**

# In[ ]:


from sklearn.svm import SVC
svc = SVC(kernel='linear',gamma='auto')
svc.fit(X_train, y_train)


# **predicting the X_test**

# In[ ]:


y2 = svc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy on unknown data is",accuracy_score(y_test,y2))


# **Ahhyeah....accuracy of 92.28% which is what we wanted..!!!!**

# **fromulating the Classification report**

# In[ ]:


from sklearn.metrics import classification_report
print("Accuracy on unknown data is",classification_report(y_test,y2))


# In[ ]:


result = pd.DataFrame({'original' : y_test,'predicted' : y2})


# In[ ]:


result


# we have moslty classified all the images correctly with their labels .doing classification on limited dataset is always a challenging task....but by SVM we have dealed with it succesfully

# *IF YOU LIKED MY KERNAL PLEASE UPVOTE IT*

# In[ ]:




