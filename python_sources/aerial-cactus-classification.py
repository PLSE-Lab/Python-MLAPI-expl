#!/usr/bin/env python
# coding: utf-8

# # Aerial Cactus Classification Challenge - Kaggle
# 
# 
# ##### check this link for description:
# https://www.kaggle.com/c/aerial-cactus-identification/
# 
# ####  Dataset Download link:
# https://www.kaggle.com/c/13435/download-all
# 

# ### Importing the Necessary Depedencies

# In[ ]:


import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# ### Reading the CSV file

# In[ ]:


data=pd.read_csv('../input/train.csv')
data.head()


# ## Visualizing the Data

# In[ ]:


sns.countplot(x='has_cactus', data=data)


# In[ ]:


label=np.array(data['has_cactus'])


# ### Applying the Convolutional Neural Network

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(2, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# ### Optimization

# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


from PIL import Image
matrix=[]
for i in range(len(data['has_cactus'])):
    #print(data['id'][i])
    im=Image.open('../input/train/train/'+data['id'][i])
    m=np.array(im)
    matrix.append(m)


# #### Converting the Image and label into an Numpy array

# In[ ]:


x=np.array(matrix)
y=np.array(label)


# ### Training the Model

# In[ ]:


model.fit(x,y,epochs=8)


# #### Overall Accuracy is 96.26%

# ### Predicitng the Test Data

# In[ ]:


from tensorflow.keras.preprocessing import image
import os
subm=[]
list=os.listdir('../input/test/test')
length=len(list)

for i in range(length):
    path = '../input/test/test/'+list[i]+''
    img = image.load_img(path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    pre=float(classes[0])
    print(pre)
    subm.append(pre)
    if classes[0]>0.5:
        print(list[i]+" is a cactus")
    else:
        print(list[i]+" is not a cactus")


# ### Converting the Predicted values into Pandas Dataframe

# In[ ]:


x=pd.DataFrame({'id':list,'has_cactus':subm})
x.head()


# ## Exporting The Prediction

# In[ ]:


x.to_csv('final_submission.csv',index=False)

