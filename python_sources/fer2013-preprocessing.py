#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Import and preprocess data
# 
# Datasets:
# 
# - fer2013

# In[ ]:


# Constants for FER2013 dataset
FER2013_PATH = "/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013/fer2013.csv"
FER2013_WIDTH = 48
FER2013_HEIGHT = 48


# In[ ]:


data = pd.read_csv(FER2013_PATH)
data.head()


# In[ ]:


data.info()


# In[ ]:


data["Usage"].value_counts()


# In[ ]:


# Seperate training and public/private test data
data_publ_test = data[data.Usage=="PublicTest"]
data_priv_test = data[data.Usage=="PrivateTest"]
data = data[data.Usage=="Training"]


# In[ ]:


Emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]  # indices 0 to 6


# In[ ]:


data["emotion"].value_counts(sort=False)


# In[ ]:


def fer2013_show_instance(index):
    """Shows the image and the emotion label of the index's instance."""
    image = np.reshape(data.at[index, "pixels"].split(" "), (FER2013_WIDTH, FER2013_HEIGHT)).astype("float")
    image -= np.mean(image)
    image /= np.std(image)
    print(Emotions[data.at[index, "emotion"]])
    plt.imshow(image, cmap="gray")


# In[ ]:


fer2013_show_instance(np.random.randint(0,len(data)))


# In[ ]:


def fer2013_to_X():
    """Transforms the (blank separated) pixel strings in the DataFrame to an 3-dimensional array 
    (1st dim: instances, 2nd and 3rd dims represent 2D image)."""
    
    X = []
    pixels_list = data["pixels"].values
    
    for pixels in pixels_list:
        single_image = np.reshape(pixels.split(" "), (FER2013_WIDTH, FER2013_HEIGHT)).astype("float")
        X.append(single_image)
        
    # Convert list to 4D array:
    X = np.expand_dims(np.array(X), -1)
    
    # Normalize image data:
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    
    return X


# In[ ]:


# Get features (image data)
X = fer2013_to_X()
X.shape


# In[ ]:


# Get labels (one-hot encoded)
y = pd.get_dummies(data['emotion']).values
y.shape


# In[ ]:


# Save data
np.save("fer2013_X", X)
np.save("fer2013_y", y)

