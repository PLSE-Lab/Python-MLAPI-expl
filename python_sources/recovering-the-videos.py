#!/usr/bin/env python
# coding: utf-8

# # Recovering the Videos
# In this notebook we will show how to view the videos contained in the [CVPR 2018 WAD Video Segmentation Challenge](https://www.kaggle.com/c/cvpr-2018-autonomous-driving) dataset. 
# ### Sources and Resources 
# * http://tiao.io/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
# * https://matplotlib.org/gallery/animation/dynamic_image2.html

# In[5]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread 
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os


# Here we take a peak at what files are in our training set folder. We notice here that the name of the files contain some information delimited by " \_". There seems to be two ids followed by a camera indicator. 

# In[6]:


os.listdir("../input/train_color")[0:10]


# Here we load the filenames into a dataframe and split the parts of the filename up so we can investigate what they are easily.  

# In[2]:


training_img_names = os.listdir("../input/train_color") 
trainingImgNameDF = pd.DataFrame( list(map(lambda s : [s]+s.split("_") , training_img_names )) ).drop(3, axis=1)
trainingImgNameDF.head()


# In respect to the leading id, column 1 in our dataframe, how many unique values are there? 

# In[8]:


trainingImgNameDF[1].unique().shape


# The first id seems to represent three different cars or video sessions. How many unique camera angles are there? 

# In[46]:


trainingImgNameDF[4].unique().shape


# We now define a function for creating little html videos from the video frames in the dataset. 

# In[47]:


def visVid(trainingImgNameDF,sessionID,cameraID,startFrame=0,endFrame=10,dataRoot="../input/train_color/"):
    res = trainingImgNameDF[ (trainingImgNameDF[1] == sessionID) & (trainingImgNameDF[4] == cameraID)].sort_values(by=2)
    imgURIs = list(res[0])
    
    fig = plt.figure(figsize=(10,10))
    frames = []
    for uri in imgURIs[startFrame:endFrame]:
        im = plt.imshow( imread(dataRoot+uri), animated=True)
        frames.append([im])
    
    plt.close(fig)
    return animation.ArtistAnimation(fig,frames,interval=50, blit=True,repeat_delay=1000)
    


# Here we create a video of 10 frames from the first session id's left camera ( 5.jpg ).  
# 
# *Make sure you press play on the video.*

# In[48]:


uniqueVideoID = trainingImgNameDF[1].unique()
ani = visVid(trainingImgNameDF,uniqueVideoID[0],"5.jpg")
HTML(ani.to_jshtml(default_mode="reflect"))


# Here we create another video with the frames of the right camera ( 6.jpg ). 

# In[50]:


uniqueVideoID = trainingImgNameDF[1].unique()
ani = visVid(trainingImgNameDF,uniqueVideoID[0],"6.jpg")
HTML(ani.to_jshtml(default_mode="reflect"))


# As we can see there seems to be three recorded driving sessions in the dataset with each frame of video having a left camera (5.jpg) and right camera(6.jpg) output. Fork this notebook and play with the parameters to view different parts of the videos and give this kernel an upvote if you found this useful. 
