#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#      for filename in filenames:
#          print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# We use VGG16 without the top layer for feature extraction and then use t-SNE algorithm to see how well the features were extracted 
# In another notebook, the results from VGG16 feature extraction will be compared with the results from ResNet50

# Importing all required modules

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.applications import VGG16 ,ResNet50
from sklearn.preprocessing import LabelEncoder
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.layers import Input
import seaborn as sns
import pandas as pd
import itertools
import shutil
import pickle
import random
import time
import csv
import os
get_ipython().system('pip install imutils')
from imutils import paths


# Setting up the path for input and working directories

# In[ ]:


base_path="/kaggle/input"
work_path="/kaggle/working"


# Loading the VGG16 model, without the top layer, since we are only intending to extract the features and not a classification
# The shape of the Input layer has been changed to 128,128 to keep the memory requirements low

# In[ ]:


print("[ALERT]...loading the VGG16 model")
modelVgg=VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(128, 128, 3)))
le=None


# Looking on the model summary to understand the Shape of output layers

# In[ ]:


modelVgg.summary()


# Setting up the datasets path and batch size

# In[ ]:


datasets=["training"]
batch_size=32


# In[ ]:


# os.remove("/kaggle/working/Vgg16.csv")


# Setting up the loop to parse the images and extract the features in batch size and then store in the prescribed excel sheet
# We use label encoder to encode the label strings

# In[ ]:


for train_images in (datasets):
    #we have to grab the images from the training path to extract the features
    p=os.path.sep.join([base_path,train_images])
    imagePaths=list(paths.list_images(p))
    #randomly shuffule the images to make sure all varianta are present randomly 
    random.shuffle(imagePaths)
    labels=[p.split(os.path.sep)[-2] for p in imagePaths]
    if le is None:
        le=LabelEncoder()
        le.fit(labels)
    print(set(labels))
    csvPathVgg=os.path.sep.join([work_path,"{}.csv".format("Vgg16")])
    csvVgg=open(csvPathVgg, "w")

    for (b, i) in enumerate(range(0, len(imagePaths), batch_size)):
        # extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
        print("[INFO] processing batch {}/{}".format(b + 1,int(np.ceil(len(imagePaths) / float(batch_size)))))
        batchPaths = imagePaths[i:i + batch_size]
        batchLabels = le.transform(labels[i:i + batch_size])
        batchImages = []
        for imagePath in batchPaths:
            # the image is resized to 128*128 pixels, since the VGG16 model is loaded with same image resolution
            image = load_img(imagePath, target_size=(128,128))
            image = img_to_array(image)

            # preprocess the image by  expanding the dimensions and preprocessing using the imagenet utils
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # add the image to the batch
            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        featuresVgg = modelVgg.predict(batchImages, batch_size=batch_size)
        #reshaping the feature array to the output layer shape (4*4*512)
        featuresVgg = featuresVgg.reshape((featuresVgg.shape[0], 4 * 4 * 512))
        # loop over the class labels and extracted features
        for (label, vec) in zip(batchLabels, featuresVgg):
            # construct a row with "," to make sure while writing csv, each value(feature) goes to a sepearte column
            vec = ",".join([str(v) for v in vec])
            csvVgg.write("{},{}\n".format(label, vec))
print("extraction completed")


# Path to the csv file

# In[ ]:


csvVgg=os.path.sep.join([work_path,"Vgg16.csv"])


# Function to read the size (count) of colmns in the csv already saved

# In[ ]:


def getcsvcount(csvfile):
    with open(csvVgg, 'r') as csv:
        first_line = csv.readline()
    ncol = first_line.count(',') + 1
    return ncol


# In[ ]:


csvVggcount=getcsvcount(csvVgg)
colsVgg=['feature_'+str(i) for i in range(csvVggcount-1)]


# Splitting the CSV file into labels and data and storing in seperate lists

# In[ ]:


def load_data_split(splitPath):
    #data and label varialbles
    data=[]
    labels=[]
    for row in open(splitPath):
        row=row.strip().split(",")
        label=row[0]
        features=np.array(row[1:],dtype="float")

        data.append(features)
        labels.append(label)
    data=np.array(data)
    labels=np.array(labels)
    return(data,labels)


# In[ ]:


dataVgg,labelsVgg=load_data_split(csvVgg)
df_Vgg=pd.DataFrame(dataVgg,columns=colsVgg)
df_Vgg['labels'] = labelsVgg
print('Size of the dataframe: {}'.format(df_Vgg.shape))


# Sizing down the initial dataset to a subset of size 8000 (any random number) to reduce the computational cost

# In[ ]:


# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df_Vgg.shape[0])
N=8000
df_subset = df_Vgg.loc[rndperm[:N],:].copy()
data_subset = df_subset[colsVgg].values
y=df_subset['labels'].values


# Evaluating t-SNE algorithm on the subset data

# In[ ]:


#tSNE dimensionality reduction
time_start=time.time()
tsne=TSNE(n_components= 2,verbose=1,perplexity=40,
          n_iter=600)
tsne_results=tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# plotting the t-SNE result in a 2D graph

# In[ ]:


df_subset['tsne-one'] = tsne_results[:,0]
df_subset['tsne-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    hue=y,
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.9
)


# This brings an insight on how VGG16 did the features extraction and t-SNE algorithm enabled to distinguish the same.
