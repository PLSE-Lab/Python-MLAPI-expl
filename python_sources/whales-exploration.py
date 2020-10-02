#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from os import listdir
import seaborn as sns
from operator import itemgetter 
import matplotlib.image as mpimg
import random
from PIL import Image
import collections as co
import cv2
import scipy as sp
import copy
import plotly.graph_objs as go
import plotly.offline as py


# # Whales identification challenge
# 
# Whales are group of aquatic marine mammals, whose closest relatives among land animals are hippopotamuses. After hundreds of years of relentless hunting, whales are now protected by international law. The North Atlantic right whales were close to extinction in the twentieth century, with a population of 450, and the North Pacific grey whale population is ranked Critically Endangered by the IUCN (according to Wiki). For the purpose of preservation of their population, it's important to count the population of whales and monitor their activity. The goal of this notebook is to propose the computer vision approach to identification of whale based on the photo of its humpback. 
# 
# 
# ## Files Statistics

# First of all we will check how many image files we have for training and testing. 

# We set the path to the folders of train and test images together with the path to the file with labels.

# In[ ]:


trainDir = "../input/whale-categorization-playground/train/train"
testDir="../input/whale-categorization-playground/test/test/"
valuesFile= "../input/whale-categorization-playground/train.csv"


# On the next step we estimate the number of files in each dataset, and compare the number of images in training set with number of values in the file with labels

# In[ ]:



lntrd=len(listdir(trainDir))
lntsd=len(listdir(testDir))

print("number of train files: "+ str(lntrd))
print("number of test files: " + str(lntsd))
trainPD=pd.read_csv(valuesFile)
if lntrd>0:
    print("lengths of test set to train set: %6.2f" % (lntsd/lntrd))
    if trainPD.shape[0]==lntrd:
        print("number of values and length of train set are consistent")
    else:
        print("number of values and length of train set are inconsistent")
else:
    print("train set is empty")


# We have larger test set than train set, but this is not an issue if the images of train set are representative set of population.

# ## Statistics of Labels

# Let's take a look at statistics of labels.

# First of all we calculate the frequency of occurence of ID in the files with values:

# In[ ]:


FrameID = trainPD.groupby("Id",as_index = False)["Image"].count()
sortedID_train = FrameID.sort_values("Image",ascending = False)
idnum=sortedID_train.shape[0]
print(idnum)


# There are 4251 unique labels. 

# In[ ]:


sortedID_train.head()


# The frequency of the first class is over 20 times larger than the one of the second most frequent one. Let's plot the statistics:

# In[ ]:


plt.plot(range(idnum),sortedID_train["Image"])

plt.xlabel("sorted index")
plt.ylabel("frequency of occurence")
plt.title("frequency of occurence of labels")


# We are dealing with extemely imbalance dataset. Let's plot without the first class: 

# In[ ]:


plt.plot(range(1,idnum),sortedID_train["Image"][1:idnum])

plt.xlabel("sorted index")
plt.ylabel("density")
plt.title("Density Plot for Labels")


# Imbalance is still evident. More than half of labels have frequency of only 1:

# The same graph in logarithmic scale:

# In[ ]:


plt.plot(range(1,idnum+1),sortedID_train["Image"])
plt.yscale("log")
plt.xlabel("ID")
plt.ylabel("Frequency of occurence")
plt.title("Frequency of occurence: log scale")


# Without the first element: 

# In[ ]:


plt.plot(range(1,idnum),sortedID_train["Image"][1:idnum])
plt.yscale("log")
#plt.yscale("log")
plt.xlabel("ID")
plt.ylabel("Frequency of occurence")
plt.title("Frequency of occurence: log scale")


# In such case of extreme distribution we would definetely require augmentation of train dataset.

# ## Image Visualization

# Let's take a look at first 25 images: 

# In[ ]:


imnum=25
plt.rcParams["figure.figsize"] = (70,70)
fig, subplots = plt.subplots(5,5)

for i in range(imnum):
    readImg=mpimg.imread(trainDir+"/"+(listdir(trainDir))[i])
    subplots[i // 5,i % 5].imshow(readImg)


# We can notice the following issues:
# 1. They are not consistent in terms of color spectrum: we can notice several images in black-and-white and most others in full color. 
# 2. They vary in size a lot. The model's pipeline would require substantial resizing.  
# 3. Some of them also have the fields with labels unrelated to the ID. 

# In[ ]:


readImg=mpimg.imread(trainDir+"/"+(listdir(trainDir))[10])
plt.rcParams["figure.figsize"] = (10,10)
plt.imshow(readImg)
print(listdir(trainDir)[10])


# In[ ]:


trainPD[trainPD["Image"] == "47841f63.jpg"]


# We see that information on the label on yellow space is not consistent with ID. Most likely, this information is useless for identification.  

# Lets take a look at test set:

# In[ ]:


imnum=25
plt.rcParams["figure.figsize"] = (70,70)
fig, subplots = plt.subplots(5,5)

for i in range(imnum):
    readImg=mpimg.imread(testDir+"/"+(listdir(testDir))[i])
    subplots[i // 5,i % 5].imshow(readImg)


# We can see the images of test set also differ in size a lot and are not consistent in terms of colour spectrum. In the next section we study how significant is this issue. 

# 
# ### Distribution of sizes of images

# Let's count the frequency of occurence of different sizes using dictionary: 

# In[ ]:


sizedict_train=dict()
filelist=listdir(trainDir)
for filename in filelist:
    size=(Image.open(trainDir+"/"+filename)).size
    if size in sizedict_train:
        sizedict_train[size]+=1
    else:
        sizedict_train[size]=1


# Let's sort the dictionary by values in descending order

# In[ ]:


sortpairs_train= sorted(sizedict_train.items(), key = itemgetter(1), reverse = True)


# We can see that three most common sizes are (1050,600), (1050,700) and (1050,450)

# Normalization of arrays:

# In[ ]:


sortsized_train = [sortpairs_train[i][1] for i in range(len(sortpairs_train))]
sortsized_train = sortsized_train/ np.sum(sortsized_train)


# Let's plot the statistics:

# In[ ]:


numsizes=len(sizedict_train)
print(numsizes)
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(sortsized_train)

plt.xlabel("index")
plt.ylabel("probability")
plt.title("probability of size")


# In[ ]:


sortsized_train[0:10]


# We have 2587 different sizes of images. The first three most frequent sizes occur with frequency: 11.2%, 9.6%, 4.1 %. The plot in logarithm scale:

# In[ ]:


numsizes=len(sizedict_train)
print(numsizes)

plt.plot(sortsized_train)
plt.yscale("log")
plt.xlabel("index")
plt.ylabel("probability")
plt.title("probability of size")


# The situation with test set is similar: 

# In[ ]:


sizedict_test=dict()
filelist=listdir(testDir)
for filename in filelist:
    size=(Image.open(testDir+"/"+filename)).size
    if size in sizedict_test:
        sizedict_test[size]+=1
    else:
        sizedict_test[size]=1


# In[ ]:


sortpairs_test= sorted(sizedict_test.items(), key = itemgetter(1), reverse = True)


# In[ ]:


sortpairs_test[0:10]


# Again we see that two most common sizes are (1050,600),(1050,700) and (1050,450) 

# In[ ]:


sortsized_test = [sortpairs_test[i][1] for i in range(len(sortpairs_test))]
sortsized_test = sortsized_test/ np.sum(sortsized_test)


# In[ ]:


numsizes=len(sizedict_test)
print(numsizes)

plt.plot(sortsized_test)

plt.xlabel("index")
plt.ylabel("probability")
plt.title("probability of size")


# In[ ]:


numsizes=len(sizedict_test)
print(numsizes)

plt.plot(sortsized_test)
plt.yscale("log")
plt.xlabel("sorted index")
plt.ylabel("density")
plt.title("Density Plot for Labels")


# In[ ]:


sortsized_test[0:10]


# The test has 3527 different sizes of images. The probability of the first three: 13%, 8.3%, 4.34%. 

# ### Color scheme

# In this section we estimate how many images are in grayscale format. 

# In[ ]:


def checkrgb(rgb):
    
    if len(rgb.shape)==3:
        return 0
    else:
        return 1


# In[ ]:


lntd=len(listdir(trainDir))
grayscale=[checkrgb(mpimg.imread(trainDir+"/"+(listdir(trainDir))[i])) for i in range(lntd)]


# In[ ]:


share_grey_train=np.sum(grayscale)/len(grayscale)
share_grey_train


# Around 15 % of images in train set are in greyscale format. 

# In[ ]:


lntd=len(listdir(testDir))
grayscale=[checkrgb(mpimg.imread(testDir+"/"+(listdir(testDir))[i])) for i in range(lntd)]


# In[ ]:


share_grey_test=np.sum(grayscale)/len(grayscale)
share_grey_test


# The test set has 22% of greyscale images. 

# ## Clustering of images

# We will now examine how similar are train set images to the images of test set. 
# 
# In order to do this we will tranform all images to grayscale and resize them. We choose the size (100,100), which will result in loss of finer details on images.
# 
# After transformation we select the subset of train and test images and use t-SNE - the machine learning algorithm for dimensionality reduction. The algorithm t-SNE maps high dimensional objects to two- or three-dimensional dots in the way that similar objects are modelled by nearby dots and dissimilar ones by distant dots. 

# Image transformation functions: 

# In[ ]:


def rgb2grey(rgb): 
    if len(rgb.shape)==3:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) 
    else:
        return rgb


def transform_image(img, rsc_dim):
    resized = cv2.resize(img, (rsc_dim, rsc_dim), cv2.INTER_LINEAR)
    
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
                         
    trans = normalized.reshape(1, np.prod(normalized.shape))

    return trans/np.linalg.norm(trans)


# In[ ]:


trainImg=[rgb2grey(mpimg.imread(trainDir+"/"+(listdir(trainDir))[i])) for i in range(400)]


# In[ ]:


testImg=[rgb2grey(mpimg.imread(testDir+"/"+(listdir(testDir))[i])) for i in range(400)]


# In[ ]:


rsc_dim=100
gray_all_images_train = [transform_image(img, rsc_dim) for img in trainImg]
gray_all_images_test  = [transform_image(img, rsc_dim) for img in testImg]


# In[ ]:


gray_imgs_mat_train = np.array(gray_all_images_train).squeeze()
gray_imgs_mat_test= np.array(gray_all_images_test).squeeze()


# We have prepared the array of transformed images for t-SNE procedure.

# In[ ]:


inputtsne=np.concatenate([gray_imgs_mat_train, gray_imgs_mat_test])


# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(inputtsne)


# In[ ]:



import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,20)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =tsne[0:400,0]
y =tsne[0:400,1]
z =tsne[0:400,2]

ax.scatter3D(x, y, z, c='r', marker='o')

x =tsne[400:800,0]
y =tsne[400:800,1]
z =tsne[400:800,2]

ax.scatter3D(x, y, z, c='b', marker='o')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# We can see there is a big heterogeneous cluster for both training and test sets and a few quite distant outliers. This means that training and test sets are quite similar after projection to low dimension. 

# # Conclusion of exploration

# We make several conclusions on the basis of exploration:
# 
# 1) We have to augment our data because there are many classes that are underrepresented. 
# 
# 2) There are thousand of different image sizes in the datasets. We need to recise the picture, and probably try different sizes for transformation.  
# 
# 3) We also see that there are different color schemes in datasets. We need to greyscale (or red scale) the images before applying the model.
# 
# 4) We have selected the subsets of training and test images and used projection to low dimensional space and found that projections from both datasets form large claster 
# with few outliers. That means that training and test are quit similar in terms of low diminsional patterns. 
