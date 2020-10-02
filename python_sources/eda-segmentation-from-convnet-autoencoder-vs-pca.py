#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# In this kernel, I am interested to understand more about the Pneumothorax diagnosis masks. I want to identify natural categories / clusters of pneumothorax diagnosis that exist. I'll use different unsupervised learning approach. Starting with PCA + KMeans, and then use different autoencoders.
# 
# The result might be useful to have better understanding about the illness itseslf. It would also be useful if we want to do hierarchical step for our prediction
# 
# I am using this kernel for references on data extraction: https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data#data

# # Sections
# 
# * A: All the data pre-processing
# * B-1: Use simple PCA & K-Means
# * C: All the autoencoders
# * C1: Shallow network AE - learned nothing
# * C2: Deep (fully-connected) AE - still not useful
# * C3: Deep Convolutional AE - finally worked well

# In[ ]:


import pydicom
import os
import glob
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

import plotly.graph_objs as go
import plotly.plotly as py
import plotly.offline as pyo
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly_express as px
init_notebook_mode(connected=True)

import tensorflow as tf

from tqdm import tqdm_notebook

# ['siim-acr-pneumothorax-segmentation-data', 'siim-acr-pneumothorax-segmentation']

import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation/')

from mask_functions import rle2mask
import gc


# In[ ]:


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)
            
def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


# # A-1 Load training data

# In[ ]:


samplesize = 5000
train_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/*.dcm'
test_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/*/*/*.dcm'
train_fns = sorted(glob.glob(train_glob))[:samplesize]
test_fns = sorted(glob.glob(test_glob))[:samplesize]
df_full = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv', index_col='ImageId')


# # A-2 Generate diagnostic masking data in image format

# ### Convert DCM into numpy array
# In this step, I will be constructing Y_train from DCM format into numpy array. The starter code was taken from [this](http://https://www.kaggle.com/jesperdramsch/intro-chest-xray-dicom-viz-u-nets-full-data#data) kernel, with some adjustments: (1) I skipped loading X_train to save memory, since I'm only interested in the shape of the mask, (2) I changed the implementation from the original additive function into taking the maximum (since some points are overlapping.
# > Y_train[n] =  np.maximum(Y_train[n], np.expand_dims(rle2mask(x, 1024, 1024), axis=2))

# In[ ]:


im_height = 1024
im_width = 1024
im_chan = 1
# Get train images and masks
# X_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_fns), im_height, im_width, 1), dtype=np.int16)
print('Getting train images and masks ... ')
sys.stdout.flush()
for n, _id in tqdm_notebook(enumerate(train_fns), total=len(train_fns)):
    dataset = pydicom.read_file(_id)
#     X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
    try:
        if '-1' in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
            Y_train[n] = np.zeros((1024, 1024, 1))
        else:
            if type(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']) == str:
                x = np.expand_dims(rle2mask(df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels'], 1024, 1024), axis=2)
                Y_train[n] = x
            else:
                Y_train[n] = np.zeros((1024, 1024, 1))
                for x in df_full.loc[_id.split('/')[-1][:-4],' EncodedPixels']:
                    Y_train[n] =  np.maximum(Y_train[n], np.expand_dims(rle2mask(x, 1024, 1024), axis=2))
    except KeyError:
        print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
        Y_train[n] = np.zeros((1024, 1024, 1)) # Assume missing masks are empty masks.

print('Done!')


# ### Rescale the image down to save memory

# In[ ]:


from skimage.transform import rescale

image_setori = []
for i in range(samplesize):
    count = Y_train[i].sum()
    if count > 0:
        image_setori.append(rescale(Y_train[i],1.0/4.0)) 


# ### Remove the last dimension and flatten the image

# In[ ]:


image_set = np.asarray(image_setori)
samplesize = len(image_set)
image_set = np.squeeze(image_set)
image_set = np.reshape(image_set, ((samplesize, image_set.shape[1] * image_set.shape[2])))


# ### Normalize the value to be [0 to 1]

# In[ ]:


image_set = image_set * 128
# for i in range(len(image_set)):
#     print(image_set[i].max())


# # Util to check memory size

# In[ ]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# We delete Y_train to save memory

# In[ ]:


del Y_train
gc.collect()


# # B-1 Start with PCA

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Review explained variance plot to determine number of PCA components (capped at 50):

# In[ ]:


pca = PCA(n_components=50).fit(image_set)
#Plotting the Cumulative Summation of the Explained Variance
y=np.cumsum(pca.explained_variance_ratio_)
data = [go.Scatter(y=y)]
layout = {'title': 'PCA Explained Variance'}
iplot({'data':data,'layout':layout})


# Even at n=20 components, we still can only explain 50% of variance. Oh well. 
# 
# Let's still zoom in with n=20 components:

# In[ ]:


pca = PCA(n_components=20)
image_PCA = pca.fit_transform(image_set)


# In[ ]:


trace1 = go.Scatter(y=pca.explained_variance_ratio_)
trace2 = go.Scatter(y=np.cumsum(pca.explained_variance_ratio_))
fig = tools.make_subplots(rows=1,cols=2,subplot_titles=('Explained Variance','Cumulative Explained Variance'))
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)
fig['layout'].update(height=600, width=1200, title="Explained Variance Ratios",showlegend=False)
iplot(fig)


# We'll also check elbow curve for number of clusters for kmeans

# In[ ]:


Nc = range(1,20)
kmeans = [KMeans(i) for i in Nc]
score = [kmeans[i].fit(image_PCA).score(image_PCA) for i in range(len(kmeans))]


# In[ ]:


data = [go.Scatter(y=score,x=list(Nc))]
layout = {'title':'Elbow Curve for KMeans'}
iplot({'data':data,'layout':layout})


# In[ ]:


n_clusters=12
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
image_kmeans = kmeans.fit_predict(image_PCA)


# In[ ]:


image_kmeans.shape


# In[ ]:


image_clusters = np.zeros((n_clusters, image_set.shape[1]), dtype=np.float64)
clustercounts = np.zeros(n_clusters,dtype=np.int)
for i in range(samplesize):
    for j in range(n_clusters):
        if image_kmeans[i] == j:
            image_clusters[j] += image_set[i]
            clustercounts[j] += 1


# Check shape of image_clusters (number of clusters, height x width). Each row is individual clusters and the columns are the flattened pixels.
# Check the number of images in each clusters and total images in our training data

# In[ ]:


print(image_clusters.shape)
print(clustercounts)
print(clustercounts.sum())


# We'll transform the image_clusters into height x width format 

# In[ ]:


for j in range(n_clusters):
    image_clusters[j] = image_clusters[j] / clustercounts[j]
image_clusters = np.reshape(image_clusters, ((n_clusters, 256, 256)))


# In[ ]:


image_clusters.shape


# We then do a simple step to sort the clustering output based on location. Simply sum the masks on the top left. Print 

# In[ ]:


image_clusters_sortingval = [np.sum(image_clusters[i,:80,:80]) for i in range(n_clusters)]
cluster_ordered = range(n_clusters)
cluster_ordered = [x for _,x in sorted(zip(image_clusters_sortingval,cluster_ordered))]


# ### Finally now we display the clusters of Pneumothorax masks
# 
# In general, we can see that the clusters are based on location (left, right) and also the size of the masks

# In[ ]:


fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
j = 1
for i in cluster_ordered:
    plt.subplot(2,6,j)
    plt.imshow(image_clusters[i].T, cmap=plt.cm.bone)
    plt.title('Cluster '+str(i)+'. Num Samples: '+str(clustercounts[i]))
    j += 1
plt.tight_layout()
plt.suptitle("Clusters of Pneumothorax Diagnosis based on simple PCA & K-Means",fontsize=16,y=1.05)


# In[ ]:


from sklearn.manifold import TSNE
imageTSNE = TSNE(n_components=2).fit_transform(image_PCA)


# In[ ]:


imageTSNEdf = pd.concat([pd.DataFrame(imageTSNE),pd.DataFrame(image_kmeans)],axis=1)
imageTSNEdf.columns = ['x1','x2','cluster']
px.scatter(imageTSNEdf,x='x1',y='x2',color='cluster',color_continuous_scale=px.colors.qualitative.Plotly,title="TSNE visualization of Image Clusters",width=800,height=500)


# ### Let's also do a simplified version with just 6 clusters
# 
# Interestingly, there are 4 variations of left-side masks but only 1 variation of right-side marks. The majority is still scattered. Most likely this reflect small spots that is not big enough to belong in the other clusters

# In[ ]:


n_clusters=6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
image_kmeans = kmeans.fit_predict(image_PCA)
image_clusters = np.zeros((n_clusters, image_set.shape[1]), dtype=np.float64)
clustercounts = np.zeros(n_clusters,dtype=np.int)
for i in range(samplesize):
    for j in range(n_clusters):
        if image_kmeans[i] == j:
            image_clusters[j] += image_set[i]
            clustercounts[j] += 1
for j in range(n_clusters):
    image_clusters[j] = image_clusters[j] / clustercounts[j]
image_clusters = np.reshape(image_clusters, ((n_clusters, 256, 256)))
image_clusters_sortingval = [np.sum(image_clusters[i,:80,:80]) for i in range(n_clusters)]
cluster_ordered = range(n_clusters)
cluster_ordered = [x for _,x in sorted(zip(image_clusters_sortingval,cluster_ordered))]
fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.05, wspace=0.05)
j = 1
for i in cluster_ordered:
    plt.subplot(1,6,j)
    plt.imshow(image_clusters[i].T, cmap=plt.cm.bone)
    plt.title('Cluster '+str(i)+'. Num Samples: '+str(clustercounts[i]))
    j += 1
plt.tight_layout()
plt.suptitle("Clusters of Pneumothorax Diagnosis based on simple PCA & K-Means",)


# In[ ]:


imageTSNEdf = pd.concat([pd.DataFrame(imageTSNE),pd.DataFrame(image_kmeans)],axis=1)
imageTSNEdf.columns = ['x1','x2','cluster']
px.scatter(imageTSNEdf,x='x1',y='x2',color='cluster',color_continuous_scale=px.colors.qualitative.Plotly,title="TSNE visualization of Image Clusters",width=800,height=500)


# In[ ]:


# Clean-up
del image_PCA
gc.collect()


# In[ ]:


# import sys

# # These are the usual ipython objects, including this one you are creating
# ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# # Get a sorted list of the objects and their sizes
# sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[ ]:


del df_full
gc.collect()


# # C-1 AutoEncoder: Simple AE with fully-connected layer

# ## One hidden layer

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
encoding_dim = 128
imgsize_flat = 256 * 256
input_img = Input(shape=(imgsize_flat,))
encoded = Dense(encoding_dim,activation='relu')(input_img)
decoded = Dense(imgsize_flat,activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# Encoder
encoder = Model(input_img,encoded)

# Decoder
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


# In[ ]:


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
xtrain = image_set
autoencoder.fit(xtrain,xtrain,epochs=20,batch_size=256,shuffle=True)


# ## Check how the original compare with the reconstructed
# 

# In[ ]:


fig = plt.figure(figsize=(15,10))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(xtrain[i].reshape(256,256).T, cmap=plt.cm.bone)
    autoencoded = autoencoder.predict(xtrain[i:i+1])
    plt.subplot(2,5,i+6)
    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)
plt.tight_layout()
plt.suptitle('Comparing original vs AE reconstruction images',fontsize=16,y=1)        


# Still random noise. As visible for large loss value, the network hasn't learned anything meaningful

# # C-2 Deep autoencoder - fully connected
# 
# tl;dr Fully-connected deep autoencoder still failed to produce meaningful encoding.

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
encoding_dim = 64
imgsize_flat = 256 * 256
layer1_multiplier = 32
layer2_multiplier = 16

input_img = Input(shape=(imgsize_flat,))
encoded = Dense(encoding_dim*layer1_multiplier ,activation='relu')(input_img)
encoded = Dense(encoding_dim*layer2_multiplier,activation='relu')(encoded)
encodedFinal = Dense(encoding_dim,activation='relu')(encoded)
decoded = Dense(encoding_dim*layer2_multiplier,activation='relu')(encodedFinal)
decoded = Dense(encoding_dim*layer1_multiplier ,activation='relu')(decoded)
decodedFinal = Dense(imgsize_flat,activation='sigmoid')(decoded)

autoencoder = Model(input_img, decodedFinal)

# Encoder
encoder = Model(input_img,encodedFinal)

# Decoder
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))


# In[ ]:


autoencoder.summary()


# In[ ]:


xtrain = image_set


# In[ ]:


autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(xtrain,xtrain,epochs=20,batch_size=64,shuffle=True)


# ### Check how the original compare with the reconstructed
# 

# In[ ]:


fig = plt.figure(figsize=(15,5))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(xtrain[i].reshape(256,256).T, cmap=plt.cm.bone)
    plt.title('Image ' + str(i))
    
    autoencoded = autoencoder.predict(xtrain[i:i+1])
    plt.subplot(2,10,i+11)
    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)
    plt.title('AE ' + str(i))
plt.tight_layout()
plt.suptitle('Comparing original vs AE reconstruction images',fontsize=16,y=1)    


# The reconstruction pretty much generated black images

# ## Running some diagnostics to check the encoded features
# 

# In[ ]:


image = []
for i in range(10):
    image.append(encoder.predict(xtrain[i:i+1]))
image = np.array(image)
image = np.squeeze(image)
imagedf = pd.DataFrame(image)
imagedf


# In[ ]:


fig = plt.figure(figsize=(24,8))
for i in range(8):
    series = imagedf.iloc[:,i]
    plt.subplot(4,8,i+1)
    series.hist()
    plt.title('Dim ' + str(i))
plt.suptitle('Histogram for each encoding dimension')
plt.tight_layout()


# The encoding pretty much failed. Many encoding dimensions are perfectly correlated. The correlation dataframe below further confirms that

# In[ ]:


imagedf.corr().iloc[:5,:5] #just a sample


# # C-3 Convolutional Autoencoder

# In[ ]:


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img=Input(shape=(256,256,1))
x = Conv2D(16,(3,3),activation='relu',padding='same')(input_img)
x = MaxPooling2D((4,4), padding='same')(x)
x = Conv2D(4,(3,3), activation='relu',padding='same')(x)
encoded = MaxPooling2D((4,4), padding='same')(x)

x = Conv2D(4,(3,3),activation='relu',padding='same')(encoded)
x = UpSampling2D((4,4))(x)
x = Conv2D(16,(3,3),activation='relu',padding='same')(x)
x = UpSampling2D((4,4))(x)
decoded = Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

autoencoderCNN=Model(input_img,decoded)
autoencoderCNN.compile(optimizer='adam',loss='binary_crossentropy')

# Encoder
encoderCNN = Model(input_img,encoded)

# Decoder
encoded_inputCNN = Input(shape=(16,16,4,))
decoder1 = autoencoderCNN.layers[-1]
decoder2 = autoencoderCNN.layers[-2]
decoder3 = autoencoderCNN.layers[-3]
decoder4 = autoencoderCNN.layers[-4]
decoder5 = autoencoderCNN.layers[-5]

decoderCNN = Model(encoded_inputCNN,decoder1(decoder2(decoder3(decoder4(decoder5(encoded_inputCNN))))))


# In[ ]:


autoencoderCNN.layers


# In[ ]:


autoencoderCNN.summary()


# In[ ]:


xtrain = np.reshape(xtrain, (len(xtrain),256,256,1))
autoencoderCNN.fit(xtrain,xtrain,epochs=20,batch_size=64,shuffle=True)


# In[ ]:


fig = plt.figure(figsize=(15,10))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(xtrain[i:i+1].reshape(256,256).T, cmap=plt.cm.bone)
    plt.title('Image ' + str(i))
    autoencoded = autoencoderCNN.predict(xtrain[i:i+1])
    plt.subplot(2,5,i+6)
    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)
    plt.title('AE ' + str(i))
plt.suptitle('Comparing original vs AE reconstruction images (5 images)',fontsize=16,y=1)
plt.tight_layout()


# In[ ]:


fig = plt.figure(figsize=(15,5))
for i in range(11,20):
    plt.subplot(2,10,i-10)
    plt.imshow(xtrain[i].reshape(256,256).T, cmap=plt.cm.bone)
    plt.title('Image ' + str(i))    
    autoencoded = autoencoderCNN.predict(xtrain[i:i+1])
    plt.subplot(2,10,i-0)
    plt.imshow(autoencoded.reshape(256,256).T, cmap=plt.cm.bone)
    plt.title('AE ' + str(i))
plt.suptitle('Comparing original vs AE reconstruction images (10 images)',fontsize=16,y=1.03)
plt.tight_layout()


# ## The reconstructions above looks pretty good!
# Finally.

# ### Quickly check what the encoded features look like (hidden)

# In[ ]:


encoderCNN = Model(input_img,encoded)
encodedX = []
for i in range(len(xtrain)):
    encodedX.append(encoderCNN.predict(xtrain[i:i+1]))
encodedX = np.array(encodedX)
print(encodedX.shape)
encodedX = np.squeeze(encodedX)
print(encodedX.shape)
encodeddf = pd.DataFrame(encodedX.reshape(encodedX.shape[0],np.prod(encodedX.shape[1:])))
encodeddf.head(20)


# # Clustering the encoded representations

# Lets use explained variance ratio to figure out a good PCA n_components

# In[ ]:


pca = PCA(n_components=50)
image_PCA = pca.fit_transform(encodeddf)
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(pca.explained_variance_ratio_)
ax[0].title.set_text("Explained variance ratio")
ax[1].plot(np.cumsum(pca.explained_variance_ratio_))
ax[1].title.set_text("Cumulative Explained variance ratio")


# We'll use 30 components for now. Now check the numer of clusters

# In[ ]:


pca = PCA(n_components=30)
image_PCA = pca.fit_transform(encodeddf)
Nc = range(1,30)
kmeans = [KMeans(i) for i in Nc]
score = [kmeans[i].fit(image_PCA).score(image_PCA) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve to evaluate number of clusters')
plt.show()


# We'll use 12 clusters again

# In[ ]:


n_clusters=12
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
encoding_kmeans = kmeans.fit_predict(image_PCA)


# In[ ]:


encoding_clusters = np.zeros((n_clusters, encodeddf.iloc[0,:].shape[0]), dtype=np.float64)
clustercounts = np.zeros(n_clusters,dtype=np.int)
for i in range(len(encoding_kmeans)):
    for j in range(n_clusters):
        if encoding_kmeans[i] == j:
            encoding_clusters[j] += encodeddf.iloc[i,:]
            clustercounts[j] += 1
encoding_clustersdf = pd.DataFrame(encoding_clusters)
for j in range(n_clusters):
    encoding_clustersdf[j] = encoding_clustersdf[j] / clustercounts[j]


# Printing out what the centroid of clusters of encoded features look like (hidden)

# In[ ]:


encoding_clustersdf


# In[ ]:


from skimage.filters import gaussian
fig = plt.figure(figsize=(15,10))
for i in range(len(encoding_clustersdf)):
    decoded = decoderCNN.predict(encoding_clustersdf.iloc[i,:].values.reshape((1,16, 16, 4)))
    plt.subplot(4,5,i+1)
    imgtoshow = gaussian(decoded.reshape(256,256).T, sigma=2)
    plt.imshow(imgtoshow, cmap=plt.cm.bone)
    plt.title('Cluster '+str(i)+' Size: '+str(clustercounts[i]))
plt.tight_layout()
plt.suptitle('Images of Final Clustering Result using Encoded Features as basis of clustering',fontsize=16,y=1.08)
plt.show()


# # Stop here for now.
# 
# A couple of next steps from here. 
# 
# Further EDA: I still want to visualize the clustering onto tsne to understand the spread of the clusters. I also want to show images from each of the clusters to compare the clustered encodings vs. actual examples
# 
# Improving Prediction: The clustering can be used to make a hierarchical prediction exercise. 
# 
# 
