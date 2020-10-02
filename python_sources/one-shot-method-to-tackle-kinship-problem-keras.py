#!/usr/bin/env python
# coding: utf-8

# # 'You seem familiar' - One-shot approach to tackle kinship problem
# 
# In this notebook we are going to solve this kinship problem using the popular one-shot learning approach and build a embedding generator to find the cosine-similarity amongst the images of people from a family. Playing with cosine similarity, we are going to design a model that is able to recognize the people of a family and people related by blood.

# ### Understanding the dataset
# The dataset comprises of files seperated into training and testing folders with a relation dataframe _train_relationships.csv_ pointing to the images of blood related members.
# Files stored in the train folder are as follows:
#     
# ```bash
# ./train
# ./train/F@@@@      - @@@@ denoting the family number
# ./train/F@@@@/MID$ - $ denoting the member of the family
# ```
# Example:
# 
# ```bash
#    +---------------+------------+
#    | F0002/MID1    | F0002/MID3 |
#    | F0002/MID2	| F0002/MID3 |
#    +---------------+------------+
# ```
# This represents that member _MID1_ and _MID2_ of family _F0002_ is in blood relation with _MID3_ whereas _MID1_ and _MID2_ are not in blood-relationship with each other.

# In[ ]:


# calling basic imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotting library
from matplotlib import pyplot as plt
import seaborn as sns

# Additional support libraries
import os
print(os.listdir("../input"))

# Library for reading images 
from PIL import Image

# Random
import random
from tqdm import tqdm_notebook


# ### Reading the dataset

# In[ ]:


train  = pd.read_csv('../input/recognizing-faces-in-the-wild/train_relationships.csv')
train.tail()


# In[ ]:


TRAIN_BASE = '../input/recognizing-faces-in-the-wild/train/'
families = sorted(os.listdir(TRAIN_BASE))
print('We have {} families in the dataset'.format(len(families)))
print(families[:5])


# In[ ]:


members = {i:sorted(os.listdir(TRAIN_BASE+i)) for i in families}


# In[ ]:


TEST_BASE='../input/recognizing-faces-in-the-wild/test/'
test_images_names = os.listdir(TEST_BASE)
test_images_names[:5]


# ### Visualizing the dataset
# 
# In this part of the kernel, we are going to visualize the dataset images to get the gist of it. The dataset structure has been discussed above and here we are going to visualize the images.
# 
# First let's being with building the support functions to view the dataset images.

# In[ ]:


def load_img(PATH): return np.array(Image.open(PATH))

def plots(ims, figsize=(12,6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])


# In[ ]:


def plot_relations(df, BASE='../input/recognizing-faces-in-the-wild/train/', rows=1, titles=None):
    tdf = df[:rows]
    tdf1 = tdf.p1
    tdf2 = tdf.p2
    figsize=(5,3*rows)
    f = plt.figure(figsize=figsize)
    x = 0
    for i in range(rows):
        sp = f.add_subplot(rows, 2, x+1)
        sp.axis('Off')
        x+=1
        image_path = os.path.join(BASE,tdf1[i])
        im = os.listdir(image_path)[-1]
        sp.set_title(tdf1[i], fontsize=16)
        plt.imshow(load_img(os.path.join(image_path, im)))
        sp = f.add_subplot(rows, 2, x+1)
        x+=1
        sp.axis('Off')
        image_path = os.path.join(BASE,tdf2[i])
        im = os.listdir(image_path)[-1]
        sp.set_title(tdf2[i], fontsize=16)
        plt.imshow(load_img(os.path.join(image_path, im)))
        
plot_relations(train, rows=10)


# In[ ]:


test_images=np.array([load_img(os.path.join(TEST_BASE,image)) for image in test_images_names])


# In[ ]:


test_images.shape


# In[ ]:


plots(test_images[:15], rows=3)


# #### Average test faces

# In[ ]:


plt.imshow(test_images.sum(axis=0)//test_images.shape[0])


# In[ ]:


avg_face = []
u=0
for family in families[u:u+1]:
    for member in os.listdir(os.path.join(TRAIN_BASE,family)):
        for image in os.listdir(os.path.join(TRAIN_BASE, family, member)):
            avg_face.append(load_img(os.path.join(TRAIN_BASE, family, member, image)))
avg_face=np.array(avg_face)
plt.imshow(avg_face.sum(axis=0)//avg_face.shape[0])


# ### Building the model
# 
# In order to generate the face embeddings, we are going to use the _vgg_face_ model trained on the faces dataset to generate images. We are going to use the _channel_first_ method in keras and change every image to channel first as per required. In order to change the keras configration to channel first, we are going to alter the _keras.json_ file in user home or by simply altering the backend.

# In[ ]:


from keras import backend as K


# In[ ]:


K.set_image_data_format('channels_first')


# In[ ]:


import keras
from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation


# As we are using the VGG16 as the base model here, so we are going to build VGG16 as per the diagram below
# 
# ![vgg16](https://d2mxuefqeaa7sj.cloudfront.net/s_8C760A111A4204FB24FFC30E04E069BD755C4EEFD62ACBA4B54BBA2A78E13E8C_1491022251600_VGGNet.png)
# 
# I have changed the last layer to 2622 as per the VGG16 pretrained configrations that could be found here: [VGG16_facenet_keras](https://drive.google.com/uc?id=0B4ChsjFJvew3NkF0dTc1OGxsOFU&export=download)

# In[ ]:


def vgg_face(weights_path=None):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, (3, 3), activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, (3, 3), activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, (3, 3), activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, (3, 3), activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, (3, 3), activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, (3, 3), activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, (3, 3), activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, (3, 3), activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, (3, 3), activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, (3, 3), activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, (3, 3), activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, (3, 3), activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, (3, 3), activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    fc6 = Convolution2D(4096, (7, 7), activation='relu', name='fc6')(pool5)
    fc6_drop = Dropout(rate = 0.5)(fc6)
    fc7 = Convolution2D(4096,( 1, 1), activation='relu', name='fc7')(fc6_drop)
    fc7_drop = Dropout(rate = 0.5)(fc7)
    fc8 = Convolution2D(2622, (1, 1), name='fc8')(fc7_drop)
    flat = Flatten()(fc8)
    out = Activation('softmax')(flat)

    model = Model(input=img, output=out)

    if weights_path:
        model.load_weights(weights_path)

    return model


# In[ ]:


vgg_facenet = vgg_face('../input/vgg16-facenet-model/vgg-face-keras.h5')


# In[ ]:


vgg_facenet.summary()


# Now let's try to generate the embeddings for 2 images from the dataset

# In[ ]:


im = Image.open('../input/recognizing-faces-in-the-wild/train/F0002/MID1/P00009_face3.jpg')
im = np.array(im).astype(np.float32)
im2 = Image.open('../input/recognizing-faces-in-the-wild/train/F0002/MID3/P00014_face1.jpg')
im2 = np.array(im2).astype(np.float32)
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
im2 = im2.transpose((2,0,1))
im2 = np.expand_dims(im2, axis=0)
np.concatenate([im,im2]).shape


# In[ ]:


out = vgg_facenet.predict(np.concatenate([im,im2]))


# In[ ]:


def cosine_similarity(a,b):
    return np.sum(np.multiply(a,b))/np.multiply( np.sqrt(np.sum(np.power(a,2))),(np.sqrt(np.sum(np.power(b,2)))))

def distance(x, y):
    return np.linalg.norm(x - y)


# In[ ]:


print(cosine_similarity(out[0], out[1]), distance(out[0], out[1]))


# Now finally reading the testset from the test_folder and predicting on the values from sample submission.

# In[ ]:


test_images = os.listdir(TEST_BASE)
test = np.array([load_img(os.path.join(TEST_BASE, i)) for i in test_images])
test_emb = vgg_facenet.predict(test.transpose(0,3,1,2))
print(test.shape, test_emb.shape)


# In[ ]:


image_mapping = {img:idx for idx, img in enumerate(test_images)}


# In[ ]:


submission = pd.read_csv('../input/recognizing-faces-in-the-wild/sample_submission.csv')
req_mapping = [i.split('-') for i in submission.img_pair]


# In[ ]:


vector_distances=[]
for i in req_mapping:
    a = i[0]
    b = i[1]
    dis = distance(test_emb[image_mapping[a]], test_emb[image_mapping[b]])
    vector_distances.append(dis)
vector_distances=np.array(vector_distances)


# In[ ]:


total_sum = vector_distances.sum()


# Now from the vector_distances we have calulated above, we are now going to transform these distances into probablity of being same.

# In[ ]:


probs = []
for dist in vector_distances:
    prob = np.sum(vector_distances[np.where(vector_distances <= dist)[0]])/total_sum
    probs.append(1 - prob)


# In[ ]:


vector_distances.shape


# In[ ]:


np.sum(vector_distances[np.where(vector_distances <= dist)[0]])/total_sum


# In[ ]:


submission.is_related = probs


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:


test_emb.shape


# In[ ]:


import sklearn.manifold


# In[ ]:


tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_emb_matrix_2d = tsne.fit_transform(test_emb)


# In[ ]:


points = pd.DataFrame(
    [
        (name, coords[0], coords[1])
        for name, coords in [
            (img, all_emb_matrix_2d[image_mapping[img]])
            for img in test_images
        ]
    ],
    columns=["name", "x", "y"]
)


# In[ ]:


sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(15, 8))


# In[ ]:


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


# In[ ]:


plot_region(x_bounds=(70.0, 80.0), y_bounds=(-10.0, 0.0))


# In[ ]:


plt.imshow(test[image_mapping['face'+'03198'+'.jpg']])


# In[ ]:


plt.imshow(test[image_mapping['face'+'05866'+'.jpg']])


# In[ ]:




