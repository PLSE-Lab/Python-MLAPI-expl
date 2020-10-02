#!/usr/bin/env python
# coding: utf-8

# # Product recommendation based on visual similarity
# 
# 
# The goal of this experiment is to make a very basic recommender system: for a given fashion product, we want to recommend products that look similar. 
# 
# This kind of recommender system is often used when browsing shopping websites. They usually appear on product pages as a "you may also like" section.
# 
# The idea behind this recommender system is simple: if a customer is showing interest towards a product by browsing its page, he may also be interested by products that are similar.
# 
# 
# ## How to proceed ?
# 
# We will used a pre-trained CNN model from Keras to extract the image features.
# 
# Then we will compute similarities between the different products using the previously extracted image features.
# 
# Other type of information can be used for this purpose such as the product category, size, color, etc. if the data is available, but that is not the case here.

# ## 0. imports and parameters setup

# In[ ]:


# imports

from keras.applications import vgg16
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# In[ ]:


# parameters setup

imgs_path = "../input/style/"
imgs_model_width, imgs_model_height = 224, 224

nb_closest_images = 5 # number of most similar images to retrieve


# ## 1. load the VGG pre-trained model from Keras
# 
# Keras module contains several pre-trained models that can be loaded very easily. 
# 
# For our recommender system based on visual similarity, we need to load a Convolutional Neural Network (CNN) that will be able to interpret the image contents.
# 
# In this example we will load the VGG16 model trained on imagenet, a big labeled images database.
# 
# If we take the whole model, we will get an output containing probabilities to belong to certain classes, but that is not what we want.
# 
# We want to retrieve all the information that the model was able to get in the images.
# 
# In order to do so, we have to remove the last layers of the CNN which are only used for classes predictions.

# In[ ]:


# load the model
vgg_model = vgg16.VGG16(weights='imagenet')

# remove the last layers in order to get features instead of predictions
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

# print the layers of the CNN
feat_extractor.summary()


# ## 2. get the images paths

# In[ ]:


files = [imgs_path + x for x in os.listdir(imgs_path) if "png" in x]

print("number of images:",len(files))


# ## 3. feed one image into the CNN
# 
# First we observe what output we get when putting one image into the CNN.
# 
# The following steps are:
# - loading the image
# - preparing the image to feed it into the CNN
# - get the CNN output which will correspond to the image features

# In[ ]:


# load an image in PIL format
original = load_img(files[0], target_size=(imgs_model_width, imgs_model_height))
plt.imshow(original)
plt.show()
print("image loaded successfully!")


# In[ ]:


# convert the PIL image to a numpy array
# in PIL - image is in (width, height, channel)
# in Numpy - image is in (height, width, channel)
numpy_image = img_to_array(original)

# convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# we want the input matrix to the network to be of the form (batchsize, height, width, channels)
# thus we add the extra dimension to the axis 0.
image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)

# prepare the image for the VGG model
processed_image = preprocess_input(image_batch.copy())


# In[ ]:


# get the extracted features
img_features = feat_extractor.predict(processed_image)

print("features successfully extracted!")
print("number of image features:",img_features.size)
img_features


# ## 4. feed all the images into the CNN
# 
# We were able to do the feature extraction process for one image. Now let's do it for all our images!

# In[ ]:


# load all the images and prepare them for feeding into the CNN

importedImages = []

for f in files:
    filename = f
    original = load_img(filename, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    importedImages.append(image_batch)
    
images = np.vstack(importedImages)

processed_imgs = preprocess_input(images.copy())


# In[ ]:


# extract the images features

imgs_features = feat_extractor.predict(processed_imgs)

print("features successfully extracted!")
imgs_features.shape


# # 5. compute cosine similarities
# 
# Now that we have features for every image, we can compute similarity metrics between every image couple.
# 
# We will use here the cosine similarity metric.

# In[ ]:


# compute cosine similarities between images

cosSimilarities = cosine_similarity(imgs_features)

# store the results into a pandas dataframe

cos_similarities_df = pd.DataFrame(cosSimilarities, columns=files, index=files)
cos_similarities_df.head()


# # 6. retrieve most similar products
# 
# The final step is to implement a function that, for any given product, returns the visually most similar products.

# In[ ]:


# function to retrieve the most similar products for a given one

def retrieve_most_similar_products(given_img):

    print("-----------------------------------------------------------------------")
    print("original product:")

    original = load_img(given_img, target_size=(imgs_model_width, imgs_model_height))
    plt.imshow(original)
    plt.show()

    print("-----------------------------------------------------------------------")
    print("most similar products:")

    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images+1]

    for i in range(0,len(closest_imgs)):
        original = load_img(closest_imgs[i], target_size=(imgs_model_width, imgs_model_height))
        plt.imshow(original)
        plt.show()
        print("similarity score : ",closest_imgs_scores[i])


# In[ ]:


retrieve_most_similar_products(files[1])


# In[ ]:


retrieve_most_similar_products(files[2])


# In[ ]:


retrieve_most_similar_products(files[3])


# In[ ]:


retrieve_most_similar_products(files[4])


# In[ ]:


retrieve_most_similar_products(files[5])


# # Conclusion
# We saw above that this very basic recommender system is able to find similar products accurately: most of the time the retrieved products have the same purpose and even look very similar.
# 
# This could be incorporated directly into a website using a web framework such as Flask.
