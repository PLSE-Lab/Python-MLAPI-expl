#!/usr/bin/env python
# coding: utf-8

# # Analysis of the Image Metadata
# 
# ## This kernel will focus on breaking apart the image metadata files and understanding the responses from the Google Vision API.
# 
# ### There are many kernels that do a great job of breaking down the main data. I believe there is still great value in the sentiment data but perhaps not as easy as simply retrieving the score and magnitude that many previous kernels do. 

# ## Current Usages of Image Metadata:
# 
# 1. Many people are using vertex_y and vertex_x inspired by kernels such as: [https://www.kaggle.com/domcastro/let-s-annoy-abhishek](https://www.kaggle.com/domcastro/let-s-annoy-abhishek) and [https://www.kaggle.com/abhishek/maybe-something-interesting-here](https://www.kaggle.com/abhishek/maybe-something-interesting-here). In this kernel I will try to break down what these values actually mean.
# 2. I will analyze the image properties section and try to evaluate the usefulness of these fields such as dominant colors. 
# 3. I will spend some time discussing the faceAnnotations section and the labelAnnotations sections.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Load the initial data
train_data = pd.read_csv("../input/train/train.csv")


# Any results you write to the current directory are saved as output.


# **Let's read in the metadata now, (referenced from the kernels linked above)**

# In[ ]:


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
face_annotations = []
label_annotations = []
text_annotations = []
nf_count = 0
nl_count = 0
for pet in train_data.PetID:
    try:
        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:
            data = json.load(f)
        face_annotations.append(data.get('faceAnnotations', []))
        text_annotations.append(data.get('textAnnotations', []))
        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        vertex_xs.append(vertex_x)
        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
        vertex_ys.append(vertex_y)
        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
        bounding_confidences.append(bounding_confidence)
        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
        bounding_importance_fracs.append(bounding_importance_frac)
        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
        dominant_blues.append(dominant_blue)
        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
        dominant_greens.append(dominant_green)
        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
        dominant_reds.append(dominant_red)
        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
        dominant_pixel_fracs.append(dominant_pixel_frac)
        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
        dominant_scores.append(dominant_score)
        if data.get('labelAnnotations'):
            label_annotations.append(data['labelAnnotations'])
            label_description = data['labelAnnotations'][0]['description']
            label_descriptions.append(label_description)
            label_score = data['labelAnnotations'][0]['score']
            label_scores.append(label_score)
        else:
            nl_count += 1
            label_annotations.append([])
            label_descriptions.append('nothing')
            label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_annotations.append([])
        label_descriptions.append('nothing')
        label_scores.append(-1)
        face_annotations.append([])
        text_annotations.append([])

print(nf_count)
print(nl_count)
train_data.loc[:, 'vertex_x'] = vertex_xs
train_data.loc[:, 'vertex_y'] = vertex_ys
train_data.loc[:, 'bounding_confidence'] = bounding_confidences
train_data.loc[:, 'bounding_importance'] = bounding_importance_fracs
train_data.loc[:, 'dominant_blue'] = dominant_blues
train_data.loc[:, 'dominant_green'] = dominant_greens
train_data.loc[:, 'dominant_red'] = dominant_reds
train_data.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
train_data.loc[:, 'dominant_score'] = dominant_scores
train_data.loc[:, 'label_description'] = label_descriptions
train_data.loc[:, 'label_score'] = label_scores
train_data.sample(5)


# ## Crop Hints Annotations
# 
# **Let's break down the vertex_x and vertex_y fields first. **
# 
# One key thing to note is that the Google Vision API did not produce very useful crop hints in the majority of cases. This results to the vertex_x and vertex_y fields being quite misleading in terms of importance.
# 
# Here's the feature importances for my LGBM model:
# 
# <img src="http://imgur.com/0IuGXV2.png" width="250px">
# 
# 
# 

# As shown, vertex_y and vertex_x are very important. But what do they represent?
# Ideally, Google API will find a subset of the image that contains the majority of the data in the image and create a bounding box around this. This is what crop hints is used for. In practice, especially on this dataset, this is not very helpful. 

# In[ ]:


count = 0
pet_ids = []
index = -1
for pet_id in train_data.PetID:
    index += 1
    try:
        im = Image.open('../input/train_images/%s-1.jpg' % pet_id)
        width, height = im.size
        vertex_y = vertex_ys[index]
        vertex_x = vertex_xs[index]
        if vertex_y < height - 10 or vertex_x < width - 10:
            pet_ids.append(pet_id)
            count += 1
    except:
        pass
    

print(f"{count} pets have their profile picture's crop hint significantly different than the whole image itself")


# # **383 Pets!**
# 
# ***Out of the 14993 pets in the training data, only 383 pets have a crop hint that is more than 10 pixels off the edge of the corner of the image. What this means is that in the vast majority of cases, the Google Vision API is simply reporting the whole image as a crop hint annotation.***
# 
# **Therefore, the vertex_x and vertex_y are simply referring to the size of the image in the vast majority of cases. Why is this so important to the model? Perhaps bigger pictures means higher quality of the posting - both in image quality and in the listing itself -  and such, the size of the image (vertex_x and vertex_y are significant predictors of the AdoptionSpeed. **
# 
# ## To test this hypothesis, I will replace the vertex_x and vertex_y fields with the Image width and height and rerun the model. I will compare the feature importances to see how the model changes.
# 
# <img src="https://imgur.com/9Skh2Tb.png" width="250px">
# 
# Simple comparison shows that the model behaves exactly the same. I actually got a slight improvment on my CV using the sizes instead of the data from the json.

# ## Face Annotations
# 
# Google Vision returns an astonishing amount of data for a detected face. In this competition we aren't really concerned about human faces but animals. Let's see if the data about human presence is useful. 

# In[ ]:


from collections import Counter
train_data.loc[:, 'num_faces'] = list(map(lambda x: len(x), face_annotations))
sns.countplot(x='num_faces', data=train_data)
Counter(train_data['num_faces'])


# There are very few profile images with faces. I am sure if we included all images these numbers would increase, perhaps I will look into importance of secondary pictures later. Let's check if the AdoptionSpeed changes with the number of faces.

# In[ ]:


sns.catplot(x='num_faces', y='AdoptionSpeed', data=train_data, kind='bar')
plt.title("AdoptionSpeed based on number of faces in image")
plt.show()


# Doesn't look like there is any definite correlation between number of faces and AdoptionSpeed. Regardless, the fraction of images that actually report faceAnnotations is low enough that it couldn't be used very reliably. I think there could be a isFacePresent feature but I don't feel very comfortable with the large confidence intervals shown on the graph. 
# 
# Let's move on to labelAnnotations.
# 
# ## Label Annotations

# In[ ]:


def dog_or_cat(label_annotation):
    if len(label_annotation) > 0:
        desc = label_annotation[0]['description']
        if desc == 'cat' or desc == 'dog':
            return True
    return False

animal_scores = []
animal_topics = []
indices = []
for label_annotation in label_annotations:
    score = -1
    topic = -1
    index = 0
    for label in label_annotation:
        if label['description'] == 'dog' or label['description'] == 'cat':
            score = label['score']
            topic = label['topicality']
            indices.append(index)
            break
        index += 1
    if score == -1:
        indices.append(-1)
    animal_scores.append(score)
    animal_topics.append(topic)

train_data.loc[:, 'dominant_animal_label'] = list(map(lambda x: dog_or_cat(x), label_annotations))
train_data.loc[:, 'animal_scores'] = animal_scores
train_data.loc[:, 'animal_topic'] = animal_topics
sns.catplot(x='dominant_animal_label', y='AdoptionSpeed', data=train_data, kind='bar')
plt.title("AdoptionSpeed based on whether first(strongest) label is Dog/Cat")
plt.show()
print("Count of Indices the Animal Label Occurs In: ", Counter(indices))
print("Count of Dominant Animal Label (True/False): ", Counter(train_data['dominant_animal_label']))
train_data.loc[:, 'animal_index'] = indices
sns.catplot(x='animal_index', y='AdoptionSpeed', data=train_data, kind='bar')
plt.title("AdoptionSpeed based on Index of Animal Label (-1 == Not Found)")
plt.show()


# The above code checked whether the dominant label of the image was either dog or cat. The reasoning is that the image should be obviously a dog or cat or the AdoptionSpeed will be lower. This is clearly shown in the above graph - images whose predominant label was a dog or cat were adopted faster than those whose label were not. 
# 
# Looking at these alternate labels, I noticed a lot of them were 'dog breed' and 'dog like mammal'. Further testing will need to be done to check if these labels should be included.
# 
# The last graph shows how the index of the animal_label is a great predictor of the AdoptionSpeed. The higher the index the longer it takes the pet to be adopted. **Remember that -1 is not found at all, and this is why the AdoptionSpeed is so large for that bar. ** *I would take the point estimates after index 4 with a grain of salt, the frequencies are too low to make any sort of conclusion. (also evidenced by the confidence intervals)*
# 
# **So this analysis shows that using the label_score of the first label probably isn't a good idea. We would likely want to include the label_score of the label which is either cat/dog (or perhaps these other somewhat related labels) and search among all the labels until we found that score. For example, in the models referenced by the kernels I linked to, a image with a lion of score .93 would be treated the same as an image with dog/cat of score .93. The label_description field should help alleviate that issue but it doesn't seem to have a very strong importance. **
# 
# Let's compare the distributions of blindly taking the dominant label_score and taking the label_score of the dog/cat field. 

# In[ ]:


sns.catplot(x="AdoptionSpeed", y="animal_scores", data=train_data, kind="strip")
plt.show()
sns.countplot(x="AdoptionSpeed", data=train_data).set_title("Distribution of AdoptionSpeed (All Data)")
plt.show()
sns.countplot(x="AdoptionSpeed", data=train_data.loc[train_data['animal_scores'] == -1]).set_title("Distribution of AdoptionSpeed with no Animal Label")
plt.show()


# 1. The first graph shows some correlation but the -1 throws the plot off. I will replot this graph without the -1's below. Meanwhile let's try to analyze how having no cat/dog label (-1's) changes the AdoptionSpeed.
# 2. Comparing the two graphs we see an astonishing difference in the distribution of AdoptionSpeeds. This shows that looking for the presence of a dog/cat label is beneficial; *rather than just looking to see if a label exists in general.* If an image doesn't have a cat/dog label it's almost certainly not going to be adopted (Predicted AdoptionSpeed: 4).** I can definitely see the hasAnimalLabel as a valid feature in models. **
# 
# **Again further work will need to be done to see what classifies as an animal label. Dog/Cat is what we choose now but what about 'Dog Breed' and 'Dog Like Mammal' (two other common dominant labels in the data)? **
# 
# Let's replot the stripplots without the -1 messing with the scale.

# In[ ]:


sns.catplot(x="AdoptionSpeed", y="animal_scores", data=train_data.loc[train_data['animal_scores'] != -1], kind="strip")
plt.title("Animal Label Score distribution for various AdoptionSpeeds")
plt.show()
sns.catplot(x="AdoptionSpeed", y="label_score", data=train_data.loc[train_data['label_score'] != -1], kind="strip")
plt.title("First Label Score distribution for various AdoptionSpeeds")
plt.show()


# **The first graph (with the Animal Label Score rather than the first Label Score) provides a far better correlated and predictable distribution than the second graph. I predict using this feature instead of label_score will bring improvements to many models. **

# I have a lot more crazy/stretch ideas for labelAnnotations but I will go ahead and move into textAnnotations and perhaps return to this section in the future.
# 
# ## Text Annotations
# 
# Let's check how many images actually have the textAnnotations field:

# In[ ]:


#Check if has textAnnnotations

has_text_annotations = list(map(lambda x: len(x), text_annotations))
print("True=Has Text Annotations in JSON: ", Counter(has_text_annotations))
train_data.loc[:, 'has_text'] = has_text_annotations
sns.catplot(x="has_text", y="AdoptionSpeed", data=train_data, kind="bar")
plt.title("AdoptionSpeed based on whether text is present in the profile")
plt.show()

def mapToDescLen(text_annotation):
    if (len(text_annotation) == 0):
        return 0
    return len(text_annotation[0]['description'])

text_length = list(map(lambda x: mapToDescLen(x), text_annotations))
train_data.loc[:, 'text_length'] = text_length

sns.catplot(x='AdoptionSpeed', y='text_length', data=train_data, kind='strip')
plt.title("Plot of text length in JSON vs AdoptionSpeed")
plt.show()

sns.catplot(x='AdoptionSpeed', y='text_length', data=train_data, kind='bar')
plt.title("Mean Lengths per AdoptionSpeed of Text in Profile Picture")
plt.show()


# Not really sure how helpful length of text found in image is. Could be helpful to do some similar analysis on the text as we did with the description but not sure. Only useful feature seems **isTextPresent**.

# ## Image Properties Annotations coming soon

# I will update the kernel with analysis about dominant colors soon. Consider upvoting if you learned something useful :)
