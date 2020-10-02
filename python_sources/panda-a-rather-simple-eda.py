#!/usr/bin/env python
# coding: utf-8

# # PANDA: A rather simple EDA
# 
# ---
# 
# Hello everyone :-)
# 
# I hope you are having a good time while in lockdown, so I decided to make a constructive use of my time by brushing up on my linear algebra and working on more EDAs. Since I got positive feedback on my last EDA, i decided to make one here.

# ## Contents:
# 
# 1. <a href="#one">Introduction</a><br>
#     1-1. <a href="#general">General Exploration</a><br>
#     1-2. <a href="#dprov">Discrepancies between data providers</a>
# 2. <a href="#two">Preprocessing</a><br>
#     2-1. <a href="#bgsub">Background subtractor</a><br>
#     2-2. <a href="#gblur">Gaussian Blur</a><br>
#     2-3. <a href="#grayscale">Grayscale</a><br>
#     2-4. <a href="#circle">Circle crop</a>

# <h1 id="one"> **1. Introduction**</h1>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; import seaborn as sns
plt.style.use('seaborn-whitegrid')
import openslide
import os
import cv2
import torch
train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu


# We can view images with Gabriel's simple function:

# In[ ]:


def show_images(df, read_region=(1780,1950)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'.tiff'
        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)
        image = openslide.OpenSlide(image_path)
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region(read_region, 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')

    plt.show()
images = [
    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',
    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   
data_sample = train.loc[train.image_id.isin(images)]
show_images(data_sample)


# <h2 id="general">General exploration</h2>

# Alright great! Let's now take a look at our metadata - I am sure there's a lot we can find.

# In[ ]:


train.head()


# We will see the distribution of data providers:

# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(train.data_provider);


# We have two data providers:
# + **Karolinska Institute:** It is one of the leading cancer research centers in the world and is located in Sweden. It covers a lot of the biological fields of study in its research.
# + **Radboud University:** It is located in the Netherlands (Nijmegen to be specific)
# 
# Let's look at `isup_grade` (our target variable BTW):

# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(train.isup_grade);


# Oho... look we have a skewed distribution! Our variables are clustered towards 0 and 1! This is interesting now. Or as a meme would put it:
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSrIvNNrFtbzZ1ge-762eJl7K44nQ24dZP-nFI8YqNv4b_duxQt&usqp=CAU)

# Gleason score distributions?

# In[ ]:


plt.figure(figsize=(10, 7))
sns.countplot(train.gleason_score);


# We have quite a strange distribution here. All this does, is make the problem much much more interesting. BTW, if you are interested in what the Gleason score is check this out:

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("1Q7ERNtLcvk", height=500, width=700)


# That was rather informative!

# <h2 id="dprov">Discrepancies between data providers</h2>

# I am not exactly 100 percent sure that the data provided by each is exactly similar, so I will have to see for myself about this suspicious business with the providers. The providers are Karolinska Institue (located in Sweden) and Radboud University (located in the Netherlands)

# In[ ]:


train[train['data_provider'] == "karolinska"]


# In[ ]:


train[train['data_provider'] == "radboud"]


# Already I feel that the discrepancies in gleason scores are visible IMHO. We will need to see for ourselves as always:

# In[ ]:


plt.figure(figsize=(20, 7))
sns.countplot(train[train['data_provider'] == "radboud"].gleason_score, color="red");
plt.legend()
plt.title("Gleason score(s) of Radboud University's Data");


# So it seems like higher gleason scores are present with the Radboud University's data. What about Karolinska Institutute?

# In[ ]:


plt.figure(figsize=(20, 7))
sns.countplot(train[train['data_provider'] == "karolinska"].gleason_score, color="blue");
plt.legend()
plt.title("Gleason score(s) of Karolinska University's Data");


# Karolinska Institue and Radboud University both have significant edge cases.
# + **For Karolinska Institute it is 0+0 gleason score.**
# + **For Radboud University it is negative gleason score.**

# We can also use isup_grade as an example now for the discrepancies in the data.

# In[ ]:


plt.figure(figsize=(20, 7))
sns.countplot(train[train['data_provider'] == "karolinska"].isup_grade, color="blue");
plt.legend()
plt.title("Grade(s) of Karolinska University's Data");


# Most of the grades are clustered towards 0 and 1. What about Radboud's data?

# In[ ]:


plt.figure(figsize=(20, 7))
sns.countplot(train[train['data_provider'] == "radboud"].isup_grade, color="red");
plt.legend()
plt.title("Grade(s) of Radboud University's Data");


# All the classes have approximately the same counts (with exception of 2 of course). 

# Now that we have established the data discrepancies, let's move on to preprocessing.

# <h1 id="#two">Preprocessing<h1>

# <h2 id="#bgsub">Background subtractor</h2>

# In[ ]:


import cv2; fgbg = cv2.createBackgroundSubtractorMOG2()

def show_images(df, read_region=(1780,1950)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'.tiff'
        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)
        image = openslide.OpenSlide(image_path)
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region(read_region, 0, (256, 256))
        patch = np.array(patch)
        image = cv2.resize(patch, (256, 256))
        image= fgbg.apply(patch)
        ax[i//3, i%3].imshow(image) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')

    plt.show()
images = [
    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',
    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   
data_sample = train.loc[train.image_id.isin(images)]
show_images(data_sample)


# It seems like we should not use background subtraction. Why? Well, we lose **entire images** using background subtraction which is 100 percent horrible for our model - loss of data.

# <h2 id="gblur">Gaussian Blur</h2>

# In[ ]:


def show_images(df, read_region=(1780,1950)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'.tiff'
        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)
        image = openslide.OpenSlide(image_path)
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region(read_region, 0, (256, 256))
        patch = np.array(patch)
        image = cv2.resize(patch, (256, 256))
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)
        ax[i//3, i%3].imshow(image) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')

    plt.show()
images = [
    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',
    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   
data_sample = train.loc[train.image_id.isin(images)]
show_images(data_sample)


# Gaussian Blur does not seem to damage our images too much, Next, we move on to grayscale.

# <h2 id="grayscale">Grayscale images</h2>

# In[ ]:


def show_images(df, read_region=(1780,1950)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'.tiff'
        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)
        image = openslide.OpenSlide(image_path)
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region(read_region, 0, (256, 256))
        patch = np.array(patch)
        image = cv2.resize(patch, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ax[i//3, i%3].imshow(image) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')

    plt.show()
images = [
    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',
    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   
data_sample = train.loc[train.image_id.isin(images)]
show_images(data_sample)


# These help us  to visualize the images and the distinctions within each image more clearly. 

# <h2 id="circle">Circle crop</h2>

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img
    
def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    
    img = crop_image_from_gray(img)    
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def show_images(df, read_region=(1780,1950)):
    data = df
    f, ax = plt.subplots(3,3, figsize=(16,18))
    for i,data_row in enumerate(data.iterrows()):
        image = str(data_row[1][0])+'.tiff'
        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)
        image = openslide.OpenSlide(image_path)
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region(read_region, 0, (256, 256))
        patch = np.array(patch)
        image = cv2.resize(patch, (256, 256))
        image = circle_crop(image)
        ax[i//3, i%3].imshow(image) 
        ax[i//3, i%3].axis('off')
        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')

    plt.show()
images = [
    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',
    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   
data_sample = train.loc[train.image_id.isin(images)]
show_images(data_sample)


# It seems like circle crop could be a feasible option, but I am not exactly sure of using circle crop on our images. Why? Well, cropping has a few disadvantages:
# + **Loss of information**: Losing the possibly helpful information in the corners of our image is very damaging (potentially, yes) because of the issue of potientially cropping away the information.

# ---
# 
# # Work in progress
# 
# ---
