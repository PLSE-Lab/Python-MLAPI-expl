#!/usr/bin/env python
# coding: utf-8

# # DAY-2: 20 Plots with the Inferences I made with them...

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# ## Pls,UPVOTE if You Liked my notebook

# Normal Dude

# In[ ]:



# Location of the image dir
img_dir = '../input/chest-xray-pneumonia/chest_xray/train/NORMAL/'
# Adjust the size of your images
plt.figure(figsize=(8,8))
norimg = plt.imread(os.path.join(img_dir,'IM-0125-0001.jpeg'))
plt.imshow(norimg, cmap='gray')
plt.colorbar()
plt.title('Normal Chest Xray Image')
plt.axis('on')
print(f"The dimensions of the image are {norimg.shape[0]} pixels width and {norimg.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {norimg.max():.4f} and the minimum is {norimg.min():.4f}")
print(f"The mean value of the pixels is {norimg.mean():.4f} and the standard deviation is {norimg.std():.4f}")    
plt.show()


# ## 1-Inference:it's funny

# Virus Dude

# In[ ]:



# Location of the image dir
img_dir = '../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'
# Adjust the size of your images
plt.figure(figsize=(6,6))
virimg= plt.imread(os.path.join(img_dir,'person1000_virus_1681.jpeg'))
plt.imshow(virimg, cmap='gray')
plt.colorbar()
plt.title('Virus Infected Pneumonia')
plt.axis('on')
print(f"The dimensions of the image are {virimg.shape[0]} pixels width and {virimg.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {virimg.max():.4f} and the minimum is {virimg.min():.4f}")
print(f"The mean value of the pixels is {virimg.mean():.4f} and the standard deviation is {virimg.std():.4f}")    
# Adjust subplot parameters to give specified padding
plt.show()    


# ## 2-Inference:it's funny

# In[ ]:


# Adjust the size of your images
plt.figure(figsize=(6,6))
bacimg = plt.imread(os.path.join(img_dir,'person1000_bacteria_2931.jpeg'))
plt.imshow(bacimg, cmap='gray')
plt.colorbar()
plt.title('Bacteria Infected Pneumonia Chest Xray')
plt.axis('on')
print(f"The dimensions of the image are {bacimg.shape[0]} pixels width and {bacimg.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {bacimg.max():.4f} and the minimum is {bacimg.min():.4f}")
print(f"The mean value of the pixels is {bacimg.mean():.4f} and the standard deviation is {bacimg.std():.4f}")       
# Adjust subplot parameters to give specified padding
plt.show() 


# ## 3-Inference:it's funny

# Investigate Pixel value Distributiion

# In[ ]:


# Plot a histogram of the distribution of the pixels
sns.distplot(norimg.ravel(), 
             label=f'Pixel Mean {np.mean(norimg):.4f} & Standard Deviation {np.std(norimg):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Normal Chest Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')


# ## 4-Inference-Fairly managed 

# In[ ]:


# Plot a histogram of the distribution of the pixels
sns.distplot(virimg.ravel(), 
             label=f'Pixel Mean {np.mean(virimg):.4f} & Standard Deviation {np.std(virimg):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the  Virus Infected Chest Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')


# ## 5-Inference-looks more weird distribution

# In[ ]:


# Plot a histogram of the distribution of the pixels
sns.distplot(bacimg.ravel(), 
             label=f'Pixel Mean {np.mean(bacimg):.4f} & Standard Deviation {np.std(bacimg):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the  Bacteria Infected Chest Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')


# ## 5-Inference-looks more weird distribution,another peak

# In[ ]:


data_wine=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data_wine.head()


# In[ ]:


data_wine.info()


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data_wine)


# ## 7-Inference-hardly related

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = data_wine)


# ## 8-Inference-As volatile acidity goes up quality goes down 

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='citric acid', data = data_wine)


# ## 9-Inference-As citric acid content goes up,Quality goes up too

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='chlorides', data = data_wine)


# ## 10-Inference-As Chlorides content goes up,quality goes down

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='residual sugar', data = data_wine)


# ## 11-Inference-doesn't seem to relate directly much

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='free sulfur dioxide', data = data_wine)


# ## 12-Inference:It seems quality is good if stays in 12.5-15.0

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='total sulfur dioxide', data = data_wine)


# ## 13-Inference:It seems quality is good if stays in 31-34

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='density', data = data_wine)


# ## 14-Inference-Doesn't seem that it makes a difference,but I think maybe I need to look closer...

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='pH', data = data_wine)


# ## 15-Inference-Quality Increases as acidity increases

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='sulphates', data = data_wine)


# ## 16-Inference-As sulphates Increases,Quality Increases

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='alcohol', data = data_wine)


# ## 17-Inference-As Alcohol content increases,Quality Increases

# In[ ]:


data_leag=pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data_leag.head(10)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'blueWins', y='blueDragons', data = data_leag)


# ## 18-Inference:You have a better chance at winning if you have dragons**

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'blueWins', y='blueFirstBlood', data = data_leag)


# ## 19-Inference-First blood player wins generally

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'blueWins', y='blueAssists', data = data_leag)


# ## 20-Inference-If you help more,you win more

# ## Pls,UPVOTE if You Liked my notebook
