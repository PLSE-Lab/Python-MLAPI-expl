#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization library
import seaborn as sns 
import glob as gb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dfclubnames = pd.read_csv('../input/ClubNames.csv')
dffullnames = pd.read_csv('../input/FullData.csv')
dfnationalnames = pd.read_csv('../input/NationalNames.csv')
dfplayernames = pd.read_csv('../input/PlayerNames.csv')


# In[2]:


dffullnames.head()


# 
# **Count all players whose contract is going to expire before 2019 **

# In[3]:


dffullnames[dffullnames['Contract_Expiry']<=2019].Contract_Expiry.count()


# **Total Penalties from all players of Portugal **

# In[4]:


dffullnames[dffullnames['Nationality']<='Portugal'].Penalties.sum()


# ****Get First 1000 Rows and Selected Columns from fullnames dataframe****

# In[5]:


ws=dffullnames.loc[:1000,('Name','Nationality','Long_Shots','Freekick_Accuracy','Penalties','Stamina','Crossing','Shot_Power','Finishing')]


# **Cross Analysis of multiple Variables**

# In[6]:


sns.lmplot(x='Crossing', y='Stamina', data=ws)
plt.title('Stamina vs Crossing')
plt.show()

sns.lmplot(x='Long_Shots', y='Shot_Power', data=ws)
plt.title('Long Shots vs Shot Power')
plt.show()

sns.lmplot(x='Freekick_Accuracy', y='Stamina', data=ws)
plt.title('Freekick_Accuracy vs Stamina')
plt.show()

sns.lmplot(x='Finishing', y='Penalties', data=ws)
plt.title('Finishing vs Penalties')
plt.show()

# sns.lmplot(x='Dribbling', y='Weak_foot', data=dffullnames, col='Dribbling')
# plt.title('Dribbling vs Weak_foot')
# plt.show()


# **Regression plot between Crossing vs Stamina**

# In[7]:


plt.scatter(ws['Crossing'], ws['Stamina'], label='data', color='red', marker='o')
sns.regplot(x='Crossing', y='Stamina', data=ws, order=7)


# **Residual plot between Crossing vs Stamina**

# In[8]:


sns.residplot(x='Crossing', y='Stamina',data=ws,color='indianred')


# **Strip Plots**

# In[9]:


wsl10=ws[-10:]
plt.subplot(2,2,1)
sns.stripplot(x='Long_Shots', y='Nationality', data=wsl10)
plt.ylabel('Nationality')

plt.subplot(2,2,2)
sns.stripplot(x='Penalties', y='Name', data=wsl10, jitter=True , size=3)
plt.ylabel('Name')
plt.tight_layout()

plt.show()


# **Swarm Plots**

# In[10]:



sns.swarmplot(x='Penalties',y='Nationality',data=wsl10)
plt.title('Penalitties vs Nationality')
plt.show()


plt.axis([0,130,0,15])
sns.swarmplot(x='Penalties',y='Nationality',data=wsl10,hue='Name', color='red')
plt.legend(loc=5)
plt.tight_layout()
plt.show()


# **Joint Plot**

# In[11]:



sns.jointplot(x='Freekick_Accuracy',y='Long_Shots',data=ws, kind='scatter', color='r')





sns.jointplot(x='Freekick_Accuracy',y='Long_Shots',data=ws, kind='hex', color='g')





sns.jointplot(x='Freekick_Accuracy',y='Long_Shots',data=ws, kind='resid')

plt.tight_layout()


# In[12]:



cmap = sns.cubehelix_palette(light=.8, as_cmap=True)
sns.jointplot(x='Freekick_Accuracy',y='Long_Shots',data=ws, kind='kde',cmap=cmap, shade=True)
sns.jointplot(x='Freekick_Accuracy',y='Long_Shots',data=ws, kind='reg')
plt.show()


# **Pair plot**

# In[13]:


wsf10=ws.loc[:10]
wsf10
# ws
sns.pairplot(wsf10, hue='Name', kind='reg')


# **Extracting a histogram from a grayscale image**
# 
# 
# list all files available at 
# '../input/ClubPictures/'
# 
# 

# In[44]:


gb.glob("../input/Pictures_f/*.*")


# In[21]:


image = plt.imread('../input/ClubPictures/Liverpool.png')
plt.imshow(image, cmap='jet')
plt.colorbar()


# In[22]:


lum_img = image[:,:,0]
plt.imshow(lum_img)
plt.colorbar()
# plt.imshow(lum_img, cmap="hot")


# In[37]:


plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray')

# Assign pixels the flattened 1D numpy array image.flatten() 
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
plt.xlim((0,1))
plt.title('Normalized histogram')
plt.hist(pixels, bins=500, color='red', alpha=0.4, range=(0,15), normed=True)
plt.show()


# In[59]:


image_list = []
for filename in gb.glob('../input/Pictures_f/*.*'): #assuming gif
    im=plt.imread(filename)
    image_list.append(im)

plt.imshow(image_list[])    
# for image in image_list:
#     plt.imshow(image)


# In[46]:


# Load the image into an array: image
image = plt.imread('../input/Pictures_f/Li Dongna.png')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
pdf = plt.hist(pixels, bins=64, range=(0,256), normed=False,color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()

# Display a cumulative histogram of the pixels
cdf = plt.hist(pixels, bins=64, range=(0,256),cumulative=True, normed=True,color='blue', alpha=0.4)
               
# Specify x-axis range, hide axes, add title and display plot
plt.xlim((0,256))
plt.grid('off')
plt.title('PDF & CDF (original image)')
plt.show()

