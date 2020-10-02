#!/usr/bin/env python
# coding: utf-8

# ## Data Fields in the train and test set
# 
# >  PetID - Unique hash ID of pet profile
# >
# > AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
# > 
# > Type - Type of animal (1 = Dog, 2 = Cat)
# > 
# > Name - Name of pet (Empty if not named)
# > 
# >  Age - Age of pet when listed, in months
# >  
# > Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
# >
# >Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
# >  
# >  Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
# >  
# >  Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
# > 
# >  Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
# > 
# >  Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
# > 
# > MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
# > 
# > FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
# > 
# > Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure
# > 
# > Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# > 
# > Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# > 
# > Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
# > 
# > Quantity - Number of pets represented in profile
# > 
# > Fee - Adoption fee (0 = Free)
# > 
# > State - State location in Malaysia (Refer to StateLabels dictionary)
# > 
# > RescuerID - Unique hash ID of rescuer
# > 
# > VideoAmt - Total uploaded videos for this pet
# > 
# > PhotoAmt - Total uploaded photos for this pet
# > 
# > Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
state_labels = pd.read_csv("../input/state_labels.csv")
breed_labels = pd.read_csv("../input/breed_labels.csv")
color_labels = pd.read_csv("../input/color_labels.csv")


# In[ ]:


train.head()


# In[ ]:


state_labels


# In[ ]:


breed_labels


# In[ ]:


color_labels


# In[ ]:





# ## We can see the unique Values -> 
# ## 8 colour Types
# ## 307 Types of breed
# ## 15 States

# In[ ]:




