#!/usr/bin/env python
# coding: utf-8

# ### Context
# 
# The [Austin Animal Center](http://www.austintexas.gov/department/aac) is the largest no-kill animal shelter in the United States that provides care and shelter to over 18,000 animals each year. As part of the AAC's efforts to help and care for animals in need, the organization makes available its accumulated data and statistics as part of the city of [Austin's Open Data Initiative](https://data.austintexas.gov/).
# 
# ### Content
# 
# The data contains intakes and outcomes of animals entering the Austin Animal Center from the beginning of October 2013 to the present day. The datasets are also freely available on the [Socrata Open Data Access API](https://dev.socrata.com/) and are updated daily. 
# 
# The following are links to the datasets hosted on Socrata's Open Data:
# 
# * [Austin Animal Center Intakes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm)
# * [Austin Animal Center Outcomes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238)
# 
# The data contained in this dataset is the outcomes and intakes data as noted above, as well as a combined dataset. The merging of the outcomes and intakes data was done on a unique key that is a combination of the given Animal ID and the intake number. Several of the animals in the dataset have been taken into the shelter multiple times, which creates duplicate Animal IDs that causes problems when merging the two datasets.
# 
# Copied from the description of the Shelter Outcomes dataset, here are some definitions of the outcome types:
# 
# * Adoption 
#   - the animal was adopted to a home
# * Barn Adoption 
#   - the animal was adopted to live in a barn
# * Offsite Missing 
#   - the animal went missing for unknown reasons at an offsite partner location
# * In-Foster Missing 
#   - the animal is missing after being placed in a foster home
# * In-Kennel Missing 
#   - the animal is missing after being transferred to a kennel facility
# * Possible Theft 
#   - Although not confirmed, the animal went missing as a result of theft from the facility
# * Barn Transfer
#   - The animal was transferred to a facility for adoption into a barn environment
# * SNR
#   - SNR refers to the city of Austin's [Shelter-Neuter-Release](http://www.austintexas.gov/blog/changes-made-shelter-neuter-return-cat-program-reflect-community-stakeholder-input) program. I believe the outcome is representative of the animal being released.
# 

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import warnings
import string
import time
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[13]:


plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['font.size'] = 15


# In[14]:


data = pd.read_csv('../input/aac_shelter_outcomes.csv')
data.head()


# ### Remove duplicates

# In[16]:


data.drop_duplicates(subset='animal_id', keep='first', inplace=True)


# ## Q. What is the most common age upon outcome?

# In[19]:


age_upon_outcome = data['age_upon_outcome'].value_counts().head(10)
plt.figure(figsize=(12,8))
_ = sns.barplot(age_upon_outcome.index, age_upon_outcome.values)
plt.xlabel("Age Upon Outcome")
plt.ylabel("Count")
for item in _.get_xticklabels():
    item.set_rotation(30)
plt.show()


# ## Q. Type of Animal?

# In[21]:


animal_type = data['animal_type'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(animal_type.values, explode=explode, labels=animal_type.index)
plt.axis('equal')
plt.tight_layout()
plt.show()


# **Dogs** and **Cats** are the most common type of  animals in the Austin Animal Center.

# ## Q. Breed of Animal?

# In[22]:


breed = data['breed'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(breed.values, explode=explode, labels=breed.index)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ## Q. Color of Animal?

# In[23]:


color = data['color'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(color.values, explode=explode, labels=color.index)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ## Q. Sex of Animal?

# In[25]:


sex_upon_intake = data['sex_upon_outcome'].value_counts().head(4)
explode = (0.05, 0.05, 0.05, 0.05)  # explode 1st slice
# Plot
plt.pie(sex_upon_intake.values, explode=explode, labels=sex_upon_intake.index)
plt.axis('equal')
plt.tight_layout()
plt.show()


# ## Q. Outcome Type of Animal?

# In[28]:


outcome_type = data['outcome_type'].value_counts().head(4)
plt.figure(figsize=(6,6))
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
#explsion
explode = (0.05,0.05,0.05,0.05)
plt.pie(outcome_type, colors = colors, labels=outcome_type.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')  
plt.tight_layout()
plt.show()


# **42.2%** animals were adopted after outcome.

# **To be continued...**
