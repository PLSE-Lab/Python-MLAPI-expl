#!/usr/bin/env python
# coding: utf-8

# **In this solution, I have tried to identify the pattern on who is more likely to be part of the olympic squad for the contries based on their Age, Height and Weight Analysis?**
# As a starter, I am sure there could be some short comings in the analysis. Please give in your inputs so that I could use them for my improvement in Data science and python. Thank you all.

# In[ ]:


#Importing the necessary packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing the data
athlete = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')


# In[ ]:


#Having a look at the athletes data
athlete.head()


# In[ ]:


#Having a look at the regions data
regions.head()


# In[ ]:


#In the below section, we are trying to find the variation in the weight of the athletes taking part in olympics irrespective of the sport


# In[ ]:


athlete["Weight"].plot(kind = "hist", bins = 200,figsize = (12,6), xlim = (0,150))
athlete_weight = athlete[(athlete["Weight"]>=60) &  (athlete["Weight"]<=76)]
athlete_weight["Weight"].plot(kind = "hist", bins = 16, color="Red", title = "Height >= 60 and <= 76")
plt.xlabel("Weight", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution by Weight",fontsize=16)
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()


# **As you can see from the above analysis, that the Weight of more than 50% of the athletes in the Olympics irrespective of gender has been in the range of 60 to 76 kgs**

# In[ ]:


#In the below section, we are trying to find the variation in the Height of the athletes taking part in olympics irrespective of the sport


# In[ ]:


f, axes = plt.subplots(1, 2, sharex = True, figsize=(12,6))
athlete["Height"].plot(kind = "hist", bins = 100, ax = axes[0], title = "Distribution by Height")
athlete_height = athlete[(athlete["Height"]>=160) &  (athlete["Height"]<=190)]
athlete_height["Height"].plot(kind = "hist", bins = 30, color="Red", ax = axes[1], title = "Height >= 160 and <= 190")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# **Similarly from the above plot it can be seen that the height of major% of athletes has been in the range of 160cm to 190cm in Olympics**

# In[ ]:


#Analysis by Age
athlete["Age"].plot(kind = "hist")
athlete_age = athlete[(athlete["Age"]>=20) &  (athlete["Age"]<=26)]
athlete_age["Age"].plot(kind = "hist", color="Red")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution by Age")
plt.show()


# **The Above analysis by Age shows that the no. of athletes participating in the Olympics has always been very high in the Age range of 20 and 26 years**

# In[ ]:


#The Relation between the athlete's height and weight is analyzed using the below joint plot


# In[ ]:


x = sns.jointplot(data = athlete, x = 'Height', y = 'Weight', kind = 'scatter', ylim = (20, 160), size = 8)


# **The Analysis of Height and Weight of the athletes using the above plot also gives a clear pattern that a large number of athletes are in the height range of 160m to 190cm and in the weight range of 60 and 76 kgs**

# In[ ]:


#Using Facet grid to identify the relation between the athlete's height and weight based on the seasons


# In[ ]:


g = sns.FacetGrid(athlete, row = 'Year', col ='Sex', hue="Season")
g = g.map(plt.scatter, 'Weight', 'Height')


# **From the Facet Grid Analysis it can be seen that the relation between Height and Weight in the winter season is much tighter than that in the summer season for both male and female athletes**
# 
# **This gives us a new dimension that the height and weight of the athletes during the summer and winter season of the Olympic games has seen a different relation between them. Let's analyse this using the below subplots**

# In[ ]:


#Using joint plots to identify the difference
x = sns.jointplot(data = athlete[athlete['Season']== "Summer"], x = 'Height', y = 'Weight', kind = 'reg', ylim = (20, 160), color="Yellow", size = 10)
y = sns.jointplot(data = athlete[athlete['Season']== "Winter"], x = 'Height', y = 'Weight', kind = 'reg', ylim = (20, 160), color="Blue", size = 10)


# **From the above 2 joint plots it can be clearly seen that the range of Height and Weight is more wider in Summer Olympics as compared to the Height and Weight in the Winter Olympics and that the nations are looking for shorter and lighter athletes for Winter Olympics**
# 
# **The Range of Height and Weight in Summer Olympics
# Height : 130 cm to 220 cm
# Weight : 35kg to 160+ kgs**
# 
# **The Range of Height and Weight in Winter Olympics
# Height : 140 cm to 200 cm
# Weight : 40kg to 120 kgs**

# **Conclusion:**
# 
# **1. As an athlete you are more likely to represent your nation irrespective of gender if your weight is in the range of 60 to 76 kgs, Height in the range of 160cm to 190 cm and in the Age range of 20 and 26 years**
# 
# **2. Specific to seasons, Taller and heavier athletes get more chances of representing the nation in Summer Olympics (This might be because of events like Weight Lifting, Wrestling, boxing etc.) while comparitively shorter and lighter athletes are preferred for Winter Olympics**
# 
# **3. The Distribution of Height and Weight in Summer Olympics represent a distribution very close to Normal distribution while in case of Winter Olympics the distribution is more wider that a normal distribution**

# ------

# In[ ]:




