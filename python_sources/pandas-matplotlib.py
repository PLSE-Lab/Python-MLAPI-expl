#!/usr/bin/env python
# coding: utf-8

# #Exploration of the dataset
# Before you start to solve the classification task you should carry out some research to get familiar with the dataset and to decide what features and observations will be useful. Let's do it together.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import DataFrame,merge
style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# First, let's count how many cats and dogs we have in the shelter.

# In[ ]:


animals = pd.read_csv("../input/train.csv")


# In[ ]:


AnimalType = animals['AnimalType'].value_counts() 
AnimalType.plot(kind='bar',color='#34ABD8',rot=0)


# Now, let's see how different outcomes are distributed.

# In[ ]:


AnimalType = animals.OutcomeType.value_counts().sort_values() 
AnimalType.plot(kind='barh',color='#34ABD8',rot=0)


# And now,we will see the OutcomType between dogs and cats

# In[ ]:


AnimalType = animals[['AnimalType','OutcomeType']].groupby(['OutcomeType','AnimalType']).size().unstack()
AnimalType.plot(kind='bar',color=['#34ABD8','#E98F85'],rot=-30)


# We can see that adoption and transfer are our leaders (good for poor animals).

# We have another column - sex upon outcome. 

# In[ ]:


SexuponOutcome = animals['SexuponOutcome'].value_counts()
SexuponOutcome.plot(kind='bar',color=['#34ABD8'],rot=-30)


# We can find out all types of sex in SexuponOutcome,and we will calssify them:

# In[ ]:


sexType = animals['SexuponOutcome'].unique()
print(sexType)


# Now we create the dict and add the data to DataFrame-animals:

# In[ ]:


M_F = {'Neutered Male':'Male','Spayed Female':'Female','Intact Male':'Male','Intact Female':'Female','Unknown':'Unknown'}
N_T = {'Neutered Male':'Neutered','Spayed Female':'Neutered','Intact Male':'Intact','Intact Female':'Intact','Unknown':'Unknown'}

animals['Sex'] = animals.SexuponOutcome.map(M_F)
animals['Neutered'] = animals.SexuponOutcome.map(N_T)


# In[ ]:


Sex = DataFrame(animals.Sex.value_counts())
Neutered = DataFrame(animals.Neutered.value_counts())
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.bar([1,2,3],Sex['Sex'],align='center')
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(Sex.index)
ax2.bar([1,2,3],Neutered['Neutered'],align='center')
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(Neutered.index)


# Actually, it contains two types of information: if animal was male or female and if it was neutered/spayed or intact. I hope it is a good idea to divided this column into two

# Well, it seems like we have approximately equal number of male and female animals, and neutered (or spayed) prevail amongst them.

# Well, the poor animals from the shelter can not boast of breed purity in most cases.

# In[ ]:


df = DataFrame(animals[['Sex','OutcomeType']])
#df.plot(kind='bar')
OutcomeSex = df.groupby(['Sex','OutcomeType']).size().unstack()
OutcomeSex.plot(kind='bar',color=['#34ABD8','#E98F85','r'],rot=-30)


# In[ ]:


df = DataFrame(animals[['Sex','OutcomeType']])
SexOutcome = df.groupby(['OutcomeType','Sex']).size().unstack()
SexOutcome.plot(kind='bar',rot=-30)


# Now we'll see if the neutered could influence the OutcomeType

# In[ ]:


OT_N = animals[['OutcomeType','Neutered']].groupby(['Neutered','OutcomeType']).size().unstack()
OT_N.plot(kind='bar',rot=-30)


# As you see:The Neutered pets are accepted mostly.The points should be on the Neutereds!

# In[ ]:


DC = animals[['OutcomeType','Neutered','AnimalType']].groupby(['AnimalType','OutcomeType','Neutered']).size().unstack().unstack()
DC.plot(kind='bar',stacked=False,figsize=(10,8),rot=-30)


# Interesting that young cats and dogs have much higher chances to be adopted or transferred than to be returned to owner or something else, while older animals with approximately equal probability can be adopted, transferred or returned.
