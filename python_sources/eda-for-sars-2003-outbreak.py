#!/usr/bin/env python
# coding: utf-8

# As **Covid-19** AKA **Coronovirus** is spreading exponentially, this notebook throw some light on a similar outbreak
# 
# **SARS 2003 Outbreak**
# 
# *This is a complete dataset and will see couple of basic graphs to see the spread.*

# In[ ]:


import pandas as pd
import numpy as np


# We will begin by reading the dataset

# In[ ]:


data=pd.read_csv("/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")


# Let's look at the dimensions of the dataset

# In[ ]:


data.shape


# This dataset contains 2538 rows spreading over 5 columns which can be seen below:

# In[ ]:


data.head()


# Looking at how many countries has been affected by SARS 

# In[ ]:


len(data['Country'].unique())


# There are total 37 countries affected by SARS, which are

# In[ ]:


print(data['Country'].unique().tolist())


# In[ ]:


countryWise=data.groupby('Country').sum().reset_index()


# Let's look at what were the Cumulative number of cases, number of deaths and number of people recovered by each Country

# In[ ]:


countryWise


# Let's plot Country and number of cases

# As we can there is wide difference between indiviudal country, hence we will use **logarithmic version** to compare the **number of cases**

# In[ ]:


nosCases=countryWise.iloc[:,0:2].copy()


# In[ ]:


nosCases.head()


# In[ ]:


nosCases['log(C nos of cases)']=np.log(nosCases['Cumulative number of case(s)'])


# In[ ]:


nosCases.head()


# In[ ]:


nosCases.plot.bar(x='Country',y='log(C nos of cases)',rot=90)


# Similarly, we can look at comparative version for number of deaths and number of recovered.
# 
# Let's look at them side by side to get better idea.

# In[ ]:


deathRec=countryWise.iloc[:,[0,2,3]].copy()


# In[ ]:


deathRec.head()


# Since, number of deaths can be zero, we cannot perform log directly, so we will add constant 1 to the column and then take log of columns
# 
# We will do similar transformation with Number recovered.

# In[ ]:


deathRec['Deaths(log version)']=deathRec['Number of deaths']+1
deathRec['Deaths(log version)']=np.log(deathRec['Deaths(log version)'])


# In[ ]:


deathRec['Recovered(log version)']=deathRec['Number recovered']+1
deathRec['Recovered(log version)']=np.log(deathRec['Recovered(log version)'])


# In[ ]:


d=deathRec.iloc[:,3]
r=deathRec.iloc[:,4]


# In[ ]:


df=pd.DataFrame({'NosDeaths':d
                 ,'NosRecovered':r
                ,'Country':deathRec.iloc[:,0]})


# In[ ]:


df.set_index('Country').plot.bar(rot=90)


# Let's Look at top 10 countries according to number of Cases and see their death and recovery statistics

# In[ ]:


top10Cases=countryWise.sort_values('Cumulative number of case(s)',ascending=False).head(10).reset_index()
top10Cases.drop(columns=(['index']),inplace=True)
top10Cases


# Here, we can see Number of Deaths has 0 for Country such as United States and Germany, so we will add constant 1 to all the countries and do log transformation to see comparative grasth

# In[ ]:


top10Cases['Deaths(log)']=top10Cases['Number of deaths']+1
top10Cases['Deaths(log)']=np.log(top10Cases['Deaths(log)'])
top10Cases['Recovered(log)']=np.log(top10Cases['Number recovered'])


# In[ ]:


top10Cases


# In[ ]:


deathRec=top10Cases.iloc[:,[0,4,5]].set_index('Country')


# In[ ]:


deathRec.plot.bar(rot=90)


# **Thus we see that every country having top number of cases reacted in fairly similar manner except United States and Germany which recorded no Deaths.**
