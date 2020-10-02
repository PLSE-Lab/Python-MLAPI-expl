#!/usr/bin/env python
# coding: utf-8

# ## What are the spending characteristics that correlate the most with high test scores? ##

#  
# * What are the lowest educational scoring states? Lets use cluster analysis to see if we can categorize the bottom of the group
# * How does spending compare to states that scoring very well? 
#   
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[ ]:


data = pd.read_csv('../input/states_all.csv')


# ## what does our data look like? Lets do some basic exploring 

# In[ ]:


data.shape


# In[ ]:


print (data.columns) 
data.head()


# Lets see a state example. 

# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


print (data['YEAR'].max())
print (data['YEAR'].min())


# ### Now that we understand what our data looks like, lets clean our data to perform some analysis 

# Lets drop NA values from the enroll column 

# In[ ]:


data = data.dropna(subset=['ENROLL'])


# In[ ]:


data.shape


# In[ ]:


1492-1229


# Remove 263 lines of data where the enrollment number was NaN

# In[ ]:


del data['PRIMARY_KEY']


# In[ ]:


data.set_index('STATE')
data.head()


# We have reindexed the data so that the data is the State

# Here we are going to try to begin the clustering process

# In[ ]:


#picking columns that are relevant to the scoring 
data.isnull().any()


# In[ ]:


scores = ['AVG_MATH_4_SCORE', 'AVG_MATH_8_SCORE','AVG_READING_4_SCORE','AVG_READING_8_SCORE']


# In[ ]:


scores_df = data[scores].dropna().copy()
print (scores_df.isna().sum())


# In[ ]:


scores_df.shape


# In[ ]:


scores_df.isna().sum()


# In[ ]:


scores_df.index


# In[ ]:


X = StandardScaler().fit_transform(scores_df)
X


# In[ ]:


kmeans = KMeans(n_clusters=4)
model = kmeans.fit(X)
print("model\n", model)


# In[ ]:


centers = model.cluster_centers_
centers


# In[ ]:


def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')

    # Zip with a column called 'prediction' (index)
    Z = [np.append(A, index) for index, A in enumerate(centers)]

    # Convert to pandas data frame for plotting
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P


# In[ ]:


from pandas.plotting import parallel_coordinates


# In[ ]:


from itertools import cycle, islice


# In[ ]:


def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')


# In[ ]:


P = pd_centers(scores, centers)
P


# In[ ]:


parallel_plot(P[P['AVG_MATH_8_SCORE'] < 1])


# In[ ]:


parallel_plot(P[P['AVG_READING_8_SCORE'] < 1])


# ## Clustering results 
# We can make a conclusion that the states that have low scores have similarly low scores across all the scoring criteria, except 8th grade reading score which does not follow the same cluster. 
# 
# We can make similar observation that the states that score well, score well in all the categories and merit further investigation into there spending techniques. 

# ## Now are we to compare a low scoring state to a high scoring school state
# * In one year, how does a high scoring state spend the revenue compared to low scoring state? 

# Lets find two states to compare based on what we know on the clustering results. 

# In[ ]:


data.head()


# In[ ]:


data[data['YEAR'].isin([2015])].sort_values(by = 'AVG_MATH_4_SCORE')


# Based on this we will compare Minnesota to Alabama. Two states in different clusters based on what we know about how 
# the reading scores for fourth graders can be a good gage on overall scoring. 

# In[ ]:


print ('Minnesota enrollment for 2015 = 807044')
print ('Alabama enrollment for 2015 = 734974')


# In[ ]:


revenue_data = data[['STATE','YEAR','TOTAL_REVENUE','FEDERAL_REVENUE','STATE_REVENUE','LOCAL_REVENUE']] 
revenue_data.tail()


# In[ ]:


year = 2016 
state1 = 'MINNESOTA'


filter1 = revenue_data['STATE'].str.contains(state1)  
filter2 = revenue_data['YEAR'].isin([year])

rev = revenue_data[filter1 & filter2] 
type(rev)


# In[ ]:


year = 2016 
state1 = 'ALABAMA'


filter1 = revenue_data['STATE'].str.contains(state1)  
filter2 = revenue_data['YEAR'].isin([year])

rev1 = revenue_data[filter1 & filter2] 
type(rev1)


# In[ ]:


df = pd.concat([rev,rev1])
##uSE THE PANDAS.PLOT.BAR() 11PM AT NIGHT GOING TO SLEEP 
df


# In[ ]:


del df['YEAR']


# In[ ]:


df


# In[ ]:


df_melt = pd.melt(df,id_vars=['STATE'] , var_name='revenue')
df_melt
                  


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = 'revenue', y= 'value', hue='STATE', data=df_melt)


# In[ ]:


expenditure_data = data[['STATE','YEAR','TOTAL_EXPENDITURE', 'INSTRUCTION_EXPENDITURE',
       'SUPPORT_SERVICES_EXPENDITURE', 'OTHER_EXPENDITURE',
       'CAPITAL_OUTLAY_EXPENDITURE']]
expenditure_data.head()


# In[ ]:


year = 2016 
state1 = 'MINNESOTA'


filter3 = expenditure_data['STATE'].str.contains(state1)  
filter4 = expenditure_data['YEAR'].isin([year])

exp = expenditure_data[filter3 & filter4] 
exp


# In[ ]:


year = 2016 
state1 = 'ALABAMA'


filter3 = expenditure_data['STATE'].str.contains(state1)  
filter4 = expenditure_data['YEAR'].isin([year])

exp1 = expenditure_data[filter3 & filter4] 
exp1


# In[ ]:


df1 = pd.concat([exp,exp1])
df1


# In[ ]:


df1 = df1.drop('YEAR', 1)


# In[ ]:


df1


# In[ ]:


df_melt = pd.melt(df1,id_vars=['STATE'] , var_name='expenditure')
df_melt
  


# In[ ]:


df_melt


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x = 'expenditure', y= 'value', hue='STATE', data=df_melt)

