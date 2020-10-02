#!/usr/bin/env python
# coding: utf-8

# Hey there... I have done some data analysis on the dataset given and even tried to make a recommendation system. I know there might be faults, flaws. Please feel free to ask and point, it will help me improve as I want to be in this field. 
# **So Lets get started**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="deep")


# In[ ]:


dataSeries=pd.read_csv(os.path.join(dirname, filename))


# In[ ]:


dataSeries


# In[ ]:


#Making deep copies
dataS1=dataSeries.copy(deep=True)


# In[ ]:


#Naming the 1st column as Sno and making it as index column
dataS1=dataS1.rename(columns={"Unnamed: 0":"Sno"})


# In[ ]:


dataS1=dataS1.set_index("Sno")


# In[ ]:


dataS1


# # General Informations

# In[ ]:


dataS1.info()


# In[ ]:


dataS1.shape


# In[ ]:


dataS1.isnull().sum()


# In[ ]:


dataS1.Year.unique()


# In[ ]:


dataS1.Age.unique()


# # General Exploration

# ### Let's see how many shows were released in each years

# In[ ]:


plt.figure(figsize=(25,7))
sns.countplot(dataS1['Year'])
plt.xticks(rotation=45)
plt.show()


# We see that during the year 2017, most of the tv series were released

# #### There are many punctuation and extra characters in the name of the series, so now we will be doing some cleaning

# In[ ]:


import string


# In[ ]:


# creating a function that will remove all the punctuations
def remove_punctuations(txt):
    text_nopunct="".join([i for i in txt if i not in string.punctuation])
    return text_nopunct


# In[ ]:


# we will create a new column with shows name with no punctuations
dataS1['Title_nopunt']=dataS1['Title'].apply(lambda x: remove_punctuations(x))
dataS1['Title_nopunt']=dataS1['Title_nopunt'].str.lower()


# In[ ]:


dataS1.tail()


# We can see that many shows don't have Age, IMDb, Rotten Tomatoes values, but we can't drop them since droping them will cause many shows to be removed from the dataset
# 
# For replacing them, as per my knowledge, these columns can't be filled by any sort of values because it may result in wrong interpretation.
# 
# Example, if we replace NaN values of Age column with let's say 7+, but actually many shows won't be 7+, this will lead to mist interpretation

# ### Lets see all the shows rating of IMDb of all the shows present in Netflix

# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='Age', y='IMDb', hue='Netflix', data=dataS1, palette='Reds')
plt.xticks(fontweight='bold')


# ### lets do the same for prime video, hulu and disney+

# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='Age', y='IMDb', hue='Prime Video', data=dataS1, palette='Blues')
plt.xticks(fontweight='bold')


# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='Age', y='IMDb', hue='Hulu', data=dataS1, palette='Greens')
plt.xticks(fontweight='bold')


# In[ ]:


plt.figure(figsize=(20,6))
sns.barplot(x='Age', y='IMDb', hue='Disney+', data=dataS1, palette='ocean')
plt.xticks(fontweight='bold')


# I tried to color the bars with those colors which are used by their actual names, like red in netflix, green in hulu and all...

# In[ ]:


plt.figure(figsize=(20,7))
plt.hist(dataS1['IMDb'],edgecolor='#DC143C', label="IMDb Ratings")
plt.legend()
plt.show()


# We can interpret that most of the shows have ratings b/w 6.5 to 8.5

# ### As we can see in Rotten Tomatoes rating are in %, which we are going to change in decimal form same as IMDb

# In[ ]:


dataS1['Rotten Tomatoes']=dataS1['Rotten Tomatoes'].str.replace("%","")


# In[ ]:


dataS1['Rotten Tomatoes']=dataS1['Rotten Tomatoes'].astype(float)


# In[ ]:


dataS1['Rotten Tomatoes']=dataS1['Rotten Tomatoes']/10


# In[ ]:


dataS1


# In[ ]:


plt.figure(figsize=(20,7))
plt.hist(dataS1['Rotten Tomatoes'],edgecolor='#DC143C', label="Rotten Tomatoes Ratings")
plt.legend()
plt.show()


# We can see that, many of the shows got good rating. and aroud 350 or so had ratings b/w 9-10

# # Recommendation System

# ### Let's try making a recommendation system based on users input:
# 
# #### User will input the Name of a series/a group of series
# 
# #### we will get its information like Age, IMDb rating, and its availablity on the platform
# 
# #### with this gather data, we will try to recommend the user smiliar shows he/she can watch

# In[ ]:


def recommend_more(df,namesoftheshows):
    #print(namesoftheshows)
    datasub=df.loc[df['Title_nopunt'].isin(namesoftheshows)] #the one with the namesoftheshows
    #print(datasub)
    datanew=df.loc[~df['Title_nopunt'].isin(namesoftheshows)] # the one without the namesoftheshows, and from where the recommendation will come
    datasub=datasub.drop(['Title'],axis=1)
    datanew=datanew.drop(['Title'],axis=1)
    # now we will make a new dataframe, with Age as base we got from previous df
    listage=list(datasub['Age'])
    #print(listage)
    datanew=datanew.loc[datanew['Age'].isin(listage)] #This one contains only those shows who's age matches with the age of namesoftheshows
    listIMDb=np.array(datasub['IMDb']) #this for multiplication purpose
    
    
    """making dummies"""
    datadummysub=pd.get_dummies(datasub['Age'])
    #print(datadummysub)
    datasub=pd.concat([datasub,datadummysub], axis=1)
    datadummynew=pd.get_dummies(datanew['Age'])
    datanew=pd.concat([datanew, datadummynew], axis=1)
    #datadummysubnetflix=pd.get_dummies(datasub['Netflix'])
    #print(datadummysubnetflix)
    
    """From this point on we are trying to make a normalized user weighted matrix given from his choice of shows"""
    #making weighted matrix for datasub which will be multiplied by listIMDb
    datasub1=datasub.drop(['Year','Age','IMDb','Rotten Tomatoes','type','Title_nopunt'], axis=1)
    #print(datasub1)
    listIMDb=listIMDb.reshape(len(listage),1) #reshaping the matrix so that it could be multiplied
    #print(listIMDb)
    datanum=np.array(datasub1) #changing our datasub into numpy array so that we can multiply listIMDb
    #print(datanum)
    weighted_array=np.multiply(listIMDb,datanum) #making our weighted array
    #print(weighted_array)
    #now making a user weighted matrix
    user_weighted_matrix=np.sum(weighted_array, axis=0) #using np.sum() so as to get column wise sum
    #print(user_weighted_matrix)
    #now making a normalized user weighted matrix
    norm_user_weighted_matrix=user_weighted_matrix/sum(user_weighted_matrix) 
    #print(norm_user_weighted_matrix)
    
    """The previous step is done"""
    
    """Now by using the norm_user_weighted_matrix, we will try to recommend the user a list to shows"""
    datanew1=datanew.drop(['Year','Age','IMDb','Rotten Tomatoes','type','Title_nopunt'], axis=1)
    #print(datanew1)
    datanum1=np.array(datanew1) #this is our candidate matrix
    #print(datanum1)
    weighted_candidate_matrix=np.multiply(norm_user_weighted_matrix,datanum1) #now making weighted candidate matrix
    #print(weighted_candidate_matrix)
    recommendation_candidate_matrix=np.sum(weighted_candidate_matrix, axis=1)
    #print(aggregated_weighted_candidate_matrix)
    #recommendation_candidate_matrix=recommendation_candidate_matrix.reshape(-1,1) **** no need to reshape, that's why commented out
    #print(recommendation_candidate_matrix)
    
    """Now since we got the recommendation matrix, now we will merge the matrix as a column in the datanew matrix"""
    datanew['recommendation_rating']=pd.Series(recommendation_candidate_matrix)
    datanew=datanew.sort_values('recommendation_rating',ascending=False)
    #print(datanew)
    print(datanew[['Title_nopunt','recommendation_rating']].head(10))


# In[ ]:


recommend_more(dataS1,['breaking bad','stranger things', 'the flash', 'one punch man'])


# In[ ]:


#Trying another example
recommend_more(dataS1,['stranger things', 'the flash', 'one punch man'])


# In[ ]:




