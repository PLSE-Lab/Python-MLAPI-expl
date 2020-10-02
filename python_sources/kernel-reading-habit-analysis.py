#!/usr/bin/env python
# coding: utf-8

# ** Reading Habit Analysis**
# 
# > **What Motivates you to Read ? **

# ![](https://images.unsplash.com/photo-1509114859430-5f2c74177f4b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=750&q=80)

# **Hi, everyone. Here I am presenting Reading Habit Analysis. Maybe there exist any other analysis like this but I have not found it as yet. I hope you will enjoy going through the below mentioned experiment :p 
# Also I have written an article corresponding to this, you can find it here:https://medium.com/@biach1312/reading-habit-analysis-895f78865bb0
# **
# 
# **Enjoy!**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading the file
df = pd.read_csv('/kaggle/input/reading-habit-analysis/Reading Habit Analysis Article.csv', engine='python')


# In[ ]:


# Checking out top 10 items from the dataset
df.head(10)


# In[ ]:


# Counting the number of males and females from the Gender column of dataset
df['Gender'].value_counts()


# In[ ]:


# Ploting pie and bar charts for Gender column 
import matplotlib.pyplot as plt
temp1 = df['Gender'].value_counts()
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
#ax1.set_xlabel('Gender')
#ax1.set_ylabel('Gender Count')
ax1.set_title("Number of Participants based on Gender")
temp1.plot(kind='pie')
fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Gender Count')
temp1.plot(kind='bar')


# In[ ]:


# Getting the idea of how people answered the Question: "What motivates you to Read articles, books, technology blogs, news etc ?" present in the dataset
df['What motivates you to Read articles, books, technology blogs, news etc ?']


# In[ ]:


#to split the answers (just the last column of dataset i.e. 'What motivates you to Read articles, books, technology blogs, news etc ?' )
from collections import Counter
split = [] #empty array to store the result of splitted data
for i in range(48):
    s = df['What motivates you to Read articles, books, technology blogs, news etc ?'][i].lower().split()
    split+= s #adding splitted values in the empty list
print(split)


# In[ ]:


words = pd.DataFrame(split) #converting split into dataframe


# > **Data Cleansing**

# ![](https://media.giphy.com/media/WoWm8YzFQJg5i/giphy.gif)

# In[ ]:


from collections import Counter
import numpy as np
from nltk.corpus import stopwords 

stop = stopwords.words('english')

useful_words = words[0].apply(lambda x: ''.join([word for word in x.split() if word not in (stop)]))  # removing English stop words from the dataset
#print(useful_words)

spacefree = pd.DataFrame(useful_words) #converting useful_words list into dataframe
spacefree[0].replace('', np.nan, inplace=True) #replacing empty strings with NAN in the dataframe
spacefree[0].replace('.', np.nan, inplace=True) #replacing . with NaN in the dataframe
spacefree[0].replace(',', np.nan, inplace=True) #replacing , with NAN in the dataframe
spacefree[0].replace('-', np.nan, inplace=True) #replacing - with NAN in the dataframe
Counter(spacefree[0].dropna(axis=0, how='any')) #droping all NAN from the dataframe
df_vals = spacefree[~spacefree[0].isnull()] #If previous dropna failed then this will drop all the null values from the dataframe, I don't know but this was working like this
#df_vals 


# > **Clustering The Common Words**

# In[ ]:


#NOW DATA IS CLEANED
Count = Counter(df_vals[0]) #to get the word count of cleaned data
C = Count.most_common(20) #to get 20 most common words with their count
C = pd.DataFrame(C)       #converting the 20 most common words with their word count into dataframe


# In[ ]:


#Plotting 20 most common words Dataframe with the help of a bar chart
C.plot.bar(x=0, y=1, rot=100)


# In[ ]:


#Other words clusters are as under with their counts
#Count = Counter(df_vals[0])
#Count
#C = pd.DataFrame(C) 
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation #For clustering

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(111)
y = np.random.rand(591)
colors = ("blue")
scatter = ax.scatter(df_vals[0],y,c=colors, marker="*")
ax.set_title('Words Clusters')
ax.set_xlabel('words')
ax.set_ylabel('Count')
plt.scatter


# In[ ]:


Count = Counter(df_vals[0])
#Count.most_common(591) 


# In[ ]:


Count.most_common(197) # 591/3 = 197
df_Count = pd.DataFrame(Count.most_common(591))

#df_Count 


# In[ ]:


list1 = [] #list to create target class

for i in range(197):
    list1 += [0] #0 stands for common words
for i in range(394):
    list1.append(1) #1 stands for un common words    

#len(list1)
#list1


# In[ ]:



data = list(zip(df_Count[0], list1))
data
  
# Create the pandas DataFrame 
df_class = pd.DataFrame(data, columns=['Words','Classes']) 
#df_class 


# > **Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[ ]:


X = df_class['Words']# Features
y = df_class['Classes'] # Target variable

X = pd.get_dummies(X,drop_first=True) # for handling categorical data 
#y = pd.get_dummies(y,drop_first=True) # not required for this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) # 90% training and 10% test



tree = DecisionTreeClassifier()
tree = tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
#y_pred
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


# > ***Happy Machine Learning :) ***

# > **------------------------------------------------------------------------------------------------------------------------------------------------**
