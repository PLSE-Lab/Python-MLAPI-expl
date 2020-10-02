#!/usr/bin/env python
# coding: utf-8

# <img src="https://resources.healthydirections.com/resources/web/articles/hd/hd-women-and-heart-disease-sinatra-rollup-hd-cover.jpg" style="width:800px;height:600px;">

# # Import and Information about Data.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


heart=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
heart.head(10)
 # it gives 10 result 


# In[ ]:


heart.info()
#It gives information about data like non values 


# In[ ]:


# Show some statistics about dataset
heart.describe()


# In[ ]:


# We check all values
heart.isnull().any()


# In[ ]:


heart.columns 
# Shows that we have which columns


# In[ ]:


heart.dtypes
#It gives int,float,object etc.


# # GRAPHICS AND VISUALIZATION

# In[ ]:


heart.target.value_counts()
#We can count with this method 


# In[ ]:


sns.countplot(x="target", data=heart, palette="deep")
plt.show()
#This graph shows how many target there are in data.


# In[ ]:


sns.countplot(x='sex', data=heart, palette="pastel")
plt.xlabel("Sex (0 = female, 1= male)") # Representing values 
plt.show()
#It gives male and female relationship


# In[ ]:


heart.groupby('target').mean()
#It calculates mean of all main topic 


# In[ ]:


threshold = sum(heart.age)/len(heart.age)
print(threshold)
heart["age_situation"] = ["old" if i > threshold else "young" for i in heart.age]
heart.loc[:10,["age_situation","age"]]
#Calculator of Age Scale and gives information which one old or young for this illness.


# In[ ]:


sns.barplot(x=heart.age.value_counts()[:10].index,y=heart.age.value_counts()[:10].values)
plt.xlabel('Age')
plt.ylabel('Number of People')
plt.title('Age Calculation')
plt.show()
#Bar Plot and this is providing distribution of age


# *  Let's we calculate min-max-mean age

# In[ ]:


#We can find min, mean and max ages 
minAge=min(heart.age)
maxAge=max(heart.age)
meanAge=heart.age.mean()
print('Min Age :',minAge)
print('Max Age :',maxAge)
print('Mean Age :',meanAge)


# In[ ]:


# Bar chart for age with sorted index
plot = heart[heart.target == 1].age.value_counts().sort_index().plot(kind = "bar", figsize=(15,5), fontsize = 15)
plot.set_title("Age Distribution", fontsize = 15)
plt.ioff()


# In[ ]:


sns.boxplot(x="age", data=heart);
# Some Box plot for Age


# In[ ]:


plt.figure(figsize=(14,6))
sns.violinplot(x='age', y='target', data=heart)
plt.xlabel('age', fontsize=10)
plt.ylabel('target', fontsize=10)
plt.show()
#I want to show one visualization of Violin Plot.


# ## That's all.Thanks for examining and if you like my kernel , will you upvote to me.Have a good day :)
