#!/usr/bin/env python
# coding: utf-8

# This is a kernel based on Zomato Ratings. I have tried to do a uni-variate analysis, bi-variate analysis to understand which factors influence customers to give a certain rate to the restaurants.Is it location based? or are customers price sensitive?
# 
# Let us start answering questions as:-
#  Which are the famous restaurants people like to dine in?
#    - What is the rating they usually give?
#    - What is the cost they are willing to pay?
#    - What is the characteristics/nature of food that customers prefer to eat here?
#    - Is Cost a factor when it comes to give rating to customers?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df = pd.read_csv('../input/zomato.csv')


# > 1. ![](http://)Here we get a basic idea on the data set wrt its columns and data present

# ### Exploratory Data Analysis

# In[ ]:


df.head()


# **By the data above, we understand rate is a dependant variable . Rate is given by customers based on factors like online_order options, booking a table option, dishes prepared, type of cuisine etc.**

# 

# 2. We assess the number of columns and rows present. This data is useful incase we want to delete certain rows which contain null values

# In[ ]:


df.shape


# > 3. Checking for NAs in the Data

# In[ ]:


df.isna().sum()


# 

# In[ ]:


#df = df.dropna(axis=0, subset=['rate'])
#df.isna()


# 4. Checking the rows whose ratings have NA as a value

# In[ ]:


df[df['rate'].isna()]


# **By this we conclude, we can delete rows who have ratings as "NA"**

# In[ ]:


#df['book_table'][df['rate'].isna()].value_counts()


# 5. Deleting rows which have "NA" 

# In[ ]:


df = df.dropna(axis=0, subset=['rate'])
df.dtypes


# 6. If you would have observed, the ratings for all restaurants are made out of 5. Hence we can remove the '/5' from the rating

# In[ ]:


df.rate.astype(str,inplace=True)
df['rate'] = [x[:-2] for x in df['rate']]


# In[ ]:


#df['rate'].fillna("0/5",inplace=True)
df['dish_liked'].fillna("None",inplace=True)
df['cuisines'].fillna("None",inplace=True)
df['approx_cost(for two people)'].fillna("0.0",inplace=True)
df['rest_type'].fillna("None",inplace=True)
df['location'].fillna("None",inplace=True)


# In[ ]:


#df.isna().sum()


# ### Conversion to Numeric and Renaming

# 7. The observation from the data set is that Rate, Votes and Average Cost per 2 person should be numeric . They are right now represented as strings.

# In[ ]:


df['rate'] = pd.to_numeric(df['rate'],errors='coerce')
df['votes'] = pd.to_numeric(df['votes'],errors='coerce')
df.rename(columns={'approx_cost(for two people)': 'avg_cost'},inplace = True)
df.rename(columns={'listed_in(type)': 'type'},inplace = True)
df.rename(columns={'listed_in(city)': 'city'},inplace = True)
df['avg_cost'] = pd.to_numeric(df['avg_cost'],errors='coerce')
df.plot(x='rate',y='votes',kind='scatter',title='Relation between Rating and Vote')
df.plot(x='avg_cost',y='votes',kind='scatter',title='Relation between Cost and Vote')


# 

# In[ ]:


df.isna().sum()


# In[ ]:


df.drop(['url','address','reviews_list'],axis=1,inplace=True)


# 

# In[ ]:


df.head()


# Now that we are done dealing with all the NAs. Let us focus more on bringing more sense from the data already present. If you see , every record in the table has a rating out of 5. Let us try removing the '/5' 

# In[ ]:


#df.drop(['production_companies','cast'],axis=1,inplace=True)
df.drop_duplicates(keep='first',inplace=True)


# In[ ]:


df['rate'].dtype


# In[ ]:


df.type.value_counts()


# ## Univariate Analysis

# In[ ]:


df.hist(figsize=(8,8))


# In[ ]:


plt.subplots(1,2,figsize=(8,4))
plt.subplot(1,2,1)
sns.countplot(df['online_order'])
plt.subplot(1,2,2)
sns.countplot(df['book_table'])
plt.tight_layout()


# Q. What is the most favourite eating destinations in Bangalore?

# In[ ]:


#plt.figure(figsize=(20,5))
#sns.countplot(df['location'],palette='ocean_r',order=df['location'].value_counts().index)
#plt.xticks(rotation=90)


# In[ ]:


"""
plt.figure(figsize=(15,5))
sns.distplot(df['avg_cost'])
plt.xticks(rotation=90)
sns.distplot(df['avg_cost'])
"""


# In[ ]:


#df.rate.dtype


# What is the top 20 famous cuisines in Bangalore?

# In[ ]:


plt.figure(figsize=(15,5))
rest_type=df['rest_type'].value_counts()[:20]
sns.barplot(rest_type.index,rest_type,palette='Pastel1')
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")
plt.xticks(rotation=90)


# **Quick Bites and Casual Dining is mostly what customers usually prefer**

# What are the famous eating places in Bangalore

# In[ ]:


plt.figure(figsize=(15,5))
sns.countplot(df['city'],palette='Purples',order = df['city'].value_counts().index)
plt.xticks(rotation=90)


# Q.What are the preferred ways of eating in Bangalore?

# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot(df['type'],palette='Accent',order = df['type'].value_counts().index)
plt.xticks(rotation=90)


# **Customers usually prefer having food delivered rather than eating out**

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(df['rate'],palette='Oranges')
plt.xticks(rotation=90)


# Which are the most Popular cuisines of Bangalore?

# In[ ]:


plt.figure(figsize=(7,4))
cuisines=df['cuisines'].value_counts()[:10]
sns.barplot(cuisines.index,cuisines)
plt.xlabel('Count')
plt.title("Most popular cuisines of Bangalore")
plt.xticks(rotation=90)


# **North Indian Cuisine is considered to be famous in Bnagalore**

# In[ ]:





# ## Bivariate Analysis

# __Average Rating given to Restaurants for Providing Booking Option__
# 
# 
# Here we check what are the average ratings given to restaurants who provide the option of booking a table at their restaurant. It Looks like customers prefer restaurants who give the provision of booking a table

# In[ ]:


plt.figure(figsize=(3,3))
df.groupby('book_table')['rate'].mean().plot.bar()
plt.ylabel('Average rating')


# **Average Rating given to Restaurants for Online Order Option**
# 
# Here we check what are the average ratings given to restaurants who provide the option of ordering food online. 
# 
# The difference is not much

# In[ ]:


plt.figure(figsize=(6,3))
df.groupby('online_order')['rate'].mean().plot.bar()
plt.ylabel('Average rating')


# In[ ]:


#df.groupby('name')['rate'].mean()


# 

# In[ ]:


#df.query('online_order=="Yes"').query('book_table=="Yes"')


# 
#    

# What are the top 15 preferred eating places in Bangalore?

# In[ ]:


#plt.figure(figsize=(7,7))
#Rest_locations=df['location'].value_counts()[:15]
#sns.barplot(Rest_locations.index,Rest_locations,palette='Blues')
#plt.xticks(rotation=90)


# In[ ]:


rest_params = df.groupby(by='type', as_index=False).mean().sort_values(by='rate',ascending=False)
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x='type', y='avg_cost', data=rest_params, ax=ax,palette='Greens')
ax2 = ax.twinx()
sns.lineplot(x='type', y='rate', data=rest_params, ax=ax2, sort=False)
ax.tick_params(axis='x', labelrotation=90)
ax.xaxis.set_label_text("")

xs = np.arange(0,10,1)
ys = rest_params['rate']
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 color='black')
    
ax.set_title('Average Cost and Rating of Restaurants by Type', size=14)
plt.tight_layout()


# In[ ]:


"""
df['dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])
df['dish_liked'].value_counts()
#rest=df['rest_type'].value_counts()[:9].index
"""


# In[ ]:


def produce_wordcloud(rest):
    
    plt.figure(figsize=(20,30))
    for i,r in enumerate(rest):
        plt.subplot(3,3,i+1)
        corpus=df[df['rest_type']==r]['dish_liked'].values.tolist()
        corpus=','.join(x  for list_words in corpus for x in list_words)
        wordcloud = WordCloud(max_font_size=None, background_color='white', collocations=False,
                      width=1500, height=1500).generate(corpus)
        plt.imshow(wordcloud)
        plt.title(r)
        plt.axis("off")
        


# ## Multi-Variable Analysis ##

# In[ ]:


numeric_vars = ['avg_cost','rate','votes']
plt.figure(figsize = [8, 5])
sns.heatmap(df[numeric_vars].corr(), annot = True, fmt = '.3f',cmap = 'vlag_r', center = 0)
plt.show()


# ### Developing Wordcloud to see which are the favorite foods in each type of eating categories ###

# In[ ]:


df['dish_liked']=df['dish_liked'].apply(lambda x : x.split(',') if type(x)==str else [''])
df['dish_liked']
rest=df['rest_type'].value_counts()[:9].index
rest
produce_wordcloud(rest) 


# In[ ]:


print(rest)


# Here we try to identify the relationship between variables like cost, Rates and Votes

# In[ ]:


numeric_vars = ['avg_cost','rate','votes']
plt.figure(figsize = [8, 5])
sns.heatmap(df[numeric_vars].corr(), annot = True, fmt = '.3f',cmap = 'vlag_r', center = 0)
plt.show()


# Votes definitely have an impact on rates. But cost overall, doesnt seem to have an impact on rates

# In[ ]:




