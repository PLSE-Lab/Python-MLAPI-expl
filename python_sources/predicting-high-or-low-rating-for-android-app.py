#!/usr/bin/env python
# coding: utf-8

# ###  Good day everyone, this is my first kernel. I am happy to share it with you.
# 
# ## Aim : To predict if an app would have a high rating or low rating knowing its number of downloads, the category it belongs, number of reviews on playstore, and app size.
# 
# ## Steps:
# * Performed a short descriptive analysis of the dataset,
# * Cleaned the data
# * Performed conversion from one datatype to the other
# * Applied labelEncoding and oneHotEncoding, Label Encoding
# * Applied machine learning algorith such as K-nearest neighbour and Random Forest.,
# 
# I was able to get a 90% score for my model which means I can be wrong 10 times in 100.
# 
# ### I hope you enjoy it and please drop your feedback.
# ### Let's go

# In[ ]:


# Let's import the necessary tools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for data visualization
import seaborn as sns


# In[ ]:


# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# In[ ]:


# Get in the data
# I am using only the googleplaystore.csv file in this kernel
path = '../input'
play_store_data = pd.read_csv(path + "/googleplaystore.csv")


#  # Data Exploration and Cleaning

# In[ ]:


play_store_data.head(10)


# In[ ]:


play_store_data.shape


# In[ ]:


play_store_data.info()


# ### Observation:  There are 10841 uncleaned samples for analysis with 13 columns. Only the Ratings column is represented as numeric. Other 'numeric' columns such as Size, number of installs, number of reviews need to be worked on

# In[ ]:


# Starting with the easiest.
# Convert Reviews to numeric
play_store_data['Reviews'] = pd.to_numeric(play_store_data.Reviews, errors = 'coerce')


# In[ ]:


play_store_data.info()


# In[ ]:


#Let's look closely at the apps in the data 
play_store_data.App.value_counts().head(20)


# ###  Observation: App duplicates !!!
# ### We will need to take care of the duplicate entries for Apps in the dataset, but let's see if there are differences between the App entries or they are exactly the same.

# In[ ]:


#Taking 3 sample Apps for exploration
play_store_data[play_store_data['App'].isin(['ROBLOX', 'Candy Crush Saga','Granny'])].sort_values(by='App')


# Inspecting some of the duplicate values indicate that they have the same data with minor changes in their **number of reviews ** during crawling.
# 
# I can therefore drop duplicates of each App keeping the one with the** highest review** at the time

# In[ ]:


# Sort App in Ascending order of reviews
play_store_data_sorted = play_store_data.sort_values(by = ['App', 'Reviews'], ascending = True)

#drops other duplicate entries keeping the App with the highest reviews
play_store_data_sorted.drop_duplicates('App',keep='last',inplace=True)


# In[ ]:


#Let's verify that duplicates has been removed
play_store_data_sorted.App.value_counts().head(10)


# In[ ]:


play_store_data_sorted.shape


# ### Observation: Data sample has been redured to 9660 from 10841 samples due to duplicate entries

# In[ ]:


# Let's check out the App categories
play_store_data_sorted.Category.value_counts()


# In[ ]:


# Drop the category named 1.9, unknown category
play_store_data_sorted[play_store_data_sorted['Category'] == '1.9']


# ### Observation: What is a 1.9 category? That has to be removed or given the correct value. I go with remove.

# In[ ]:


play_store_data_sorted = play_store_data_sorted.drop([10472])


# In[ ]:


#Let's check for null values and start dealing with them.
play_store_data_sorted.isnull().sum()


# ### Observation: Ratings is not given for over 15 percent of the data. which is necessary for my analysis.
# 
# ### I will have to remove rows with NA ratings since this is what I will be predicting.

# In[ ]:


play_store_data_sorted.dropna(axis = 0, inplace = True, subset = ['Rating'])


# In[ ]:


play_store_data_sorted.isnull().sum()


# # Type Conversions from Object to Numeric
# 1. Size in Megabyte (1e6), Kilobyte (1e3) and a 3rd option, "Varies with Device"
# 2. Number of Instals, removing "+" and ","
# 

# ## Working on Size column
# ### The column currently contail alphanumeric values. I call a function to work on the strings and convert to numbers

# In[ ]:


play_store_data_sorted.Size.value_counts()


# In[ ]:


#Convert non numeric values in App size to NAN
play_store_data_sorted['Size'][play_store_data_sorted['Size'] == 'Varies with device'] = np.nan

#Replace M with 1 million and k with 1 thousand
play_store_data_sorted['Size'] = play_store_data_sorted.Size.str.replace('M', 'e6')
play_store_data_sorted['Size'] = play_store_data_sorted.Size.str.replace('k', 'e3')

#convert column to numeric, dropping non numeric values
play_store_data_sorted['Size'] = pd.to_numeric(play_store_data_sorted['Size'], errors = 'coerce')


# In[ ]:


play_store_data_sorted.info()


# ## Observation: Ratings, Reviews and Size are now numeric columns.
#   ### Let's move on to Installs

# In[ ]:


play_store_data_sorted['Installs'].value_counts()


# In[ ]:


# To eliminate the '+' and ',' signs and convert to numeric
play_store_data_sorted['Installs'] = play_store_data_sorted.Installs.str.replace('+', '')
play_store_data_sorted['Installs'] = play_store_data_sorted.Installs.str.replace(',', '')

# Convert to numeric type
play_store_data_sorted['Installs'] = pd.to_numeric(play_store_data_sorted['Installs'], errors = 'coerce')


# In[ ]:


play_store_data_sorted['Installs'].value_counts()


# ## Create bins for the Install size as it was given in data

# In[ ]:


#Get the bin levels
bin_array = play_store_data_sorted.Installs.sort_values().unique()
#convert to array
bins = [x for x in bin_array]

# Added 5 billion for the higher range of app installs
bins.append(5000000000)


# In[ ]:


#Create bins for Installs
play_store_data_sorted['Installs_binned'] = pd.cut(play_store_data_sorted['Installs'], bins)

# Digitize the bins for encoding
Installs_digitized = np.digitize(play_store_data_sorted['Installs'], bins = bins )

#Add to the data frame as a column
play_store_data_sorted = play_store_data_sorted.assign(Installs_d = pd.Series(Installs_digitized).values)


# In[ ]:


play_store_data_sorted.info()


# ## Data cleaning done for the prediction

# In[ ]:


play_store_data_sorted.describe()


# ## Observation: 
# ### The rating is on a scale of 1 - 5 with 1 being minimum and 5 being maximum
# ### The mean rating is 4.17 while the median rating is 4.3. This implies that average rating is greater than 4.1
# ### The minimum App size is 8.5 kb with maximum size being 100 Mb

# # Machine Learning  - Predicting Ratings
# 
# ### Using ['Category', 'Reviews', 'Size' , 'Installs'] to predict 'Rating'

# In[ ]:


#as most machine learning models do not work well with NA, I have to drop rows having them.
attributes = ['Category', 'Reviews', 'Size' , 'Installs_d','Rating']
psa = play_store_data_sorted[attributes].dropna().copy()
psa.shape


# ### Observation: 7020 samples are available for my training and testing

# ## A.Convert Ratings to two categories
# * High rating -: 3.5 - 5.0 
# * Low Rating  -:  < 3.5

# In[ ]:


#convert ratings to high and low categories.
Rating_cat = dict()
for i in range(0,len(psa['Rating'])):
    if psa['Rating'].iloc[i] >= 3.5:
        Rating_cat[i] = 'High'
    else: Rating_cat[i] = 'Low'
        
#Add the categorical column to the data 
psa = psa.assign(Rating_cat = pd.Series(Rating_cat).values)


# In[ ]:


psa['Rating_cat'].value_counts()


# In[ ]:


#drop the Ratings column
psa = psa.drop(['Rating'], axis = 1)

#To encode the Ratings labels for learning
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
psa['Rating_cat'] = le.fit_transform(psa['Rating_cat'])


# In[ ]:


#To view the encoded labels
list(le.classes_)


# In[ ]:


#Applying One-Hot Encoding to the Categorical Column 'Category' and 'Installs_d'
psa_encode = pd.get_dummies(psa, columns= ['Category','Installs_d'])
print(psa_encode.columns)


# In[ ]:


X = psa_encode.drop(['Rating_cat'], axis = 1)
y = psa_encode['Rating_cat']


# In[ ]:


#Checking for correlation using heatmap
plt.figure(figsize=(20,15)) 

sns.heatmap(X.corr())


# # Apply K- Nearest Neighbour to model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


print('Training Set Score: {} \nTest Set Score: {}'.format(knn.score(X_train, y_train),knn.score(X_test, y_test) ))


# ## Observation: The model seems to have  **overfitted** the data

# In[ ]:


# Looking for optimum value of n_neighbours for the dataset.
for i in range(1,7):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train);
    print('For n = {}, Test = {}, Train = {}'.format(i,knn.score(X_train, y_train),knn.score(X_test, y_test) ))


# * ### Observation: n_neighbours = 4 seems to optimise the model.

# ## Using Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10, max_depth = 10, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest.fit(X_train, y_train)


# In[ ]:


print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))


# ### We can apply this model to a sample and be correct 90% of the time.

# # Thank you.

# 

# 

# 

# 

# In[ ]:




