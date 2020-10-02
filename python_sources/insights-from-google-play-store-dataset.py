#!/usr/bin/env python
# coding: utf-8

# In[161]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../inp ut/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import MinMaxScaler
import os
print(os.listdir("../input"))
scale  = MinMaxScaler()
# Any results you write to the current directory are saved as output.


# In[162]:


googleps_data = pd.read_csv('../input/googleplaystore.csv')
user_reviews = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# ### <font color = 'blue'>Missing Value Analysis</font>

# In[163]:


(googleps_data.isna().sum()/len(googleps_data))*100


# In[164]:


#Imputing the nan by shifting to the right adjacent column value of Life Made Wi-Fi Touchscreen Photo Frame

googleps_data.loc[10472,'Android Ver'] = googleps_data.loc[10472,'Current Ver']
googleps_data.loc[10472,'Current Ver'] = googleps_data.loc[10472,'Last Updated']
googleps_data.loc[10472,'Last Updated'] = googleps_data.loc[10472,'Genres']
googleps_data.loc[10472,'Genres'] = googleps_data.loc[10472,'Content Rating']
googleps_data.loc[10472,'Content Rating'] = googleps_data.loc[10472,'Price']
googleps_data.loc[10472,'Price'] = googleps_data.loc[10472,'Type']
googleps_data.loc[10472,'Type'] = googleps_data.loc[10472,'Installs']
googleps_data.loc[10472,'Installs'] = googleps_data.loc[10472,'Size']
googleps_data.loc[10472,'Size'] = googleps_data.loc[10472,'Reviews']
googleps_data.loc[10472,'Reviews'] = googleps_data.loc[10472,'Rating']
googleps_data.loc[10472,'Rating'] = googleps_data.loc[10472,'Category']
#Photo-Frams lie in the PHOTOGAPHY CATEGORY
googleps_data.loc[10472,'Category'] = 'PHOTOGRAPHY'
#Doing little feature engineering to check correlation between 


# In[165]:


googleps_data['Rating']= googleps_data["Rating"].astype(np.float64)


# In[166]:


#Imputing for missing Android Version with "4.0 and above because of the apps with that rating and installs have that Android Version"
googleps_data.loc[4453,'Android Ver'] = '4.0 and up'
googleps_data.loc[4490,'Android Ver'] = '4.0 and up'


# In[167]:


####### Since most of the data is in version 1.0, we'll impute it by version 1.0
for item in list(googleps_data[googleps_data['Current Ver'].isna()].index):
    googleps_data.loc[item,'Current Ver'] = '1.0'
    
googleps_data.loc[10472,'Genres'] ='Photography'
#The Price is zero therefore the Type is Free
googleps_data.loc[9148,'Type'] = 'Free'
#converting the column to int64
googleps_data['Reviews'] = googleps_data['Reviews'].astype('int64')


# As we can see, 13.6% values are missing in Rating Column and <1% value is missing in Type, Content Rating, Current Ver, Android Ver

# In[168]:


googleps_data['Installs'] = googleps_data['Installs'].apply(lambda x : x.replace('+',''))
googleps_data['Installs'] = googleps_data['Installs'].apply(lambda x : x.replace (',',''))
googleps_data[googleps_data['App'].str.contains('Theme')]['Android Ver'].value_counts()


# In[ ]:





# In[169]:


googleps_data.loc[9148,'Type'] = 'Free'
googleps_data.loc[9148,'Rating'] = 0
for it in list(googleps_data[googleps_data.Rating.isna()].index):
    googleps_data.loc[it,'Rating'] = 0
    


# Since the rest of photo frame apps are in PHOTOGRAPHY, then we'll put 'Life Made WI-Fi Touchscreen Photo Frame' in the same category.
# #### Missing Values imputed successfully

# In[170]:


#copying the dataset
copy_dataset = googleps_data.copy()


# ## Feature Engineering for EDA

# In[171]:


#Removing all the duplicate values
copy_dataset.drop_duplicates(subset = 'App', keep='first', inplace=True)


# In[172]:


#Best CG Photography is a photography app and can be used by anyone 
copy_dataset.loc[7312,'Content Rating'] = 'Everyone'
#DC UNiverse Online Maps is a topic most popular among teens
copy_dataset.loc[8266,'Content Rating'] = 'Teen'


# In[173]:


#converting to required data types
copy_dataset['Rating'] = copy_dataset['Rating'].astype('float64')
copy_dataset['Price'] = copy_dataset['Price'].apply(lambda x : x.replace('$',''))
copy_dataset['Price'] = copy_dataset['Price'].astype('float64')
copy_dataset['Installs'] = copy_dataset['Installs'].astype('int64')

#Replacing all the sizes having less than 1MB with 1MB
copy_dataset.loc[copy_dataset['Size'].str.contains('k'), 'Size']  = '1M'
#Removing the M and k so that the number becomes comparable
copy_dataset['Size'] =  copy_dataset['Size'].apply(lambda x : x.replace('M',''))
copy_dataset['Size'] =  copy_dataset['Size'].apply(lambda x : x.replace('k',''))

#Let's assume apps whose size varies with device are of 0 just to do some EDA
copy_dataset.loc[copy_dataset['Size'].str.contains('Varies'), 'Size']  = 0
copy_dataset['Size'] = copy_dataset['Size'].astype('float64')


# In[174]:


## Reducing the Genres into one simple category
copy_dataset['Genres'] = copy_dataset['Genres'].apply(lambda x : x.split(";")[0])
copy_dataset['Genres'].value_counts()


#  ## EDA
# Doing exloratory data analysis on the google play store dataset. I believe it is important to start any analysis with a prior assumption and hypothesis. Here are my assumptions (What I think is true for this datasets) and hypothesis (Theory I believe will be true):
# 1. The ratings and number of Reviews will be linearly correlated
# 2. More the Rating and Reviews -  more the number of installs
# 3. Apps which are updated more recently and have been constantly updated in the past would have more installs
#   

# In[175]:


def count_relationship(a,b):
    df = copy_dataset.groupby(a)[b].count()
    df = df.reset_index()
    df = df.sort_values(by=[b])
    return df.plot.barh(x=a,y=b, figsize = (12,10))


# In[176]:


def sum_relationship(a,b):
    df = copy_dataset.groupby(a)[b].sum()
    df = df.reset_index()
    df[b] = (df[b]*100)/sum(copy_dataset[b])
    df = df.sort_values(by=[b])
    return df.plot.barh(x=a,y=b, figsize = (12,20))


# In[177]:


# Apps with most installs genre, size, cateory ,reviews and ratingss
count_relationship('Category','Installs')


# The Most number of Apps on the play store are in the category of:
#  - Family
#  - Game
#  - Tools

# In[178]:


for a in (copy_dataset['Category'].unique()):
    if (len(copy_dataset[copy_dataset['Category'] == a].groupby('Genres').count()) > 1) :
        print (a)


# In[179]:


fig, axarr = plt.subplots(4, 2, figsize=(26, 15))
plt.subplots_adjust(top=1.2, hspace=0.5)

for i,col in enumerate(['GAME','FAMILY']):
    axarr[0][i].set_title(col)
    axarr[0][i].set(ylabel = 'Number of Apps' )
    axarr[1][i].set(ylabel = 'Total Installs')
    axarr[2][i].set(ylabel = 'Average Ratings')
    axarr[3][i].set(ylabel = 'Number of people who rated')
    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Installs'].count().plot.bar(ax = axarr[0][i])
    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Installs'].sum().plot.bar(ax = axarr[1][i])    
    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Rating'].mean().plot.bar(ax = axarr[2][i])
    copy_dataset[copy_dataset['Category'] == col].groupby('Genres')['Rating'].count().plot.bar(ax = axarr[3][i])    


# #### In games, the most common genre is 
# 1. Action
#   - Has the most number of Apps
#   - Comes 2nd when it comes to most number of Installs after Arcade
#   - Action has aroun 4.0 star rating
# - Arcade 
#   - Arcade category has 2nd most number of Apps
#   - It has the most number of Installs
#   - Its Rating is similar tothe Action category.
# - Racing
#   - This Genre has 3rd most number of Apps on the play store
#   - But it is 4th in terms of number of Installs, comes after Casual.
#   - The average rating is also bad in the Games category overall
# 
# #### In Family the most installed Genre is :
# 1. Education
#   - 2nd Most number of Apps on the Play Store under this FAMILY Genre
#   - Surprisingly this genre doesn't has that much number of App Installs
#   - The average rating is around 3.3 which is not very good.
#   
# - Entertainment
#   - Most popular genre in Family Category, Has the most number of Apps
#   - The number of Installs although are at 2nd after Causal Genre
#   - The average rating is not upyo par with other categories, also reason being the variety of apps with more number of people who rated. 
# - Casual
#   - Number of Apps are far lesser than Education and Entertainment but still comes 3rd
#   - The number of Installs are pretty high infac this genre has most number of Installs surprisingly
#   - The mean rating is 3.8 which is above average in FAMILY category.

# #### Insights
#  - Causal in both FAMILY & GAME both is a good genre to make Apps in
#  - One should definitely not make apps in Education Genre in FAMILY
#  - Puzzle Games are a good niche to make games in 
#    - Less Number of Apps 
#    - More number of Installs
#    - Good Ratings
#  - Racing Games doesn't has most number of Installs infact they have bad ratings but there are lot of such games on the play store.
#    - Definitely not a good genre to make apps in. 
#    - Highly populated with Apps
#    - Saturated Market

# In[180]:


plt.figure(figsize=(12,6))
pd.DataFrame(copy_dataset[copy_dataset['Type'] == 'Paid'].groupby(['Category','Genres']).mean()['Rating'] ).sort_values(['Rating'],ascending = False).plot.bar(figsize=(24,6))   
plt.title('Ratings of Paid Apps Grouped by Category and Genres')


# #### Price Comparison of the Apps across various Categories and Genres
# - People really pay for News/Magazines(and those are highly rated i.e. people are happy paying it)
# - Education Apps also see good grossings
# - People are satisfied paying for Games, Video Editors, Shoppings Apps, Sports.
# 
# - Nobody wants to pay for Event Apps and Library & Demo
# - One should never make dating, auto and vehicles apps paid

# ### Effect of Content Rating & App Version
# - The Content Rating and App Version doesn't has any significant effect on number of Installs
# - The Number of Installs are more for high rated and reviewed Apps
# - The Number of Installs are less for less rated and reviewed Apps

# In[181]:


corr = copy_dataset.corr()
corr.style.background_gradient(cmap='coolwarm')


# Reviews are somewhat correlated with number of App Installs.
# - More reviews means more Installs therefore always make sure people review your apps.
# - Ratings are not directly correlated with Reviews....Surprisingly
