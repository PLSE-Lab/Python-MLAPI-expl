#!/usr/bin/env python
# coding: utf-8

# # Android Market Analysis

# by [Prashant Brahmbhatt](https://www.github.com/hashbanger)

# --------------

# ![App](https://ewerdroid.com.br/wp-content/uploads/2018/02/maxresdefault-1440x564_c.jpg)

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pylab import rcParams


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


plt.style.use('ggplot')


# ### Importing the data

# In[ ]:


df = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


df.head()


# In[ ]:


print("The shape of the data is ",df.shape)


# #### Data Summary

# In[ ]:


df.describe().T


# Rating is the only numerical column in the data

# ## Missing values

# First glance at the missing values

# In[ ]:


plt.figure(figsize=(8,6,))
sns.heatmap(df.isnull(), cbar = False)


# We can observe that there are considerable missing values in the **Rating** column

# In[ ]:


total = df.isnull().sum().sort_values(ascending  = False)
percent = (df.isnull().sum()/df.count()).sort_values(ascending = False)
temp = pd.concat([total, percent], axis = 1, keys = ['total','percentage'])
temp.head()


# There are some amount of missing values in other columns as well which are initially not observable in the heatmap

# In[ ]:


#Dropping observations having missing values in any column
df.dropna(how = 'any', inplace = True)


# #### Checking for Duplicate apps

# In[ ]:


print("Length of Unique App names = ", len(df['App'].unique()))
print("Legth of the Total App name = ", df.shape[0])
print("Duplicate Apps = ",df.shape[0]- len(df['App'].unique()))


# We can see that there are 1181 duplicate apps, and this is true as we can see for some apps like:

# In[ ]:


df[df['App'] == 'Coloring book moana']


# So we better remove duplicacy as well and keeping first observations

# In[ ]:


df.drop_duplicates(subset = 'App', keep = 'first', inplace = True)


# ______

# ## EDA

# Visualising the percetages of CATEGORIES in the playstore

# In[ ]:


temp = df['Category'].value_counts().reset_index() #A temporary dataframe for this plot

plt.figure(figsize=(12,12))
ax = plt.subplot(111)
plt.pie(x = temp['Category'], labels= temp['index'],autopct= '%1.1f%%')
plt.legend()
ax.legend(bbox_to_anchor=(1.4, 1))
plt.show()


# The **FAMILY**, **EVENTS** and **TOOLS** are the most dominating applications in the playstore

# ### -----------------------------------------------------RATING-----------------------------------------------------------------------

# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df['Rating'])
plt.legend(['Rating'])
plt.show()


# In[ ]:


print("The average rating in the appstore is ",np.average(df['Rating']))


# Visualising the most often rated catgories

# In[ ]:


top = np.array(df.Category.value_counts().reset_index()['index'])
print("Most Occuring Categories\n",top[:6])


# In[ ]:


plt.figure(figsize= (15,10))
plt.suptitle("Ratings of Different Categories",fontsize = 22)

plt.subplot(2,3,1)
sns.kdeplot(df[df['Category'] == 'FAMILY']['Rating'], shade = True)
plt.title('Rating of FAMILY Apps')

plt.subplot(2,3,2)
sns.kdeplot(df[df['Category'] == 'GAME']['Rating'], shade = True)
plt.title('Rating of GAME Apps')


plt.subplot(2,3,3)
sns.kdeplot(df[df['Category'] == 'TOOLS']['Rating'], shade = True)
plt.title('Rating of TOOLS Apps')


plt.subplot(2,3,4)
sns.kdeplot(df[df['Category'] == 'FINANCE']['Rating'], shade = True)
plt.title('Rating of FINANCE Apps')


plt.subplot(2,3,5)
sns.kdeplot(df[df['Category'] == 'LIFESTYLE']['Rating'], shade = True)
plt.title('Rating of LIFESTYLE Apps')


plt.subplot(2,3,6)
sns.kdeplot(df[df['Category'] == 'PRODUCTIVITY']['Rating'], shade = True)
plt.title('Rating of PRODUCTIVITY Apps')

plt.show()


# We can observe that ratings vary according to the categories

# #### ANOVA Analysis

# Null Hypothesis - All Means are equal  
# Alternate Hypothesis - Atleast one mean is different

# In[ ]:


import scipy.stats as stats
htest = stats.f_oneway(df[df['Category'] == 'FAMILY']['Rating'],
              df[df['Category'] == 'GAME']['Rating'],
              df[df['Category'] == 'TOOLS']['Rating'],
              df[df['Category'] == 'FINANCE']['Rating'],
              df[df['Category'] == 'PRODUCTIVITY']['Rating'],
              df[df['Category'] == 'LIFESTYLE']['Rating'],
              )
print("The P value of the test is ",htest[1])


# Since the **p-Value** is small we reject the Null Hypothesis

# In[ ]:


plt.figure(figsize=(18,9))
f = sns.violinplot(x = df['Category'], y = df['Rating'], palette= 'coolwarm')
f.set_xticklabels(f.get_xticklabels(), rotation = 90)
plt.show()


# Categories like **EVENTS** , **BOOKS_AND_REFERENCE** and **HEALTH_AND_FITNESS** are the best performing with more than half of the apps rated above average.  
# While categories like **DATING** are worst performing.

# ### -----------------------------------------------------REVIEWS---------------------------------------------------------------------------

# In[ ]:


print(df['Reviews'].head())


# Converting the value to integer type

# In[ ]:


df['Reviews'] = df['Reviews'].astype(dtype = 'int')
plt.figure(figsize=(15,8))
sns.kdeplot(df['Reviews'], color = 'Green', shade = True)
plt.title('Distribution of Ratings')


# In[ ]:


print("Number of Apps with more than 1M reviews",df[df['Reviews'] > 1000000].shape[0])
print("\nTop 20 apps with most reviews: \n",df[df['Reviews'] > 1000000].sort_values(by = 'Reviews', ascending = False).head(20)['App'])


# ![Fb](https://parentinfo.org/sites/default/files/styles/main_article_image/public/Social_Media_image_New_IG_Logo.png?itok=jDLmx9B5)
# 

# ### Ratings vs Reviews

# In[ ]:


print("For all apps")
sns.jointplot(x = 'Reviews', y= 'Rating',data = df[df['Reviews']>100000], color = 'darkorange') 
plt.show()

print("For apps below 1M reviews")
sns.jointplot(x = 'Reviews', y= 'Rating',data = df[df['Reviews']<100000], color = 'darkorange') 
plt.show()


# Observation: (**Joint Scatter Plot**)  
# The most reviewed apps are likely to be better rated as well

# ### -----------------------------------------------------INSTALLS---------------------------------------------------------------------------

# In[ ]:


df['Installs'].dtype


# Since the data type is still object we should convert it to integer for plotting 

# In[ ]:


df['Installs'].head()


# Before parsing to integer we better remove the commas and plus symbols

# In[ ]:


df['Installs'] = df['Installs'].apply(lambda x: x.replace(',',''))
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+',''))
df['Installs'] = df['Installs'].astype(dtype = 'int')


# In[ ]:


df['Installs'].head()


# In[ ]:


df['Installs'].unique()


# In[ ]:


plt.figure(figsize=(12,8))
f = sns.countplot(df['Installs'], palette= "viridis" )
f.set_xticklabels(f.get_xticklabels(), rotation = 30)
plt.show()


# We can observe that largest number of apps belong to 1M+ installs 

# We can treat these values as intervals and map them to numeric categories as:  
# 5+ installs --> 1  
# 50+ installs --> 2
# 

# In[ ]:


sorted_values = sorted(df['Installs'].unique())
df['Installs Classes'] = df['Installs'].replace(sorted_values, range(0,len(sorted_values)))
df['Installs Classes'].head()


# In[ ]:


plt.figure(figsize=(12,9))
sns.boxplot(y = df['Rating'], x = df['Installs Classes'], palette= 'Blues')
plt.show()


# It seems that there is less variation in **Rating** of apps with higher **Installs**

# In[ ]:


import scipy.stats as sp

plt.figure(figsize=(13,13))
plt.subplot(2,2,1)
f = sns.kdeplot(df[df['Installs Classes'] == 5]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 5 installs")
f.set_xticks([1,2,3,4,5])

plt.subplot(2,2,2)
f = sns.kdeplot(df[df['Installs Classes'] == 6]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 500 installs")
f.set_xticks([1,2,3,4,5])

plt.subplot(2,2,3)
f = sns.kdeplot(df[df['Installs Classes'] == 17]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 500M installs")
f.set_xticks([1,2,3,4,5])

plt.subplot(2,2,4)
f = sns.kdeplot(df[df['Installs Classes'] == 18]['Rating'], shade = True, color = 'purple')
plt.title("Ratings variation for apps above 1B installs")
f.set_xticks([1,2,3,4,5])

plt.show()
print("Variation in Rating of installs above 100 installs ",sp.variation(df[df['Installs Classes'] == 5]['Rating']))
print("Variation in Rating of installs above 500 installs ",sp.variation(df[df['Installs Classes'] == 6]['Rating']))
print("Variation in Rating of installs above 500M installs ",sp.variation(df[df['Installs Classes'] == 17]['Rating']))
print("Variation in Rating of installs above 1B installs ",sp.variation(df[df['Installs Classes'] == 18]['Rating']))


# From the above distributions it is evident that higher installed apps have less variation in their ratings.

# ### -----------------------------------------------------SIZE---------------------------------------------------------------------------

# In[ ]:


df['Size'].head()


# In[ ]:


print(df['Size'].unique())


# There's an uncanny observation as a value has label like *'Varies with device'* rather than the actual size. So we better replace it with some mean or median values

# There are two problems with this column:  
# * It contains values named 'Varies with device'
# * It contains *M*s and *k*s in the size
# * Type of values is string

# We can't omit the Ms and Ks since that would mess up the KB and MB size scale. So we convert the MBs to KBs.  
# One way to do this is below

# In[ ]:


df['Size'] = df['Size'].apply(lambda x: x.replace('M', '*1000'))
df['Size'] = df['Size'].apply(lambda x: x.replace('k', ''))


# Using -1 as the signal for the values 'Varies with device' as it will allow eval to be used on the column.

# In[ ]:


df['Size'].replace('Varies with device', '-1', inplace = True)


# In[ ]:


df['Size'] = df['Size'].apply(lambda x: eval(x))


# In[ ]:


df['Size'] = df['Size'].replace(-1,np.nan) #Changing the values to null then we can fill them with mean value
df['Size'].fillna(np.mean(df['Size']), inplace = True) 


# Noew all the sizes in the data are in KB scale

# In[ ]:


plt.figure(figsize=(18,9))
sns.distplot(df['Size'], color = 'darkred')
plt.xlabel('Size in KBs')
plt.xticks(list(range(0, int(max(df['Size'])), 5000)))
plt.show()


# We can clearly observe the peaks near 20MBs and 4-5MBs so most apps are commonly of that size 

# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(x = df['Size'], y = df['Rating'], color = 'orange')
plt.xlabel('Size in KBs')
plt.ylabel('Rating')
plt.title('Rating vs Size')
plt.show()


# In[ ]:


plt.figure(figsize= (12,12))
sns.regplot(y = df['Size'], x = df['Installs Classes'], color = 'grey')
plt.title('Size vs Installs')
plt.show()


# It is pretty evident from the plot that mid-sized apps tend to perform mostly better.

# ### -----------------------------------------------------TYPE---------------------------------------------------------------------------

# In[ ]:


temp = df['Type'].value_counts().reset_index()

# plt.figure(figsize=(9,9))
rcParams['figure.figsize'] = 9,9
plt.pie(x = temp['Type'], labels= temp['index'], autopct= '%1.1f%%', colors = ['lightblue','lightgreen'], 
        shadow= True, explode=(0.25,0), startangle= 90)
plt.show()


# Only 7.4% of the Apps in the store are paid apps

# ### -----------------------------------------------------PRICE---------------------------------------------------------------------------

# In[ ]:


df['Price'].unique()


# converting the type to float

# In[ ]:


df['Price'] = df['Price'].apply(lambda x: x.replace('$',''))
df['Price'] = df['Price'].astype('float')


# In[ ]:


plt.figure(figsize=(10,7))
sns.kdeplot(df[df['Type'] == 'Paid']['Price'], color = 'blue', shade = True)
plt.xlabel('Prices of Apps')
plt.title('Pricing Distribution of Paid Apps')
plt.show()


# We can observe that even in the paid apps most of the apps are cheap!

# ### Rating vs Pricing

# In[ ]:


paid_prices = df[df['Type'] == 'Paid']['Price']


# In[ ]:


sns.jointplot(y = df[df['Type'] == 'Paid']['Rating'], x = df[df['Type'] == 'Paid']['Price'], color= 'teal')
plt.show()


# The above suggests that even the most expensive apps do not have exceptionally better rating and even less rated than many cheaper apps.

# We can better visualize pricing by breaking them down in different intervals.

# In[ ]:


df.loc[df['Price'] == 0,'Price_Class'] = 'Free'
df.loc[(df['Price'] > 0) & (df['Price'] <=1), 'Price_Class'] = 'Cheap'
df.loc[(df['Price'] > 1) & (df['Price'] <=3), 'Price_Class'] = 'Above Cheap'
df.loc[(df['Price'] > 3) & (df['Price'] <=6), 'Price_Class'] = 'Average'
df.loc[(df['Price'] > 6) & (df['Price'] <=16), 'Price_Class'] = 'Above Average'
df.loc[(df['Price'] > 16) & (df['Price'] <=40), 'Price_Class'] = 'Expensive'
df.loc[(df['Price'] > 40), 'Price_Class'] = 'Too Expensive'


# In[ ]:


temp = df[df['Type'] == 'Paid']['Price_Class'].value_counts().reset_index()

sns.barplot(x = temp['index'], y = temp['Price_Class'], palette= 'autumn')
plt.xlabel('Price Classes')
plt.ylabel('Counts')
plt.show()


# Pricing of the Apps is found to be mostly above Average(Normal)

# In[ ]:


df[['Price_Class','Rating','Reviews']].groupby('Price_Class').mean()


# In[ ]:


plt.figure(figsize=(13,10))
f = sns.violinplot(x = df['Price_Class'], y = df['Rating'], palette= 'Wistia')
f.set_xticklabels(f.get_xticklabels(), fontdict= {'fontsize':13})
f.set_xlabel('Price Class', fontdict= {'fontsize':17})
f.set_ylabel('Rating', fontdict= {'fontsize':17})
f.set_title('Rating vs Price Class',fontdict= {'fontsize':17})
plt.show()


# As can be observed the **Too Expensive** apps have average **Rating** less than others.

# ### -----------------------------------------------------ContentRating---------------------------------------------------------------------------

# In[ ]:


df['Content Rating'].head()


# In[ ]:


df['Content Rating'].unique()


# In[ ]:


plt.figure(figsize=(16,7))
plt.suptitle('Content Rating Shares on playstore')
plt.subplot(1,2,1)
sns.countplot(x = df['Content Rating'], palette='summer')

plt.subplot(1,2,2)
temp = df['Content Rating'].value_counts().reset_index()
plt.pie(x = temp['Content Rating'], labels = temp['index'])

plt.show()


# As observed the **Adult** or **Unrated** apps are almost nil

# In[ ]:


sns.boxplot(x = df['Content Rating'], y = df['Rating'], palette= 'hls')


# There is not much effect of the **Content Rating** on the **Rating**, the better rating trend in adult and unrated apps is due to very few observations.

# ### -----------------------------------------------------GENRE---------------------------------------------------------------------------

# In[ ]:


df['Genres'].head()


# In[ ]:


df['Genres'].unique()


# In[ ]:


df['Genres'].value_counts()


# Many of the **Genres** in the column have sub-genres so we better omit them

# In[ ]:


df['Genres'] = df['Genres'].apply(lambda x: x.split(';')[0])
df['Genres'].unique()


# In[ ]:


df['Genres'].value_counts()


# As we can observe that **Music** and **Music & Audio** are redundant so we better convert them as one.

# In[ ]:


df['Genres'].replace('Music & Audio','Music', inplace = True)


# In[ ]:


df['Genres'].value_counts().tail()


# Geeting the mean **Rating** and **Reviews** of each genre.

# In[ ]:


temp = df[['Genres','Rating','Reviews']].groupby(by = 'Genres').mean().sort_values(by = 'Rating',ascending = False)
print(temp.head(1))
print(temp.tail(1))


# As observed the **Dating** genre is the least rated on average, but there is not a very vast difference to the highest averagely rated **Events** genre.

# In[ ]:


plt.figure(figsize=(14,8))
f = sns.boxplot(x = df['Genres'], y = df['Rating'], palette= 'rainbow')
f.set_xticklabels(f.get_xticklabels(), rotation = 90)
plt.show()


# ### -----------------------------------------------------Last_Updated---------------------------------------------------------------------------

# In[ ]:


df['Last Updated'].head(10)


# The dates need to parsed from strings to dates

# You can follow the kernel [here](https://www.kaggle.com/hashbanger/data-cleaning-challenge-parsing-dates) for datetime handling in pandas

# In[ ]:


from datetime import datetime


# In[ ]:


df['Last Updated'] = pd.to_datetime(df['Last Updated'])


# Finding the lastest update in the data

# In[ ]:


df['Last Updated'].max()


# Getting the last update in days form

# In[ ]:


df['Last Updated TimeDelta'] = df['Last Updated'].max() - df['Last Updated'] 
print(df['Last Updated TimeDelta'][0])


# The observation above has a timedelta format so we can get days count from it

# Now we can look if there's a relation with rating and last update

# In[ ]:


sns.jointplot(df['Last Updated TimeDelta'].dt.days, df['Rating'], COLOR = 'brown')
plt.show()


# There's not a very strong effect of update on the app ratings

# ### -------------------------------------------Exploring_Correlations--------------------------------------------------------------

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot = True, cmap = 'Reds')
plt.show()


# There's one noticeable strong relation between **Reviews** and **Installs** which could mean that people tend to reviewed(likely download) the apps more reviewed than most rated.  

# In[ ]:


sns.regplot(x = df['Reviews'], y = df['Installs'], color = 'green')


# # de nada!

# Any suggestions or corrections are very much welcome!
# 
