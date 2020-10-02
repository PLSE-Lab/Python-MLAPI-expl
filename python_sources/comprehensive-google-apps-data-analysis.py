#!/usr/bin/env python
# coding: utf-8

# # Breakdown of this notebook:
# 
# 1. **Data Cleaning**:
#               Removing redundant data
#               Dropping NA,nulls
#               Making new features
#               Renaming columns
# 1. **Data Visualization**:
#               Visualizing Distribution of the useful features
#               Visualizing relation between different features
# 1. **How much do Free Apps and Paid Apps differ?**
# 1. **Is there a relation between the Android Update Dates of all the Apps?**
# 1. **Which are the best Apps on App Store?**
# 1. **Which Category has more likeable Apps?**
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import os
from matplotlib import rcParams

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/googleplaystore.csv')
data2 = pd.read_csv('../input/googleplaystore_user_reviews.csv')

data.head()


# In[ ]:


print("Shape of the dataframe is",data.shape)
print("The number of nulls in each column are \n", data.isna().sum())


# In[ ]:


print("Percentage null or na values in df")
((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)


# In[ ]:


data.head()
# data.Installs.unique()
# data.Size.unique()


# In[ ]:


df=data
df.Installs = df.Installs.replace("Free", np.nan)
df.dropna(how ='any', inplace = True)
df.Installs = df.Installs.astype(str)
df.Installs = df.Installs.apply(lambda x: x.replace(',',''))
df.Installs = df.Installs.apply(lambda x: x.replace('+',''))
df.Installs = df.Installs.apply(lambda x: int(x))
df.head()


# In[ ]:


df['Size'].replace('Varies with device', np.nan, inplace = True ) 
df.Size = (df.Size.replace(r'[kM]+$', '', regex=True).astype(float) * df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))
df.Size = df.Size.apply(lambda x: x/(10**6))
df.rename(columns={'Size': 'Size(in MB)'}, inplace=True)
# df.Size.unique()
# df.head()


# In[ ]:


df.Category = df.Category.apply(lambda x: x.replace('_',' '))
df.Price = df.Price.apply(lambda x: x.replace('$',''))
df.rename(columns={'Price': 'Price(in $)'}, inplace=True)


# # Data Visualization

# In[ ]:


rcParams['figure.figsize'] = 15,7
category_plot = sns.countplot(x="Category",data=df, palette = "spring")
category_plot.set_xticklabels(category_plot.get_xticklabels(), rotation=90, ha="right")
category_plot 
plt.title('TOTAL apps in each category',size = 20)


# 
# **Family,Game and Tools** are the most downloaded categories.
# 
# 
# 

# In[ ]:


# X['Content Rating'].value_counts()
rcParams['figure.figsize'] = 15,7
content_plot = sns.countplot(x="Content Rating",data=df, palette = "pink")
content_plot.set_xticklabels(content_plot.get_xticklabels(), rotation=90, ha="right")
content_plot 
plt.title('Content Rating distribution',size = 20)


# * **Everyone , Teen** constitute the majority Content Rating of all the apps.
# * **Unrated and Adults ony 18+** have count less then 10.

# In[ ]:


# # X['Type'].value_counts()
rcParams['figure.figsize'] = 10,7
type_plot = sns.countplot(x="Type", data=df, palette = "twilight")
type_plot.set_xticklabels(type_plot.get_xticklabels(), rotation=90, ha="right")
type_plot 
plt.title('Number of Free Apps vs Paid Apps',size = 20)


# * The number of **FREE APPS**=7588
# * The number of **PAID APPS**=602
# 
# Most of the Apps in the App Store are **Free apps**.We will visualize both the Types in detail later in the kernel.

# In[ ]:


rcParams['figure.figsize'] = 10,7
type_size = sns.boxplot(x="Type",y="Size(in MB)", data=df, palette = "rainbow")
type_size.set_xticklabels(type_size.get_xticklabels(), rotation=90, ha="right")
type_size 
plt.title('Size Range for free and paid Apps',size = 20)


# In[ ]:


rcParams['figure.figsize'] = 15,10
content_price = sns.stripplot(y="Content Rating",x="Price(in $)", data=df, palette = "Set1")
content_price.set_xticklabels(content_price.get_xticklabels(), rotation=90, ha="right")
content_price 
plt.title('Content Rating vs Price',size = 20)


# * Most of the Apps with Content Rating EVERYONE lie below 50 dollars. However there are some apps that lie in the 350-400 dollars bracket!!
# * The mature rated apps(10+, 17+, 18+) are mosty free or cheap.

# ## PRICE DISTRIBUTION

# In[ ]:


df['Price(in $)'] = df['Price(in $)'].apply(lambda x: float(x))
rcParams['figure.figsize'] = 15,7
priced_apps=df[df['Price(in $)'] != 0.0]
price_plot = sns.countplot(priced_apps['Price(in $)'], palette = "inferno")
price_plot.set_xticklabels(price_plot.get_xticklabels(), rotation=90, ha="right")
price_plot 
plt.title('Number of apps for every price',size = 20)


# 1. We have filtered only the priced apps from the Price column to see the price distributution of priced apps.
# 2. Most paid apps are priced **3.02 Dollars** interestingly.
# 3. Paid apps mostly cost **3.02 dollars, 0.99 dollars ,5.49 dollars ,2.00 dollars or 4.29 dollars** 
# 

# ## INSTALLS DISTRIBUTION

# In[ ]:


# X['Installs'].value_counts()
rcParams['figure.figsize'] = 15,7
install_plot = sns.countplot(y="Installs",data=df, palette = "PuBu")
install_plot.set_xticklabels(install_plot.get_xticklabels(), rotation=90, ha="right")
install_plot 
plt.title('Installs count distribution',size = 20)


# Most of the apps in the dataframe have **1 million** installs followed by **10 million** and **100k**.

# ## RATINGS

# In[ ]:


rcParams['figure.figsize'] = 15,7
ratings_plot = sns.countplot(x="Rating",data=df, palette = "inferno")
ratings_plot.set_xticklabels(ratings_plot.get_xticklabels(), rotation=90, ha="right")
ratings_plot 
plt.title('Rating Distribution',size = 20)


# 1. The Ratings distribution is **left skewed.**
# 2. Most of the apps are rated **between 4.0 and 4.7.**

# In[ ]:


content_ratings = sns.violinplot(x="Content Rating",y="Rating",data=df, kind="box", height = 10 ,palette = "Set3")
content_ratings.set_xlabel(xlabel = 'Content Rating', fontsize = 9)
content_ratings.set_ylabel(ylabel = 'Rating', fontsize = 9)
content_ratings.set_title(label = 'Rating vs Content Rating', fontsize = 20)
plt.show()


# # How much do Free Apps and Paid Apps differ?

# ### Exploring Free Apps

# ### First step is to get the most installed free apps.

# In[ ]:


# X.Installs.mean()
# X.Installs.median()
df= df.drop_duplicates(subset='App',keep='first')
df['Installs'] = df['Installs'].apply(lambda x: int(x))
df[['App','Installs','Category','Content Rating','Price(in $)']].head()
newdf=df[['App','Installs','Category','Content Rating','Price(in $)','Reviews','Genres']].groupby(['Installs'], sort = True)
newdf=newdf.filter(lambda x: x['Installs'].mean() >= 1000000000)
newdf=newdf.sort_values(by=['Installs'])
newdf


# ### Next step is to visualize the TOP CATEGORIES out of most installed FREE APPS in the dataframe obtained above.

# In[ ]:


rcParams['figure.figsize'] = 15,7
free_categories = sns.countplot(x="Category",data=newdf, palette = "bone")
free_categories.set_xticklabels(free_categories.get_xticklabels(), rotation=90, ha="right")
free_categories 
plt.title('Top Categories for Free Apps',size = 20)


# ### Now, lets get the TOP GENRES out of all the highest installed FREE APPS

# In[ ]:


rcParams['figure.figsize'] = 15,7
free_genres = sns.countplot(y="Genres",data=newdf, palette = "spring")
free_genres.set_xticklabels(free_genres.get_xticklabels(), rotation=90, ha="right")
free_genres 
plt.title('Top Genres for Free Apps',size = 20)


# ## Exploring Paid Apps

# ### First step is to find the highest installed paid Apps

# In[ ]:


new=df[['App','Category','Content Rating','Price(in $)','Reviews']].groupby(['Price(in $)'], sort = True)
new=new.filter(lambda x: x['Price(in $)'].mean() != 0)
new=new.sort_values(by=['Price(in $)'])
new
newdf2=df[['App','Installs','Genres']].groupby(['Installs'], sort = True)
newdf2=newdf2.filter(lambda x: x['Installs'].mean() >= 1000000)
newdf2=newdf2.sort_values(by=['Installs'])
newdf2

s1 = pd.merge(new, newdf2, how='inner', on=['App'])
s1


# ### Next step is to visualize the TOP CATEGORIES out of most installed PAID APPS in the dataframe obtained above.

# In[ ]:


rcParams['figure.figsize'] = 15,7
paid_categories = sns.countplot(x="Category",data=s1, palette = "bone")
paid_categories.set_xticklabels(paid_categories.get_xticklabels(), rotation=90, ha="right")
paid_categories 
plt.title('Top Categories for PAID APPS',size = 20)


# ### Now, lets get the TOP GENRES out of all the highest installed PAID APPS

# In[ ]:


rcParams['figure.figsize'] = 15,7
paid_genres = sns.countplot(y="Genres",data=s1, palette = "spring")
paid_genres.set_xticklabels(paid_genres.get_xticklabels(), rotation=90, ha="right")
paid_genres 
plt.title('Top Genres for PAID APPS',size = 20)


# # Summary for Free vs Paid Apps:-

# 1. **People tend to spend their money on games and Personalization but when it comes to Social Networking and Communication they tend to install the free Apps on the Android Store**.
# 1.  The **2 most installed Paid Apps** are both **games(Minecraft and Hitman:Sniper)**
# 1.  The majority of most installed **free** Apps lie in the **Communication** Category.
# 1.  The majority of most installed **Paid** Apps lie in the **Gaming** Category.
# 1. The **Content Rating** for Free apps is mostly for Everyone or Teen , but for Paid Apps 6 out of 22 apps are not for users under 10 years of age.

# # Is there a relation between the Android Update Dates of all the Apps?

# In[ ]:


df['new'] = pd.to_datetime(df['Last Updated'])
df.drop(labels = ['Last Updated'], axis = 1, inplace = True)
df.rename(columns={'new': 'Last Updated'}, inplace=True)


# In[ ]:


freq= pd.Series()
freq=df['Last Updated'].value_counts()
newfreq=freq[freq>50]
newfreq.plot()
plt.xlabel("Dates")
plt.ylabel("Number of updates")
plt.title("Time series plot of Last Updates")


# * I have filtered and selected the dates on which most number of Apps were last updated.
# * The above plot shows that between **July 15,2018** and **August 1,2018** majority of the Apps were updated as is clear by the upward trend in the graph above.

# # Which are the best Apps on App Store?

# ### The criteria I have selected are:
# 1. High **rating**
# 1. Large number of **reviews**
# 1. Large number of **installs**
# 
# 
# * I haven't taken **Price** as a parameter for this selection because I believe an expensive app can be perceived to be better than a similar free app if it has high rating,reviews and installs.
# * I think features like **Content Rating, Android Version, Last Update time, Size of the App, Current version of the App** are irrelevant in this selection as well for obvious reasons.

# In[ ]:


df['Rating'] = df['Rating'].apply(lambda x: float(x))
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))

newdf_rate=df[['App','Rating','Category','Content Rating']].groupby(['Rating'], sort = True)
newdf_rate=newdf_rate.filter(lambda x: x['Rating'].mean() >= 4.5)
newdf_rate=newdf_rate.sort_values(by=['Rating'])

newdf_reviews=df[['App','Reviews']].groupby(['Reviews'], sort = True)
newdf_reviews=newdf_reviews.filter(lambda x: x['Reviews'].mean() >= 255435)
newdf_reviews=newdf_reviews.sort_values(by=['Reviews'])

newdf_installs=df[['App','Installs']].groupby(['Installs'], sort = True)
newdf_installs=newdf_installs.filter(lambda x: x['Installs'].mean() >= 10000000)
newdf_installs=newdf_installs.sort_values(by=['Installs'])

s1 = pd.merge(newdf_reviews, newdf_rate, how='inner', on=['App'])
s2 = pd.merge(s1, newdf_installs, how='inner', on=['App'])
s2


# As you can see only **359** apps make it to this list from all categories.The apps you see in this list are well known and famous and best in their respective category.
# To name a few, apps like **Google Photos, Instagram, Subway surfers,Clash of Clans, Shareit, Dictionary** etc are part of this list.
# 
# 

#  # Which Category has more likeable Apps?

# **I think an App which has large number of reviews , high rating on App store and large number of installs can be deemed to be likeable for any new user * i.e * *any new user is more likely to like the App experience if these criteria are met.***
# 
# *From the dataframe obtained above we'll plot and see which categories make up most of the likeable apps.*

# In[ ]:


rcParams['figure.figsize'] = 15,7
likeable_apps = sns.countplot(y="Category",data=s2, palette = "Set3")
likeable_apps.set_xticklabels(likeable_apps.get_xticklabels(), rotation=90, ha="right")
likeable_apps 
plt.title('CATEGORIES OF MOST LIKEABLE APPS ON THE ANDROID APP STORE',size = 20)


#  **Gaming Apps** outruns other categories by huge margin in the criteria of our most LIKEABLE category.
# 
# **Does this mean people tend to give reviews and ratings whenever they play a game?** I think yes! **The graph above suggests that you are more likely to give a rating or review for a gaming app then any other app.** Games like **Subway surfer,8ballpool, Clash royale, Clash of Clans** have millions of reviews,high rating and millions of installs.
# 
# Please note that Communication Apps like **Instagram** etc individually have more number of installs, reviews and higher rating but more Gaming Apps make it to this list than Communication Apps as is evident by the plot above.

# 
# 
# 

# ## Please upvote and feel free to give any feedback/comment below!!
