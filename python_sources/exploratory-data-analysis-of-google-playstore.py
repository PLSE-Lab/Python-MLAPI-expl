#!/usr/bin/env python
# coding: utf-8

# ![](http://pmcvariety.files.wordpress.com/2018/03/google-play.png?w=1000&h=563&crop=1)

# ## Importing required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# ## **Loading Dataset**

# In[ ]:


playstore_df = pd.read_csv("../input/googleplaystore.csv")
playstore_df.head()


# In[ ]:


playstore_df.shape


#  The dataset contains 10841 rows and 13 columns

# In[ ]:


playstore_df.info()


# ># **Data Cleaning**

# ## 1. Removing Duplicate Values

# In[ ]:


playstore_df = playstore_df.drop_duplicates(['App'], keep='first')


# ## 2. Handling Missing Values

# In[ ]:


playstore_df.isnull().sum()


# In[ ]:


playstore_df.dropna(how='any', inplace=True)  # Dropping Missing Values


# * Convert values of Object type to int type
# * Remove  'M' from  Size of the App  and convert to KB
# * Remove '+' and ',' from number of Installs and convert it to int type
# * Remove '$' from Price and convert it to float type

# In[ ]:


# Converting to int
playstore_df['Reviews'] = playstore_df['Reviews'].apply(lambda x: int(x) if str(x).isnumeric() == True else x )
playstore_df['Reviews'] = playstore_df['Reviews'].astype(int)


# In[ ]:


#Size - Removing 'M' and mutltiplying by 10^3
playstore_df['Size'] = playstore_df['Size'].apply(lambda x: str(x).replace('Varies with device','NaN') if 'Varies with device' in x else x)
playstore_df['Size'] = playstore_df['Size'].apply(lambda x: float(str(x).rstrip('M'))*(10**3) if 'M' in str(x) else x)
playstore_df['Size'] = playstore_df['Size'].apply(lambda x: float(str(x).rstrip('k')) if 'k' in str(x) else x)
playstore_df = playstore_df[~(playstore_df['Size'] == 'NaN')]
playstore_df['Size'] = playstore_df['Size'].astype(int)


# In[ ]:


playstore_df['Installs'].unique()


# In[ ]:


# Removing '+' and ','
playstore_df['Installs'] = playstore_df['Installs'].str.rstrip('+').str.replace(',','')
playstore_df['Installs'] = playstore_df['Installs'].astype(int) #Converting to int


# In[ ]:


playstore_df['Price'].unique()


# In[ ]:


# Removing '$'
playstore_df['Price'] =  playstore_df['Price'].apply(lambda x : float(str(x).lstrip('$')) if '$' in str(x) else x)
playstore_df['Price'] = playstore_df['Price'].astype(float) #converting to float


# In[ ]:


playstore_df.head()


# >## **Categories**

# In[ ]:


playstore_df['Category'].nunique()


# ## Distribution of Apps Category-Wise

# In[ ]:


cat_df = playstore_df['Category'].value_counts().head(10).to_frame().reset_index()
data = cat_df.Category
recipe = cat_df['index']

fig, ax = plt.subplots(figsize=(10, 12), subplot_kw=dict(aspect="equal"))
wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)

plt.show()


# * **Family**, **Game**  and **Tools** have the highest presence on Playstore

# >## How are free and paid apps distributed category-wise?

# In[ ]:


playstore_df['Category'].value_counts().head(6).index.tolist()

Free = playstore_df[playstore_df.Type == 'Free']['Category'].value_counts().to_frame()
Paid = playstore_df[playstore_df.Type == 'Paid']['Category'].value_counts().to_frame()

Total = Free.join(Paid, lsuffix='_Free', rsuffix='_Paid')
Total = Total.head(6)
Total.reset_index()


# In[ ]:


N = 6
ind = np.arange(N)
w = 0.45
plt.figure(figsize=(14,8))
p1 = plt.bar(ind, Total.Category_Free.values, width=w, color='#DB5C5C')
p2 = plt.bar(ind, Total.Category_Paid.values, bottom=Total.Category_Free.values, width=w, color='#39A86B')

plt.xticks(ind, Total.index, rotation=35)
plt.legend((p1[0], p2[0]),('Free', 'Paid'))
plt.show()


# >## **How are apps distributed rating-wise?**

# In[ ]:


ratings = playstore_df['Rating']

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,12))

plt.subplot(2,1,1)
plt.hist(ratings, bins=None)
plt.title('Rating Distribution', fontsize=22, color='#191970')
plt.xlabel('Ratings', fontsize=14, color='#191970')

plt.subplot(2,1,2)
sns.set(style="whitegrid")
sns.stripplot(data=ratings, jitter=True, orient='h')


# ## Insights
# 
# * The graphs clearly show that most of the apps have an average rating between 4.0 and 4.5.
# * Very few apps have an averating rating below 2.5

# > # **Top 6 Categories in Playstore**

# In[ ]:


df_catinst = playstore_df[['Category', 'App']].groupby(['Category']).                          count().sort_values('App', ascending=0).head(6).reset_index()

plt.figure(figsize=(12,8))
sns.set_style("darkgrid")
ax=sns.barplot(x=df_catinst.Category, 
               y=df_catinst.App, 
               data=df_catinst, 
               palette='YlGnBu_r')

for p in ax.patches:
    ax.text(p.get_x()+0.3, p.get_height()+48, p.get_height(),             fontsize=11, color='black', rotation=15)

ax.set_xlabel('Categories',fontsize=16, fontweight='bold', color='#191970')
ax.set_ylabel('Number of Apps', fontsize=16, fontweight='bold', color='#191970')


#  >## **Which are the best performing categories?**

# In[ ]:


df_categeories = playstore_df.groupby('Category').filter(lambda x: len(x) >= 120)

plt.figure(figsize=(15,10))
sns.boxplot(y=df_categeories.Rating, x=df_categeories.Category,data=playstore_df);

plt.xticks(rotation=50)
ax.set_xlabel('Categories',fontsize=16, fontweight='bold', color='#191970', )
ax.set_ylabel('Ratings', fontsize=16, fontweight='bold', color='#191970')


# ## Insights
# * Most of the categories have performed moderately well.
# * **Health and Fitness**,having average rating 4.5, has the best quality apps. Next best performing categories are **Books and References** and **Personalization**.

# >## **Relationship bewteen the number of reviews and ratings of the apps**

# In[ ]:


df2 = playstore_df[['Rating', 'Reviews']].reset_index()

sns.lmplot(y='Reviews', x='Rating',data=df2,fit_reg=False, size=8)


# ## Insights
# * As the number of reviews increases, the average ratings of the apps get affected. 
# * Higher the number of reviews, more likely the apps will have an average rating above 4.0. 
# * There are exceptions too. The graph shows that some apps, which have recieved low number of reviews, have an average rating 5.0. 

# >## **Which categories have the highest number of installs?**

# In[ ]:


temp_df = playstore_df.groupby(['Category'])                  .agg({'Installs':'sum'})                  .sort_values(by='Installs',ascending=False).reset_index()

plt.figure(figsize=(15,20))
sns.set_style("darkgrid")

sns.barplot(x=temp_df['Installs'], 
            y=temp_df['Category'], 
            linewidth=2, 
            edgecolor="k"*len(temp_df), 
            palette="Blues_r" )
plt.yticks(rotation=10)

plt.xlabel('Installs', fontsize=15, color='#191970')
plt.ylabel('Categories', fontsize=15, color='#191970')


# ## Insights
# *  'Game' has the highest number of installs. One interesting fact is that  'Game' has the 2nd highest presence on Playstore. It suggests that majority of users prefer games over other apps.

# ># **Free and Paid Apps**

# ### Distribution of Ratings

# In[ ]:


categories = ['FAMILY', 'GAME', 'TOOLS']
    
df_temp_cat = playstore_df.loc[playstore_df.Category.isin(categories)]

plt.figure(figsize=(16,10))
sns.violinplot(x='Category', y='Rating', data=df_temp_cat,hue='Type',                   palette='Paired',split=True, scale='count',               kde=False,linewidth=1)


# >## **Does price depend on the size of the app?**

# In[ ]:


df_temp5 = playstore_df.loc[playstore_df['Type'] == 'Paid'][['Price', 'Size']].reset_index()

sns.lmplot(x='Price', y='Size', data=df_temp5, fit_reg=False ,size=10)


# ## Insights
# 
# * The answer is  **No**.

# ># **Best Performing Apps**
# 
#  Let's select some of the best performing apps. For the selection, we need to set some criteria. 
#  
# * Number of Installs should be equal to or greater than 1,000,000
# * Average Rating of an app should be equal to or greater than 3.5
# * Minimum number of Reviews should be 10,000

# In[ ]:


df_temp = playstore_df[(playstore_df.Installs >= 1000000)
           & (playstore_df.Rating >= 3.5)
           & (playstore_df.Reviews >=10000)][['App','Category','Rating','Size','Type','Installs']].sort_values('Installs', ascending=0)


# ## Best Performing Apps (Free)

# In[ ]:


df_freeapp = df_temp[(df_temp.Type == 'Free')][['App']].head(10)
df_freeapp


# ## Best Performing Apps (Paid)

# In[ ]:


df_paidapp = df_temp[(df_temp.Type == 'Paid')][['App']].head(10)
df_paidapp


# ## Top 10 Games 

# In[ ]:


a=df_temp[(df_temp.Category == 'GAME')&(df_temp.Type == 'Free')][['App']].head(10)
b=df_temp[(df_temp.Category == 'GAME')&(df_temp.Type == 'Paid')][['App']].head(10)

from IPython.display import display
print("Top 10 Games(Free)")
display(a)
print("Top 10 Games(Paid)")
display(b)

