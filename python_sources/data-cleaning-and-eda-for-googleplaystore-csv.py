#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# The data frame has maximum null values in the rating column, followed by some anamolies in type, content rating, genres and versions. The process flow would be following:
# 
# - Go one by one to each column and try understanding the data set
# -- Start from Category -> Type -> Review -> Installs -> Price -> Size -> Last Updated -> Version -> Rating
# 

# ### Category

# In[ ]:


df['Category'].value_counts()


# In[ ]:


#One odd value of 1.9


# In[ ]:


df[df['Category']=='1.9']


# In[ ]:


df['Type'].value_counts()


# In[ ]:


df[df['Type']=='0']


# In[ ]:


df['Installs'].value_counts()


# In[ ]:


#This row has many odd values in many columns such as categories, installs, type and other null values too. It is better to drop this row
df.drop(index=10472,inplace=True)


# In[ ]:


sns.countplot('Category',data=df,order=df['Category'].value_counts().index)
plt.xticks(rotation= 'vertical')
plt.show()


# Count of apps in the playstore are ordered as:
# - Family
# - Game
# - Tools
# - Medical
# - Busuness
# - and so on..
# - we will deep dive later into these after Data Cleaning

# In[ ]:


df.isnull().sum()


# ### Type

# In[ ]:


df[df['Type'].isnull()]


# In[ ]:


df['Type'].value_counts()


# In[ ]:


#Assuming that if price is 0 then the type should be Free. I'd rename the type for this entry to Free
#df[['Price','Type']]


# In[ ]:


df['Type'][9148]='Free'


# In[ ]:


df.isnull().sum()


# ### Cleaning Data Types- Review

# In[ ]:


df.dtypes


# In[ ]:


#converting reviews into integer
df['Reviews']=df['Reviews'].astype('int')


# In[ ]:


sns.jointplot('Reviews','Rating',data=df)


# ### Installs

# In[ ]:


df['Installs'].value_counts()


# In[ ]:


#removing the commas and + from installs
df['Installs']=df['Installs'].str.replace('+','')
df['Installs']=df['Installs'].str.replace(',','')


# In[ ]:


#converting object into integer
df['Installs']=df['Installs'].astype(int)


# In[ ]:


sns.jointplot('Installs','Rating',data=df)


# ### Price

# In[ ]:


df['Price$']=df['Price'].str.replace('$','').astype('float')


# In[ ]:


sns.distplot(df['Price$'],kde=True,hist=False,rug=True)
#Rug plot is analogous to a histogram with zero-width bins, or a one-dimensional scatter plot. 


# In[ ]:


df['Price$'].hist()
#Few apps are extremely expensive which is kind of odd, will have to look moew into it


# In[ ]:


#number of categories and genres
print(df['Category'].nunique(), df['Genres'].nunique())
#currently we are keeping both


# ### Cleaning the size column
# - Uniform Units in KiloBytes
# - Varies with device being changed to category wise mean

# In[ ]:


#df['Size'].str.replace('M','') - Cannot do this because then won't be able to bring uniformity to data
df['SizeM']=df[df['Size'].str.contains('M')]['Size'].str.replace('M','').astype(float)*1000 #changing in kilobytes


# In[ ]:


df['SizeM'].fillna(df['Size'],inplace=True)


# In[ ]:


df['Sizek']=df['SizeM'].str.replace('k','')


# In[ ]:


df['Sizek'].fillna(df['SizeM'],inplace=True)


# In[ ]:


df['Sizek']=pd.to_numeric(df['Sizek'], errors='coerce') 
#change the size column to numeric and varies with device would change to null values


# In[ ]:


#Assuming that majorly apps in the same category would lie in similar size (mb) range
df['Sizek']=df.groupby('Category')['Sizek'].transform(lambda x: x.fillna(x.mean())) #fill null values with category wise mean 


# In[ ]:


df['Sizek'].hist() 


# In[ ]:


df.isnull().sum()


# ### Last Updated

# In[ ]:


#Last Updated Format Changed
df['Last Updated']=pd.to_datetime(df['Last Updated'],infer_datetime_format=True)


# ### Working on Current Version and Android Version

# In[ ]:


df.groupby(df['Last Updated'].dt.year)['Android Ver'].size()


# In[ ]:


df['Android Ver'].value_counts()


# In[ ]:


#renaming 4.4W and up to cleaner version
df['Android Ver']=df['Android Ver'].str.replace('4.4W and up','4.4 and up')


# In[ ]:


yrtover=pd.crosstab(index=df['Last Updated'].dt.year, columns=df['Android Ver'])


# In[ ]:


pd.set_option('max_columns', 100)


# In[ ]:


yrtover.head(10)


# In[ ]:


yrtover.plot(kind="barh", figsize=(15,15),stacked=True)
plt.legend(bbox_to_anchor=(1.0,1.0))


# In[ ]:


df[df['Android Ver'].isnull()]


# ### Taking a look at the above insights-
# - I can deduce that the two apps would've Android requirment of 4.0 and up
# - All the apps updated in 2018 majorly fell into the category
# - However, it is kind of difficult to solve the Version problem because that'd mean taking all lot of Assumptions
# - These assumptions would be true for rows which say Varies with device. However, the above insights can be used to drive those assumptions

# ### Rating
# - My assumption would be that number of installs would be a good correlation for rating. However the plot below shows that thought higher installs lead to higher rating but lower number of installs may have higer rating too
# - Eventually for my EDA i'd drop the na values for rating

# In[ ]:


sns.jointplot('Installs','Rating',data=df)


# In[ ]:


df['Rating'].hist(bins=20)


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isnull().sum()


# ### More Exploratory Data Analysis

# In[ ]:


columns_to_use=['Category','Rating','Reviews','Sizek','Installs','Type','Price$','Content Rating','Genres','Last Updated']
#Currently for EDA these would provide more value to understand the rating of an app
#These features may have a better correlation with rating
#This hypothesis can be confirmed through EDA


# In[ ]:


new_df=df[columns_to_use]


# In[ ]:


new_df.head(4)


# In[ ]:


sns.countplot('Type',data=new_df)


# In[ ]:


new_df['Content Rating'].value_counts()


# In[ ]:


sns.countplot('Content Rating',data=new_df)
plt.xticks(rotation= 'vertical')
plt.show()


# In[ ]:


sns.pairplot(data=new_df,hue='Type')


# In[ ]:


sns.countplot('Category',data=new_df,order=new_df['Category'].value_counts().index)
plt.xticks(rotation= 'vertical')
plt.show()


# #### Let's compare the top 5 categories on different parameters

# In[ ]:


top5=new_df[new_df['Category'].isin(['FAMILY','GAME','TOOLS','PRODUCTIVITY','MEDICAL'])]


# In[ ]:


sns.barplot('Category','Installs',hue='Type',data=top5)


# - This shows that even though Family has large number of applications, but the number of installations for productivity and game is more
# Let's look at all the installations category wise

# In[ ]:


sns.barplot('Category','Installs',data=new_df,order=new_df.groupby('Category')['Installs'].mean().sort_values(ascending=False).index)
plt.xticks(rotation= 'vertical')
plt.show()


# The order for **mean** number of installations is:
# 
# - Communication
# - Social
# - Prodcutivity
# - Video Players
# - News & Magazine
# - Game
# 
# The order for installations calculated below:
# 
# - Game
# - Communication
# - Productivity
# - Social
# - Tools

# In[ ]:


pd.pivot_table(data=new_df, index='Category',values='Installs',aggfunc=np.sum,).sort_values(by='Installs',ascending=False)


# In[ ]:


#Finally Looking at the correlation
sns.heatmap(new_df.corr(),annot=True,fmt='0.2f')


# Installs and Reviews seem very correlated 

# In[ ]:


sns.lmplot('Installs','Reviews',data=new_df,x_jitter=True)
#Though installs is a numerical variable it seems more appropriate to plot it as categorical


# In[ ]:


sns.boxplot('Installs','Reviews',data=new_df)
plt.xticks(rotation='vertical')
plt.show()


# This plot definitely shows that increase in number of installations means more reviews which naturally makes sense because as the number of people using the app increases the reviews would also increase 

# In[ ]:




