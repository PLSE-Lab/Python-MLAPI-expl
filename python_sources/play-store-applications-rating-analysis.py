#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


#handling missing data
totale=data.isnull().sum().sort_values(ascending=False)
percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([totale,percentage],axis=1,keys=['totale','percent'])
missing_data.head(6)


# In[ ]:


data.dropna(how='any',inplace=True)


# In[ ]:


totale=data.isnull().sum().sort_values(ascending=False)
percentage=(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([totale,percentage],axis=1,keys=['totale','percent'])
missing_data.head(6)


# ## Results:
after romove the missing data we get a dataset with 9360 rows and 13 columns
# ## Rating analysis

# In[ ]:


data.Rating.describe()


# In[ ]:


sns.set(style='darkgrid')
rcParams['figure.figsize']=11,9
g=sns.kdeplot(data.Rating,color='y',shade=True)
g.set_xlabel('Rating')
g.set_ylabel('Frequency')
plt.title("Distribution of Rating",size=30)


# ## Results:
Average of rating of application in store is around 4 which is very high
# ## Category

# In[ ]:


print('Dataset has a ', len(data['Category'].unique()) , "categories")

print("\n", data['Category'].unique())


# In[ ]:


## counting 
categories=data.groupby(['Category']).App.count().reset_index().sort_values(by='App',ascending=False)
a=sns.barplot(x='Category',y='App',data=categories,palette='nipy_spectral')
a.set_xticklabels(a.get_xticklabels(), rotation=90, ha="right")
plt.title('Count of app in each category',size=30)

As we can see Family ,Games and Tools category are the most  apperance for  App in store
# In[ ]:


a=sns.boxplot(x='Rating',y='Category',data=data,palette='magma')
plt.title('Raing of Apps in each category',size=25)

we can notice that Rating in each category is not diffrent too much
# ## Reviews

# In[ ]:


# converting Reviews's data type to numerical
data.Reviews=data.Reviews.astype(int)


# In[ ]:


data.Reviews.head()


# In[ ]:


# Reviews distribution
rcParams['figure.figsize']=11,9
a=sns.kdeplot(data.Reviews,color='y',shade=True)
a.set_xlabel('Reviews',size=17)
a.set_ylabel('Frequency',size=17)
plt.title('Distribution of Reviews',size=25)


# In[ ]:


data[data.Reviews<1000000].shape


# In[ ]:


data[data.Reviews>3000000].head()

Most of application in this store have less than 1M in reviews.
Obviously, well-known applictions have a lot of reviews
# In[ ]:


a=sns.jointplot('Reviews','Rating',data=data,size=9,color='y')


# In[ ]:


rcParams['figure.figsize']=11,9
a=sns.regplot(x='Reviews',y='Rating',data=data[data.Reviews<1000000],color='g')
plt.title('Rating vs Reviews',size=25)

seems like well_known appliacations will ger a good Rating 
# # Number of Installation

# In[ ]:


data.Installs.unique()


# In[ ]:


## It is preferable to encode it by numbs


# In[ ]:


data.Installs=data.Installs.replace(r'[\,\+]', '', regex=True).astype(int)


# In[ ]:


install_sorted=sorted(data.Installs.unique())


# In[ ]:


install_sorted


# In[ ]:


data.Installs.replace(install_sorted,range(0,len(install_sorted),1),inplace=True)


# In[ ]:


from scipy.stats import spearmanr
a=sns.jointplot(x='Installs',y='Rating',data=data,kind='kde',size=9,color='y',stat_func=spearmanr)


# In[ ]:


a=sns.regplot(x='Installs',y='Rating',data=data,color='pink')
plt.title('Relation Between Rating And Installs',size=20)

We can observe that installs affect Rating
# ## Size

# In[ ]:


data.Size.unique()


# In[ ]:


data.Size.replace('Varies with device',np.nan,inplace=True)


# In[ ]:


## Change size values to the same units
data.Size=(data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            \
            .replace(['k','M'], [10**3, 10**6]).astype(float))


# In[ ]:


## filling null values by the mean of each category
data.Size.fillna(data.groupby('Category')['Size'].transform('mean'),inplace=True)


# In[ ]:


data.Size


# ## Rating VS Size 

# In[ ]:


a=sns.jointplot(x='Size',y='Rating',data=data,color='y',size=9,kind='kde',stat_func=spearmanr)


# In[ ]:


a=sns.regplot(x='Size',y='Rating',data=data,color='black')
plt.title('Rating vs Size',size=25)


# ## Relation between type ans Rating

# In[ ]:


data.Type.unique()


# ## Counts Type

# In[ ]:


a=sns.countplot(x="Type",data=data)


# In[ ]:


percent=round(data.Type.value_counts(sort=True)/data.Type.count()*100,2).astype(str)+'%'
Type_values=pd.concat([data.Type.value_counts(sort=True),percent],axis=1,keys=['Totale','percent'])
Type_values

Most of store's application are for free
# we need to change the format a little bit which allows us use it in modeling after if we want 

# In[ ]:


type_dum=pd.get_dummies(data['Type'])
type_dum.drop(['Paid'],axis=1,inplace=True)
data=pd.concat([data,type_dum],axis=1)


# In[ ]:


# now we drop Type column
data.drop(['Type'],axis=1,inplace=True)


# ## Price

# In[ ]:


data.Price.unique()


# In[ ]:


data.Price=data.Price.apply(lambda x:float(x.replace('$','')))


# In[ ]:


data.Price.describe()

as we can see the average of price is 0.96$ but remembre that almost applaction are for free
# In[ ]:


a=sns.regplot(x='Price',y='Rating',data=data,color='m')
plt.title('Relation Between Price and Rating',size=25)


# In[ ]:


## let's us check little bit for more details
data[data.Price==0].shape


# In[ ]:


bins=[-1,0.98,1,3,5,16,30,401]
labels=['Free','Cheap','Not Cheap','Medium','Expensive','Very expensive','Extra expensive']
data['Price category']=pd.cut(data.Price,bins,labels=labels)


# In[ ]:


data.groupby(['Price category'], as_index=False)['Rating'].mean()


# In[ ]:


a=sns.catplot(x='Rating',y='Price category',data=data,kind='bar',height=10,palette='mako')
plt.title('Barplot of Rating\'s mean for each Price category',size=25)


# ## Results
We can understand immediately that Price will not affect to Rating except if it's too expensive and don't desrve too much
# ## Genres:

# In[ ]:


data.Genres.unique()


# In[ ]:


data.Genres.value_counts()

A lot of values let's see what we can do about it..
we can keep just the main Genre by drop sub-genre after (;)
# In[ ]:


data['Genres'] = data['Genres'].str.split(';').str[0]


# In[ ]:


data.Genres.value_counts()


# In[ ]:


## We can Group Music & Audio  as  Music
data['Genres'].replace('Music & Audio', 'Music',inplace = True)


# In[ ]:


data.groupby('Genres',as_index=False)['Rating'].mean().describe()


# In[ ]:


a=sns.catplot(x='Rating',y='Genres',data=data,kind='bar',height=10,palette='coolwarm')
plt.title('Barplot of Rating\'s mean for each Genre',size=25)


# ## Results
Observing from Standard Deviation and this plot seem like genre is not effect too much to rating.
# ## Centent Rating

# In[ ]:


data['Content Rating'].unique()


# In[ ]:


data['Content Rating'].value_counts()


# In[ ]:


plt.figure(figsize=(13,10))
a=sns.boxenplot(x='Content Rating',y='Rating',data=data,palette='Accent')
plt.title('boxen plot of Rating Vs Content Rating',size=25)

Rating of application in each Content Rate is not different too much 
# ## Thanks you for read my kernel waiting for your opinions !!!

# In[ ]:




