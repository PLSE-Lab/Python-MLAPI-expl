#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("../input/17k-apple-app-store-strategy-games/appstore_games.csv")


# In[ ]:


df.head()


# In[ ]:


df.drop(["URL","ID","Subtitle","Icon URL","In-app Purchases","Description","Languages","Size","Genres","Original Release Date","Current Version Release Date"],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.columns=["name","average_rating","user_rating_count","price","developer","age_rating","genre"]


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


def impute_median(series):
    return series.fillna(series.median())


# In[ ]:


df.average_rating=df.average_rating.transform(impute_median)


# In[ ]:


df.user_rating_count=df.user_rating_count.transform(impute_median)


# In[ ]:


df.price=df.price.transform(impute_median)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.age_rating=df.age_rating.apply(lambda x: str(x).replace("+","") if '' in str(x) else str(x))
df["age_rating"]=df["age_rating"].apply(lambda x: str(x).replace('4','child') if '4' in str(x) else str(x))
df["age_rating"]=df["age_rating"].apply(lambda x: str(x).replace('9','child') if '9' in str(x) else str(x))
df["age_rating"]=df["age_rating"].apply(lambda x: str(x).replace('12','teenager') if '12' in str(x) else str(x))
df["age_rating"]=df["age_rating"].apply(lambda x: str(x).replace('17','teenager') if '17' in str(x) else str(x))


# In[ ]:


df = df.sort_values(by=["user_rating_count","genre"], ascending=False)
df['rank']=tuple(zip(df.user_rating_count,df.genre))
df['rank']=df.groupby('user_rating_count',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values
df.head()


# In[ ]:


df.reset_index(inplace=True,drop=True)
df.head()


# In[ ]:


df.drop(["rank"],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# In[ ]:


df.corr()


# In[ ]:


df.nunique()


# In[ ]:


df["age_rating"].value_counts()


# In[ ]:


sns.set(style='whitegrid')
ax=sns.barplot(x=df['age_rating'].value_counts().index,y=df['age_rating'].value_counts().values,palette="Blues_d",hue=['female','male'])
plt.legend(loc=8)
plt.xlabel('age_rating')
plt.ylabel('Frequency')
plt.title('Show of age_rating Bar Plot')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(x=df['average_rating'].value_counts().index,
              y=df['average_rating'].value_counts().values)
plt.xlabel('average_rating ')
plt.ylabel('Frequency')
plt.title('Show of average_rating  Bar Plot')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(x=df['price'].value_counts().index,
              y=df['price'].value_counts().values)
plt.xlabel('price ')
plt.ylabel('Frequency')
plt.title('Show of price  Bar Plot')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x = "genre", y = "average_rating", hue = "age_rating", data = df)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(df.user_rating_count[:750],df.genre[:750])
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(df.user_rating_count[:20],df.name[:20])
plt.show()


# In[ ]:


plt.figure(figsize=(18,18))
sns.catplot(y="age_rating", x="average_rating",
                 hue="genre",
                 data=df, kind="bar")
plt.show()


# In[ ]:


labels=df['genre'].value_counts().index
colors=['blue','red','yellow','green','brown']
explode=[0,0.9,0.9,0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
values=df['genre'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('genre According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


labels=df['age_rating'].value_counts().index
colors=['blue','red']
explode=[0,0.2]
values=df['age_rating'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('age_rating According Analysis',color='black',fontsize=10)
plt.show()


# In[ ]:


sns.kdeplot(df['average_rating'])
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Average Rating Score Kde Plot System Analysis')
plt.show()


# In[ ]:


plt.figure(figsize=(18,5))
sns.violinplot(x=df['genre'],y=df['average_rating'])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.boxenplot(x="genre", y="average_rating",
              color="b",
              scale="linear", data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.stripplot(x=df.genre,y=df.average_rating,data=df)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


ax = sns.distplot(df['average_rating'])
plt.show()


# In[ ]:


df[df['age_rating']=='child']['average_rating'].value_counts().sort_index().plot.line(color='b')
df[df['age_rating']=='teenager']['average_rating'].value_counts().sort_index().plot.line(color='r')
plt.xlabel('average_rating')
plt.ylabel('Frequency')
plt.title('average_rating vs age_rating')
plt.show()


# In[ ]:


sns.lineplot(x='average_rating',y='user_rating_count',hue='price',data=df)
plt.show()

