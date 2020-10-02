#!/usr/bin/env python
# coding: utf-8

# # Google Play Store Apps
# Google Play (previously Android Market) is a digital distribution service operated and developed by Google LLC. It serves as the official app store for the Android operating system, allowing users to browse and download applications developed with the Android software development kit (SDK) and published through Google. Google Play also serves as a digital media store, offering music, books, movies, and television programs. It previously offered Google hardware devices for purchase until the introduction of a separate online hardware retailer.
# 
# Applications are available through Google Play either free of charge or at a cost. They can be downloaded directly on an Android device through the Play Store mobile app or by deploying the application to a device from the Google Play website. Applications exploiting hardware capabilities of a device can be targeted to users of devices with specific hardware components, such as a motion sensor (for motion-dependent games) or a front-facing camera (for online video calling). The Google Play store had over 82 billion app downloads in 2016 and has reached over 3.5 million apps published in 2017.It has been the subject of multiple issues concerning security, in which malicious software has been approved and uploaded to the store and downloaded by users, with varying degrees of severity.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ps=pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


ps.head()


# In[ ]:


df=ps.copy()


# In[ ]:


df.head()


# In[ ]:


df.columns=df.columns.map(str.lower).str.replace(' ','_')


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.rating=df.rating.astype(str).replace('nan','0')


# In[ ]:


df.head(1)


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()


# In[ ]:


df.isnull().sum()


# In[ ]:


df[df.duplicated(['app','category','genres'])]


# In[ ]:


df=df.drop_duplicates(['app','category','genres'],keep='first')


# In[ ]:


df[df.duplicated(['app','category','genres'])]


# In[ ]:


orders=df.category.value_counts().index
plt.figure(figsize=(40,15))
ax=sns.countplot(x='category',data=df,order=orders)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=25)
plt.xlabel('Count',fontsize=25)
plt.ylabel('Category',fontsize=25)
plt.xticks(fontsize=25,rotation=90)
plt.yticks(fontsize=25)
plt.title('Category count',fontsize=25);


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot(x='type',data=df)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
                ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('App Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of App Types',fontsize=15);


# In[ ]:


plt.figure(figsize=(20,10))
orders=df.content_rating.value_counts().index
ax=sns.countplot('content_rating',data=df,order=orders)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xlabel('Content Rating',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of Content Ratings for Apps',fontsize=15);


# In[ ]:


plt.figure(figsize=(70,20))
orders=df.genres.value_counts().index
ax=sns.countplot('genres',data=df,order=orders[:60])
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
                ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=30)
plt.ylabel('Count',fontsize=35)
plt.xlabel('Genres',fontsize=35)
plt.xticks(rotation=90,fontsize=35)
plt.yticks(fontsize=35)
plt.title('Count of Content Ratings for Apps',fontsize=35);


# In[ ]:


data_types={'rating':float,'reviews':int}
df=df.astype(data_types)


# In[ ]:


plt.figure(figsize=(30,10))
cat=df.groupby('category')['reviews'].sum().sort_values(ascending=False).reset_index()
orders=cat['category'].value_counts().index
sns.barplot(x='category',y='reviews',data=cat)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Category',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Count of ratings according to the category',fontsize=20);


# In[ ]:


rating=df.groupby('category')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
orders=rating.sort_values('category')
sns.barplot(x='category',y='rating',data=rating)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Category',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Count of ratings according to the category',fontsize=20);


# In[ ]:


df.installs=df.installs.str.replace(',','')
df.installs=df.installs.str.replace('+','')
df.installs=df.installs.astype(int)


# In[ ]:


installs=df.groupby('category')['installs'].sum().reset_index()
plt.figure(figsize=(30,10))
orders=df.category.value_counts().index
sns.barplot(x='category',y='installs',data=installs,order=orders)
plt.xlabel('Category',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.title('Number of installs according to category',fontsize=20);


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('installs',data=df)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Installs',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title('Installation Count of google apps',fontsize=20);


# In[ ]:


df.rename(columns={'size':'size_app'},inplace=True)


# In[ ]:


df.size_app=df.size_app.str.replace('M','')
df.size_app=df.size_app.str.replace('k','')
sa=df[df.size_app!='Varies with device']
sa.size_app=sa.size_app.astype(float)


# In[ ]:


plt.figure(figsize=(20,10))
sap=sa.groupby('category')['size_app'].sum().sort_values(ascending=False).reset_index()
sns.barplot(x='category',y='size_app',data=sap)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Category',fontsize=15)
plt.ylabel('Size',fontsize=15)
plt.title('Total size of google apps according to the category',fontsize=20);


# In[ ]:


df_p=df.copy()
df_p.price=df_p.price.str.replace('$','').astype(float)


# In[ ]:


p_price=df_p.groupby('category')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(20,10))
sns.barplot(x='category',y='price',data=p_price)
plt.xlabel('Category',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Total Price according to the category in google apps',fontsize=15);


# In[ ]:


p_content=df_p.groupby('content_rating')['price'].sum().reset_index()
plt.figure(figsize=(20,10))
sns.barplot(x='content_rating',y='price',data=p_content)
plt.xlabel('Content Rating',fontsize=15)
plt.ylabel('Price',fontsize=15)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Total Price according to the Content Rating in google apps',fontsize=15);


# In[ ]:


p_content


# In[ ]:


content_price=df_p.groupby('content_rating')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(20,10))
sns.barplot(x='content_rating',y='price',data=content_price)
plt.ylabel('Content Rating',fontsize=15)
plt.xlabel('Price',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Total Price of content rating og google apps',fontsize=15);


# In[ ]:


df.head()


# In[ ]:


df.category.value_counts()


# In[ ]:


family=df[df.category=='FAMILY']
games=df[df.category=='GAME']
tools=df[df.category=='TOOLS']
business=df[df.category=='BUSINESS']
medical=df[df.category=='MEDICAL']
personal=df[df.category=='PERSONALIZATION']
product=df[df.category=='PRODUCTIVITY']
lifestyle=df[df.category=='LIFESTYLE']
finance=df[df.category=='FINANCE']
sport=df[df.category=='SPORTS']
communication=df[df.category=='COMMUNICATION']
health=df[df.category=='HEALTH_AND_FITNESS']
photo=df[df.category=='PHOTOGRAPHY']
news=df[df.category=='NEWS_AND_MAGAZINES']
social=df[df.category=='SOCIAL']
books=df[df.category=='BOOKS_AND_REFERENCE']
travel=df[df.category=='TRAVEL_AND_LOCAL']
shop=df[df.category=='SHOPPING']
dating=df[df.category=='DATING']
video=df[df.category=='VIDEO_PLAYERS']
maps=df[df.category=='MAPS_AND_NAVIGATION']
education=df[df.category=='EDUCATION']
food=df[df.category=='FOOD_AND_DRINK']
entertainment=df[df.category=='ENTERTAINMENT']
auto=df[df.category=='AUTO_AND_VEHICLES']
library=df[df.category=='LIBRARIES_AND_DEMO']
weather=df[df.category=='WEATHER']
house=df[df.category=='HOUSE_AND_HOME']
event=df[df.category=='EVENTS']
art=df[df.category=='ART_AND_DESIGN']
parent=df[df.category=='PARENTING']
comics=df[df.category=='COMICS']
beauty=df[df.category=='BEAUTY']


# In[ ]:


family.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=family,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Family category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=family)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Family Category',fontsize=20);


# In[ ]:


gener=family.groupby('genres')['rating'].sum().sort_values(ascending=False,kind='quicksort').reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='rating',data=gener[:30])
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in family category',fontsize=20);


# In[ ]:


reviews=family.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='reviews',data=reviews[:10])
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in family category',fontsize=20);


# In[ ]:


install=family.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='installs',data=install[:10])
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in family category',fontsize=20);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
family.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
family.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


fp=family.copy()
fp.price=fp.price.str.replace('$','').astype(float)
f=fp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=f,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Family category',fontsize=25);


# In[ ]:


games.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=games,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Game category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=games)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Family Category',fontsize=20);


# In[ ]:


gener=games.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='rating',data=gener[:30])
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Game category',fontsize=20);


# In[ ]:


reviews=games.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='reviews',data=reviews[:10])
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Game category',fontsize=20);


# In[ ]:


install=games.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='installs',data=install[:10])
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Game category',fontsize=20);


# In[ ]:


gp=games.copy()
gp.price=gp.price.str.replace('$','').astype(float)
g=gp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=g,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Family category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
games.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
games.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


tools.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=tools,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Tools category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=tools)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Tools Category',fontsize=20);


# In[ ]:


gener=tools.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Tools category',fontsize=20);


# In[ ]:


reviews=tools.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Tools category',fontsize=20);


# In[ ]:


install=tools.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Tools category',fontsize=20);


# In[ ]:


tp=tools.copy()
tp.price=tp.price.str.replace('$','').astype(float)
t=tp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=t,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Family category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
tools.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
tools.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


business.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=business,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Business category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=business)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Business Category',fontsize=20);


# In[ ]:


gener=business.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Business category',fontsize=20);


# In[ ]:


reviews=business.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Business category',fontsize=20);


# In[ ]:


install=business.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Business category',fontsize=20);


# In[ ]:


bp=business.copy()
bp.price=bp.price.str.replace('$','').astype(float)
b=bp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=b,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Family category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
business.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
business.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


medical.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=medical,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in medical category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=medical)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in medical Category',fontsize=20);


# In[ ]:


gener=medical.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Medical category',fontsize=20);


# In[ ]:


reviews=medical.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Medical category',fontsize=20);


# In[ ]:


install=medical.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Medical category',fontsize=20);


# In[ ]:


mp=medical.copy()
mp.price=mp.price.str.replace('$','').astype(float)
m=mp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=m,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Family category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
medical.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
medical.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


personal.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=personal,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Personalization category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=personal)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Personalization Category',fontsize=20);


# In[ ]:


gener=personal.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Personalization category',fontsize=20);


# In[ ]:


reviews=personal.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Personalization category',fontsize=20);


# In[ ]:


install=personal.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Personalization category',fontsize=20);


# In[ ]:


pp=personal.copy()
pp.price=pp.price.str.replace('$','').astype(float)
p=pp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Personalization category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
personal.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
personal.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


product.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=product,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in PRODUCTIVITY category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=product)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Productivity Category',fontsize=20);


# In[ ]:


gener=product.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Productivity category',fontsize=20);


# In[ ]:


reviews=product.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Productivity category',fontsize=20);


# In[ ]:


install=product.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Productivity category',fontsize=20);


# In[ ]:


pp=product.copy()
pp.price=pp.price.str.replace('$','').astype(float)
p=pp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Productivity category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
product.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
product.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


lifestyle.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=lifestyle,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Lifestyle category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=lifestyle)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Life Style Category',fontsize=20);


# In[ ]:


gener=lifestyle.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Life Style category',fontsize=20);


# In[ ]:


reviews=lifestyle.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Life Style category',fontsize=20);


# In[ ]:


install=lifestyle.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Life Style category',fontsize=20);


# In[ ]:


lsp=lifestyle.copy()
lsp.price=lsp.price.str.replace('$','').astype(float)
p=lsp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Life style category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
lifestyle.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
lifestyle.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


finance.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=finance,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Finance category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=finance)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Finance Category',fontsize=20);


# In[ ]:


gener=finance.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Finance category',fontsize=20);


# In[ ]:


reviews=finance.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Finance category',fontsize=20);


# In[ ]:


install=finance.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Finance category',fontsize=20);


# In[ ]:


fp=finance.copy()
fp.price=fp.price.str.replace('$','').astype(float)
p=fp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Finance category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
finance.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
finance.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


sport.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=sport,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Sports category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=sport)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Sports Category',fontsize=20);


# In[ ]:


gener=sport.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Sports category',fontsize=20);


# In[ ]:


reviews=sport.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Sports category',fontsize=20);


# In[ ]:


install=sport.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Sports category',fontsize=20);


# In[ ]:


sp=sport.copy()
sp.price=sp.price.str.replace('$','').astype(float)
p=sp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Sports category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
sport.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
sport.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


communication.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=communication,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Communication category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=communication)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Communication Category',fontsize=20);


# In[ ]:


gener=communication.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Communication category',fontsize=20);


# In[ ]:


reviews=communication.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Communication category',fontsize=20);


# In[ ]:


install=communication.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Communication category',fontsize=20);


# In[ ]:


cp=communication.copy()
cp.price=cp.price.str.replace('$','').astype(float)
p=cp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Communication category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
communication.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
communication.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


health.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=health,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Health & Fitness category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=health)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Health & Fitness Category',fontsize=20);


# In[ ]:


gener=health.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Health & Fitness category',fontsize=20);


# In[ ]:


reviews=health.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Health & Fitness category',fontsize=20);


# In[ ]:


install=health.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Health & Fitness category',fontsize=20);


# In[ ]:


hp=health.copy()
hp.price=hp.price.str.replace('$','').astype(float)
p=hp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Health & Fitness category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
health.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
health.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


photo.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=photo,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Photography category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=photo)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Photography Category',fontsize=20);


# In[ ]:


gener=photo.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Photography category',fontsize=20);


# In[ ]:


reviews=photo.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Photography category',fontsize=20);


# In[ ]:


install=photo.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Photography category',fontsize=20);


# In[ ]:


php=photo.copy()
php.price=php.price.str.replace('$','').astype(float)
p=php.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Photography category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
photo.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
photo.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


news.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=news,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in News and Magazine category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=news)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in News and Category Category',fontsize=20);


# In[ ]:


gener=news.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in News and Magazine category',fontsize=20);


# In[ ]:


reviews=news.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in News and Magazine category',fontsize=20);


# In[ ]:


install=news.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in News and Magazine category',fontsize=20);


# In[ ]:


np=news.copy()
np.price=np.price.str.replace('$','').astype(float)
p=np.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Photography category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
news.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
news.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


social.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=social,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Social category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=social)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Social Category',fontsize=20);


# In[ ]:


gener=social.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Social category',fontsize=20);


# In[ ]:


reviews=social.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Social category',fontsize=20);


# In[ ]:


install=social.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Social category',fontsize=20);


# In[ ]:


sp=social.copy()
sp.price=sp.price.str.replace('$','').astype(float)
p=sp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Social category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
social.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
social.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


books.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=books,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Books and Reference category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=books)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Books and Refernce Category',fontsize=20);


# In[ ]:


gener=books.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Books and Reference category',fontsize=20);


# In[ ]:


reviews=books.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Books and Reference category',fontsize=20);


# In[ ]:


install=books.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Books and Reference category',fontsize=20);


# In[ ]:


bp=books.copy()
bp.price=bp.price.str.replace('$','').astype(float)
p=bp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Books and Reference category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
books.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
books.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


travel.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=travel,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Travel and Local category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=travel)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Travel and Local Category',fontsize=20);


# In[ ]:


gener=travel.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Travel and Local category',fontsize=20);


# In[ ]:


reviews=travel.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Travel and Local category',fontsize=20);


# In[ ]:


install=travel.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Travel and Local category',fontsize=20);


# In[ ]:


tp=travel.copy()
tp.price=tp.price.str.replace('$','').astype(float)
p=tp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Travel and Local category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
travel.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
travel.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


shop.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=shop,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Shopping category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=shop)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Shopping Category',fontsize=20);


# In[ ]:


gener=shop.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Shopping category',fontsize=20);


# In[ ]:


reviews=shop.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Shopping category',fontsize=20);


# In[ ]:


install=shop.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Shopping category',fontsize=20);


# In[ ]:


sp=shop.copy()
sp.price=sp.price.str.replace('$','').astype(float)
p=sp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Shopping category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
shop.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
shop.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


dating.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=dating,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Dating category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=dating)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Dating Category',fontsize=20);


# In[ ]:


gener=dating.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Dating category',fontsize=20);


# In[ ]:


reviews=dating.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Dating category',fontsize=20);


# In[ ]:


install=dating.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Dating category',fontsize=20);


# In[ ]:


dp=dating.copy()
dp.price=dp.price.str.replace('$','').astype(float)
p=dp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Dating category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
dating.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
dating.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


video.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=video,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Video Player category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=video)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Video Player Category',fontsize=20);


# In[ ]:


gener=video.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Video Player category',fontsize=20);


# In[ ]:


reviews=video.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Video player category',fontsize=20);


# In[ ]:


install=video.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Video player category',fontsize=20);


# In[ ]:


vp=video.copy()
vp.price=vp.price.str.replace('$','').astype(float)
p=vp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Video Player category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
video.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
video.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


maps.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=maps,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Maps and Navigation category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=maps)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Maps and Navigation Category',fontsize=20);


# In[ ]:


gener=maps.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Maps and Navigation category',fontsize=20);


# In[ ]:


reviews=maps.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Maps and Navigation category',fontsize=20);


# In[ ]:


install=maps.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Maps and Navigation category',fontsize=20);


# In[ ]:


mp=maps.copy()
mp.price=mp.price.str.replace('$','').astype(float)
p=mp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Maps and Navigation category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
maps.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
maps.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


education.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=education,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Education category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=education)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Education Category',fontsize=20);


# In[ ]:


gener=education.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Education category',fontsize=20);


# In[ ]:


reviews=education.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Education category',fontsize=20);


# In[ ]:


install=education.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Education category',fontsize=20);


# In[ ]:


ep=education.copy()
ep.price=ep.price.str.replace('$','').astype(float)
p=ep.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Education category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
education.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
education.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


food.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=food,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Food and Drinks category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=food)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Food and Drinks Category',fontsize=20);


# In[ ]:


gener=food.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Food and Drinks category',fontsize=20);


# In[ ]:


reviews=food.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Food and Drinks category',fontsize=20);


# In[ ]:


install=food.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Food and Drinks category',fontsize=20);


# In[ ]:


fp=food.copy()
fp.price=fp.price.str.replace('$','').astype(float)
p=fp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Food and Drinks category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
food.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
food.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


entertainment.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=entertainment,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Entertainment category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=entertainment)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Entertainment Category',fontsize=20);


# In[ ]:


gener=entertainment.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Entertainment category',fontsize=20);


# In[ ]:


reviews=entertainment.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Entertainment category',fontsize=20);


# In[ ]:


install=entertainment.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Entertainment category',fontsize=20);


# In[ ]:


ep=entertainment.copy()
ep.price=ep.price.str.replace('$','').astype(float)
p=ep.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Entertainment category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
entertainment.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
entertainment.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


auto.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=auto,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Auto and Vehicle category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=auto)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Auto and Vehicle Category',fontsize=20);


# In[ ]:


gener=auto.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Auto and Vehicle category',fontsize=20);


# In[ ]:


reviews=auto.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Auto and Vehicle category',fontsize=20);


# In[ ]:


install=auto.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Auto and Vehicle category',fontsize=20);


# In[ ]:


ap=auto.copy()
ap.price=ap.price.str.replace('$','').astype(float)
p=ap.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Auto and Vehicle category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
auto.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
auto.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


library.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=library,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Libarary and Demo category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=library)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Library and Demo Category',fontsize=20);


# In[ ]:


gener=library.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Library and Demo category',fontsize=20);


# In[ ]:


reviews=library.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Library and Demo category',fontsize=20);


# In[ ]:


install=library.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Library and Demo category',fontsize=20);


# In[ ]:


lp=library.copy()
lp.price=lp.price.str.replace('$','').astype(float)
p=lp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Library and Demo category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
library.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
library.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


weather.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=weather,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Weather category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=weather)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Weather Category',fontsize=20);


# In[ ]:


gener=weather.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Weather category',fontsize=20);


# In[ ]:


reviews=weather.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Weather category',fontsize=20);


# In[ ]:


install=weather.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Weather category',fontsize=20);


# In[ ]:


wp=weather.copy()
wp.price=wp.price.str.replace('$','').astype(float)
p=wp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Weather category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
weather.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
weather.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


house.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=house,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in House and Home category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=house)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in House and Home Category',fontsize=20);


# In[ ]:


gener=house.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in House and Home category',fontsize=20);


# In[ ]:


reviews=house.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in House and Home category',fontsize=20);


# In[ ]:


install=house.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in House and Home category',fontsize=20);


# In[ ]:


hp=house.copy()
hp.price=hp.price.str.replace('$','').astype(float)
p=hp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in House and Home category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
house.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
house.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


event.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=event,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Events category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=event)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Events Category',fontsize=20);


# In[ ]:


gener=event.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Events category',fontsize=20);


# In[ ]:


reviews=event.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Events category',fontsize=20);


# In[ ]:


install=event.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Events category',fontsize=20);


# In[ ]:


ep=event.copy()
ep.price=ep.price.str.replace('$','').astype(float)
p=ep.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Events category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
event.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
event.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


art.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=art,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Arts and Design category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=art)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Arts and Design Category',fontsize=20);


# In[ ]:


gener=art.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Arts and Design category',fontsize=20);


# In[ ]:


reviews=art.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Arts and Design category',fontsize=20);


# In[ ]:


install=art.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Arts and Design category',fontsize=20);


# In[ ]:


arp=art.copy()
arp.price=arp.price.str.replace('$','').astype(float)
p=arp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Arts and Design category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
art.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
art.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


parent.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=parent,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Parenting category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=parent)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Parenting Category',fontsize=20);


# In[ ]:


gener=parent.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Parenting category',fontsize=20);


# In[ ]:


reviews=parent.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Parenting category',fontsize=20);


# In[ ]:


install=parent.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Parenting category',fontsize=20);


# In[ ]:


pap=parent.copy()
pap.price=pap.price.str.replace('$','').astype(float)
p=pap.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Parenting category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
parent.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
parent.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


comics.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=comics,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Comics category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=comics)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Comics Category',fontsize=20);


# In[ ]:


gener=comics.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Comics category',fontsize=20);


# In[ ]:


reviews=comics.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Comics category',fontsize=20);


# In[ ]:


install=comics.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Comics category',fontsize=20);


# In[ ]:


cp=comics.copy()
cp.price=cp.price.str.replace('$','').astype(float)
p=cp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Comics category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
comics.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
comics.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:


beauty.head()


# In[ ]:


plt.figure(figsize=(20,10))
ax=sns.countplot('type',data=beauty,palette='rainbow')
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=15)
plt.xlabel('Type',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Count of app Types in Beauty category',fontsize=15);


# In[ ]:


plt.figure(figsize=(30,10))
ax=sns.countplot('content_rating',data=beauty)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xlabel('Content rating',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Count of content rating in Beauty Category',fontsize=20);


# In[ ]:


gener=beauty.groupby('genres')['rating'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='rating',data=gener)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Rating',fontsize=20)
plt.title('Most rated genres in Beauty category',fontsize=20);


# In[ ]:


reviews=beauty.groupby('genres')['reviews'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='reviews',data=reviews)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Reviews',fontsize=20)
plt.title('Most Reviewed genres in Beauty category',fontsize=20);


# In[ ]:


install=beauty.groupby('genres')['installs'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
ax=sns.barplot(x='genres',y='installs',data=install)
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+p.get_width()/2,p.get_height()),
               ha='center',va='center',xytext=(0,10),textcoords='offset points',color='black',fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Genres',fontsize=20)
plt.ylabel('Installs',fontsize=20)
plt.title('Most Installed genres in Beauty category',fontsize=20);


# In[ ]:


btp=beauty.copy()
btp.price=btp.price.str.replace('$','').astype(float)
p=btp.groupby('genres')['price'].sum().sort_values(ascending=False).reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x='genres',y='price',data=p,palette='rainbow')
plt.xlabel('Genres',fontsize=25)
plt.ylabel('Price',fontsize=25)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.title('Price of each genres in Beauty category',fontsize=25);


# In[ ]:


plt.figure(figsize=(30,20))
plt.subplot(121)
beauty.groupby('content_rating')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Content rating according to genres',fontsize=20)

plt.subplot(122)
beauty.groupby('type')['genres'].nunique().plot.pie(autopct='%1.f%%',
                                                             wedgeprops={'linewidth':5,'edgecolor':'white'},
                                                             shadow=True,
                                                             fontsize=20)
plt.ylabel('')
circle=plt.Circle((0,0),0.7,color='white')
plt.gca().add_artist(circle)
plt.title('Google app type according to genres',fontsize=20);


# In[ ]:





# In[ ]:





# In[ ]:




