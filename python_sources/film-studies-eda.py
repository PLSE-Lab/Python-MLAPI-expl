#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


old_data = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


old_data['release_date'] = pd.to_datetime(old_data['release_date']).apply(lambda x: x.date())
old_data['title_year'] = pd.to_datetime(old_data['release_date']).apply(lambda x: x.year)
old_data['title_year']=old_data['title_year'].values.astype(int)


# In[ ]:


data=old_data.copy()


# In[ ]:


genres_list = []
for i in range(0,4802):
    if old_data.genres[i][8:10] == '10':
        first_type_of_film = old_data.genres[i][8:13]
    elif old_data.genres[i][8:10] == '87':
        first_type_of_film = old_data.genres[i][8:11]
    else:
        first_type_of_film = old_data.genres[i][8:10]
    genres_list.append(first_type_of_film)
df=pd.DataFrame(genres_list)
data['genres_no'] = df
data['genres_no'].replace('','13',inplace=True)
data['genres_no'].dropna(inplace=True)
data['genres_no'].unique()


# In[ ]:


genres_name = []
for i in range(0,4802):
    if len(data['genres_no'][i]) == 3:
        type_name_of_film = data.genres[i][22:26]
    elif len(data['genres_no'][i]) == 5:
        type_name_of_film = data.genres[i][24:27]
    else:
        type_name_of_film = data.genres[i][21:24]
    genres_name.append(type_name_of_film)
df1=pd.DataFrame(genres_name)
data['genres_name'] = df1
data['genres_name'].replace('','others',inplace=True)
data['genres_name'].dropna(inplace=True)
data['genres_name'].value_counts()


# In[ ]:


production_countries_list = []
for i in range(0,4802):
    production_countries_name = old_data.production_countries[i][17:19]
    production_countries_list.append(production_countries_name)
df2=pd.DataFrame(production_countries_list)
data['production_countries_name'] = df2
data['production_countries_name'].replace('','others',inplace=True)
data['production_countries_name'].dropna(inplace=True)
data['production_countries_name'].unique()


# In[ ]:


spoken_languages_list = []
for i in range(0,4802):
    spoken_languages_str = old_data.spoken_languages[i][16:18]
    spoken_languages_list.append(spoken_languages_str)
df3=pd.DataFrame(spoken_languages_list)
data['spoken_languages_str'] = df3
data['spoken_languages_str'].dropna(inplace=True)
data['spoken_languages_str'].replace('','others',inplace=True)
data['spoken_languages_str'].unique()


# In[ ]:


data['profit']=data['revenue']-data['budget']


# In[ ]:


data['runtime'].dropna(inplace=True)


# In[ ]:


data['runtime_rate']=data['runtime']/data['runtime'].max()
data['vote_average_rate']=data['vote_average']/data['vote_average'].max()


# In[ ]:


data['profit_rate']=data['profit']/data['budget']
data['profit_rate'].dropna(inplace=True)
data['profit_rate'].replace('',0.1,inplace=True)


# In[ ]:


data['vote_average'].describe()


# In[ ]:


rate_category_list=[]
for each in data['vote_average']:
    if each<1:
        rate_of_category = "Ridiculous"
    elif each>=1 and each<2:
        rate_of_category = "Awful"
    elif each>=2 and each<3:
        rate_of_category = "Bad"
    elif each>=3 and each<4:
        rate_of_category = "Might"
    elif each>=4 and each<5:
        rate_of_category = "Average"
    elif each>=5 and each<6:
        rate_of_category = "Good"
    elif each>=6 and each<7:
        rate_of_category = "Good+"
    elif each>=7 and each<8:
        rate_of_category = "Very Good"
    elif each>=8 and each<9:
        rate_of_category = "Excellent"
    else:
        rate_of_category = "Amazing"
    rate_category_list.append(rate_of_category)
category_df=pd.DataFrame(rate_category_list)
data['category_of_rate'] = category_df
data['category_of_rate'].unique()


# In[ ]:


data.info()


# In[ ]:


data.head()


# In[ ]:


missing_data = data.isnull().sum(axis=0).reset_index()
missing_data.columns = ['column_name', 'missing_count']
missing_data['filling_factor'] = (data.shape[0] 
                                - missing_data['missing_count']) / data.shape[0] * 100
missing_data.sort_values('filling_factor').reset_index(drop = True)


# In[ ]:


f,ax = plt.subplots(figsize=(15, 9))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


count_df=pd.DataFrame(data['genres_name'].value_counts())
count_df.columns=['Total Amount of Genres Type']
sorted_count=count_df.sort_values('Total Amount of Genres Type',ascending=False)
sorted_count.head()


# In[ ]:


f, ax= plt.subplots(figsize=(12,5))
sns.barplot(x=sorted_count.index,y=sorted_count['Total Amount of Genres Type'])
plt.title("Main Genres",color = 'blue',fontsize=15)
plt.xlabel('Genres Types')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


category_financial = data[['budget','revenue','profit']].groupby(data['genres_name'])
financial_df=pd.DataFrame(category_financial.sum())
sorted_profit=financial_df.sort_values('profit',ascending=False)
sorted_profit.head()


# In[ ]:


f,ax = plt.subplots(figsize = (15,7))
sns.barplot(x=sorted_profit.index, y=sorted_profit['budget'],color='magenta',alpha = 0.8,label='Budget')
sns.barplot(x=sorted_profit.index, y=sorted_profit['revenue'],color='yellow',alpha = 0.4,label='Revenue')
sns.barplot(x=sorted_profit.index, y=sorted_profit['profit'],color='green',alpha = 0.5,label='Profit')
plt.xticks(rotation= 45)
ax.legend(loc='upper right',frameon = True)
ax.set(xlabel='Types of Genres', ylabel='Amount($)(times 10^11)',title = "Comparing Amounts for Each Category ")
plt.show()


# In[ ]:


labels = data['production_countries_name'].value_counts().head().index
colors = ['orange','red','magenta','yellow','green']
explode = [0.1,0.1,0.1,0.1,0.1]
sizes = data['production_countries_name'].value_counts().head().values

plt.figure(figsize = (8,8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Production Countries Distribution',color = 'blue',fontsize = 15)
plt.show()


# In[ ]:


data['spoken_languages_str'].unique()
data['spoken_languages_str'].replace('','others',inplace=True)


# In[ ]:


labels = data['spoken_languages_str'].value_counts().head().index
colors = ['green','grey','pink','orange','yellow']
explode = [0.1,0.5,0.5,0.5,0.5]
sizes = data['spoken_languages_str'].value_counts().head().values

plt.figure(figsize = (8,8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Spoken Languages Distribution',color = 'blue',fontsize = 15)
plt.show()


# In[ ]:


movies_after_1980 = data['title_year'][data['title_year']>=1980]
movies_after_1980_df=pd.DataFrame(movies_after_1980)
plt.figure(figsize=(15,7))
sns.countplot(movies_after_1980_df.title_year)
plt.xticks(rotation=90)
plt.title("Film Quantities",color = 'blue',fontsize=15)
plt.show()


# In[ ]:


years_financial = data[['budget','revenue','profit']].groupby(data['title_year'][data['title_year']>=1980])
financial_df2=pd.DataFrame(years_financial.sum())
sorted_profit2=financial_df2.sort_values('profit',ascending=False)
sorted_profit2.head()


# In[ ]:


f,ax = plt.subplots(figsize = (15,7))
sns.barplot(x=sorted_profit2.index.astype(int), y=sorted_profit2['budget'],color='magenta',alpha = 0.8,label='Budget')
sns.barplot(x=sorted_profit2.index.astype(int), y=sorted_profit2['revenue'],color='yellow',alpha = 0.4,label='Revenue')
sns.barplot(x=sorted_profit2.index.astype(int), y=sorted_profit2['profit'],color='green',alpha = 0.5,label='Profit')
plt.xticks(rotation= 90)
ax.legend(loc='upper left',frameon = True)
ax.set(xlabel='Years', ylabel='Amount($)(times 10^11)',title = "Comparing Amounts for Each Year ")
plt.show()


# In[ ]:


rates_of_data = data[['runtime_rate','vote_average_rate']].groupby(data['genres_name'])
rof_df=pd.DataFrame(rates_of_data.sum())
sorted_rof=rof_df.sort_values('vote_average_rate',ascending=False)
sorted_rof['runtime_s']=sorted_rof['runtime_rate']/sorted_rof['runtime_rate'].max()
sorted_rof['vote_s']=sorted_rof['vote_average_rate']/sorted_rof['vote_average_rate'].max()
sorted_rof.head()


# In[ ]:


f,ax1 = plt.subplots(figsize =(10,5))
sns.pointplot(x=sorted_rof.index,y='runtime_s',data=sorted_rof,color='lime',alpha=0.8)
sns.pointplot(x=sorted_rof.index,y='vote_s',data=sorted_rof,color='red',alpha=0.8)
plt.text(10,0.6,'Vote Ratio',color='red',fontsize = 17,style = 'italic')
plt.text(10,0.55,'Runtime Ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Genres Types',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Vote  VS  Runtime',fontsize = 20,color='blue')
plt.grid()


# In[ ]:


financial_df2['title_year']=financial_df2.index
movies_after_1995 = data['title_year'][data['title_year']>=1995].value_counts()
budget_after_1995 = financial_df2['budget'][financial_df2['title_year']>=1995]


# In[ ]:


movies_after_1995.name='Film Quantities'


# In[ ]:


g = sns.jointplot(movies_after_1995, budget_after_1995, kind="kde", size=7)
plt.show()


# In[ ]:


x = pd.concat([budget_after_1995,movies_after_1995],axis=1)
x.columns=['budget','Film Quantities']
g = sns.jointplot('budget', 'Film Quantities', data=x,size=5, ratio=3, color="r")
plt.show()


# In[ ]:


sns.lmplot(x='budget', y='Film Quantities', data=x)
plt.show()


# In[ ]:


filt_df = data[['category_of_rate','runtime_rate','title_year']]
filt_df1=filt_df[np.logical_or(filt_df['title_year']==2014, filt_df['title_year']==2015 )]
filt_df1.head()


# In[ ]:


sns.swarmplot(x="title_year", y="runtime_rate",hue="category_of_rate", data=filt_df1)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[ ]:


f,ax=plt.subplots(figsize=(20,7))
sns.boxplot(x="category_of_rate", y="runtime_rate", hue="title_year", data=filt_df1, palette="PRGn")
plt.show()


# In[ ]:


filt_df2 = data[['vote_average_rate','runtime_rate','category_of_rate']]
filt_df2.head()


# In[ ]:


pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=filt_df2, palette=pal, inner="points")
plt.show()


# In[ ]:


filt_df2['vote_average_rate'].dropna(inplace=True)
filt_df2['runtime_rate'].dropna(inplace=True)


# In[ ]:


sns.pairplot(filt_df2)
plt.show()


# In[ ]:





# In[ ]:




