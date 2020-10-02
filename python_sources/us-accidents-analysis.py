#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Displaying the first five lines of  the dataset
df.head()


# In[ ]:


df=pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')


# **Finding the summary of the data like number of rows and columns etc**

# In[ ]:


print('Rows',df.shape[0])
print('Number of columns',df.shape[1])
print('Features or Column names',df.columns.tolist())
print('\n Missing values :',df.isnull().values.sum())
print('\n Unique values : \n',df.nunique())


# **Finding out the columns with categorical values  using df.select_dtypes() method **

# In[ ]:


df.select_dtypes(exclude=['int','float']).columns


# In[ ]:


df['Description'].head()


# **Displaying the catogerical values**

# In[ ]:


print(df['Source'].unique())
print(df['Description'].unique())
print(df['Timezone'].unique())
print(df['Amenity'].unique())
print(df['No_Exit'].unique())


# **Finding the correlations in the data**

# In[ ]:


#first lets print some columns
df.columns


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# **Drawing the heat maps** using .gcf() method 

# In[ ]:


fig=sns.heatmap(df[['TMC','Severity','Start_Lat','End_Lat','Distance(mi)'
                   ,'Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)','Visibility(mi)'
                   ,'Wind_Speed(mph)']].corr(),annot=True,cmap='RdYlGn',linewidths=0.2 ,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# **Finding the sources of the data**

# In[ ]:


fig ,ax = plt.subplots(1,2,figsize=(18,8))
df['Source'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Share of Sources')
ax[0].set_ylabel('Count')
sns.countplot('Source',data=df,ax=ax[1],order=df['Source'].value_counts().index)
ax[1].set_title('Count of Source')
plt.show()


# ** Severity of the accidents**

# In[ ]:


fig ,ax = plt.subplots(1,2,figsize=(18,8))
df['Side'].value_counts().plot.pie(explode=[0,0.1,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Side types')
ax[0].set_ylabel('Count')
sns.countplot('Side',data=df,ax=ax[1],order=df['Side'].value_counts().index)
ax[1].set_title('Count of Side')
plt.show()


# *As you can see most accidents occur on the right side i.e. drivers side*

# ** Side **
# There are three things mentioned regarding the side R, L and third one is Blank No idea for that.

# In[ ]:


df['Side'].unique()


# **Time Zone**

# **Accidents in different timezones**

# In[ ]:


df['Timezone'].unique()


# In[ ]:


fig ,ax = plt.subplots(1,2,figsize=(18,8))
df['Timezone'].value_counts().plot.pie(explode=[0,0,0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Accidents in different timezones')
ax[0].set_ylabel('Count')
sns.countplot('Timezone',data=df,ax=ax[1],order=df['Timezone'].value_counts().index)
ax[1].set_title('Accidents Count Based on Timezone')
plt.show()


# **Time taken to clear the traffice**

# In[ ]:


st = pd.to_datetime(df.Start_Time, format = '%Y-%m-%d %H:%M:%S')
end= pd.to_datetime(df.End_Time, format='%Y-%m-%d %H:%M:%S')


# In[ ]:


diff = (end-st)
top20 = diff.astype('timedelta64[m]').value_counts().nlargest(20)
print('top 20 accident durations correspond to {:.1f}% of the data'.format(top20.sum()*100/len(diff)))
(top20/top20.sum()).plot.bar(figsize=(8,8))
plt.title('Accident Duration [Minutes]')
plt.xlabel('Duration [minutes]')
plt.ylabel('Fraction')


# **Accident in different states**

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(15,8))
clr = ('blue','forestgreen','gold','red','purple','cadetblue','hotpink','orange','darksalmon','brown')
df['State'].value_counts()[0:10].sort_values().plot(kind='barh',color=clr,ax=ax[0])
ax[0].set_title('Top 10 Accident Prone States',size=20)
ax[0].set_xlabel('States', size=18)

count = df['State'].value_counts()
groups=list(df['State'].value_counts().index)[:10]
counts=list(count[:10])
counts.append(count.agg(sum)-count[:10].agg('sum'))

groups.append('Other')
type_dict=pd.DataFrame({'group':groups , 'counts':counts})
clr1=('brown','darksalmon','orange','hotpink','cadetblue','purple','red','gold','forestgreen','blue','plum')
qx=type_dict.plot(kind='pie',y='counts',labels=groups,colors=clr1,autopct='%1.1f%%'
                 , pctdistance=0.9,radius=1.2,ax=ax[1])
plt.legend(loc=0 , bbox_to_anchor=(1.15,0.4))
plt.subplots_adjust(wspace = 0.6 ,hspace=0)
plt.ioff()
plt.ylabel( '')


# **STATE SPECIFIC ANALYSIS **

# In[ ]:


df_top_Severity_State = df.groupby('State').agg({'Severity':'mean'}).sort_values('Severity').reset_index()


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(y='Severity', x='State',data=df_top_Severity_State.head(10))
plt.ioff()


# **LOOKING AT THE WEATHER CONDITIONS **

# In[ ]:


plt.figure(figsize=(14,8))
df.groupby('Weather_Condition')      .size()    .sort_values(ascending =False)     .iloc[:5]     .plot.pie(explode=[0,0,0.1,0,0],autopct='%1.1f%%',shadow=True)
plt.ioff()


# **Top Weather Conditions for accidents**

# In[ ]:


fig ,ax =plt.subplots(figsize=(16,7))
df['Weather_Condition'].value_counts().sort_values(ascending=False).head(5).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('5 Top weather conditions for accidents ',fontsize=25)
plt.grid()
plt.ioff()


# * It says that people drive more carefully when the weather is bad and most accidents occur in the clear weather due to carelessness*

# **PLACES WHERE ACCIDENTS OCCUR MOST**

# In[ ]:


bool_cols=[col for col in df.columns if df[col].dtype == np.dtype('bool')]
booldf=df[bool_cols]
not_one_hot=booldf[booldf.sum(axis=1) > 1]
print('There are {} non one hot metadata rows , which are {:.1f}% of the data'.format(len(not_one_hot),100*len(not_one_hot)/len(df)))


# In[ ]:


bools = booldf.sum(axis=0)
bools


# In[ ]:


bools.plot.pie(autopct='%1.1f%%',shadow=True,figsize=(10,10))
plt.ylabel(" ")
plt.title('Proximity to the Traffic Object')


# *As one can see that most accidents occur near traffic signal, junction and crossings*

# In[ ]:


df['time'] = pd.to_datetime(df.Start_Time , format = '%Y-%m-%d %H:%M:%S')
df= df.set_index('time')
df.head()


# In[ ]:


freq_text = {'D':'Daily','W':'Weekly','Y':'Yearly'}
plt.subplots(1,3,figsize=(21,7))
for i, (fr,text) in enumerate(freq_text.items(),1):
    plt.subplot(1,3,i)
    sample = df.ID['2016':].resample(fr).count()
    sample.plot(style='.')
    plt.title('Accidents, {} count'.format(text))
    plt.xlabel('Date')
    plt.ylabel('Accident Count');


# In[ ]:


df['Start_Time'] = pd.to_datetime(df['Start_Time'], format="%Y/%m/%d %H:%M:%S")
df['DayOfWeekNum'] = df['Start_Time'].dt.dayofweek
df['DayOfWeek'] = df['Start_Time'].dt.weekday_name
df['MonthDayNum'] = df['Start_Time'].dt.day
df['HourOfDay'] = df['Start_Time'].dt.hour
fig, ax=plt.subplots(figsize=(16,7))
df['DayOfWeek'].value_counts(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Day of the Week',fontsize=20)
plt.ylabel('Number of accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Accident on Different Days of Week',fontsize=25)
plt.grid()
plt.ioff()


# **Keywords Used in the description**

# In[ ]:


from wordcloud import WordCloud
plt.style.use('seaborn')
wrds1 = df['Description'].str.split("(").str[0].value_counts().keys()
wc1=WordCloud(scale=5 , max_words=1000,colormap='rainbow',background_color='black').generate(" ".join(wrds1))
plt.figure(figsize=(20,14))
plt.imshow(wc1 , interpolation='bilinear')
plt.axis('off')
plt.title('Key words in Accident Description', color='b')
plt.show()


# **Factor plot**

# In[ ]:


sns.factorplot('State', 'Severity',data=df)
fig=plt.gcf()
fig.set_size_inches(20,7)
plt.show()


# In[ ]:


plt.figure(figsize=(14,8))
sub_6=df[df.Severity<5]
viz_4=sub_6.plot(kind='scatter', x='Start_Lng',y='Start_Lat',label='Severity',c='Severity',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))
viz_4.legend()
plt.ioff()

