#!/usr/bin/env python
# coding: utf-8

# ## I need to answer several questions. The answers to those questions must be supported by data and analytics. These are the questions:
# 
# ### 1. Which type of complaint should the Department of Housing Preservation and Development of New York City focus on first?
# ### 2. Should the Department of Housing Preservation and Development of New York City focus on any particular set of boroughs, ZIP codes, or street (where the complaints are severe) for the specific type of complaints you identified in response to Question 1?
# ### 3. Does the Complaint Type that you identified in response to question 1 have an obvious relationship with any particular characteristic or characteristics of the houses or buildings?
# ### 4. Can a predictive model be built for a future prediction of the possibility of complaints of the type that you have identified in response to question 1?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Import Data

# In[ ]:


df_pluto = pd.read_csv('/kaggle/input/nyc-pluto/pluto_20v2.csv')
df= pd.read_csv('/kaggle/input/nyc-311-hpd-calls/311_Service_Requests_from_2010_to_Present.csv')
print('Shape of NYC 311 Dataframe is ',df.shape)
print('Shape of PLUTO Dataframe is ',df_pluto.shape)
df.head()


# In[ ]:


df.columns


# # Data Wrangling

# ### Let's get rid of all the unnecessary fields

# In[ ]:


df_pluto=df_pluto[['address','bldgarea','bldgdepth','builtfar','commfar','facilfar','lot','lotarea',
                   'lotdepth','numbldgs','numfloors','officearea','resarea','residfar','retailarea',
                   'yearbuilt','yearalter1','zipcode','ycoord','xcoord']]

df=df[['Unique Key', 'Created Date', 'Closed Date',
       'Complaint Type', 'Descriptor', 'Location Type', 'Incident Zip',
       'Incident Address', 'Street Name','Address Type',
       'City', 'Status', 'Due Date',
       'Resolution Description','Borough',
       'Latitude', 'Longitude']]


print('Shape of PLUTO dataframe is ',df_pluto.shape)
print('Shape of the NYC 311 call dataframe is', df.shape)


# In[ ]:



df.head()


# In[ ]:


df[['Address Type']].describe()


# ### The field "Address Type" seems to have only one value. It's not useful information. We will LET IT GO

# In[ ]:


df=df.drop(columns=['Address Type'])
df_comp=df.groupby('Complaint Type')[["Unique Key"]].count()
df_comp=df_comp[df_comp['Unique Key']>80000].sort_values(by='Unique Key')
df_comp.columns=['No of complaints']


df_comp.plot(kind='barh',figsize=(10,8))
plt.xlabel('Number of Complains  $x10^6$', fontsize=14)
plt.ylabel('Complaint Type',fontsize=14)
plt.show()


# ***It seems like the highest complaints are for HEAT or HOT WATER!!***

# We are only going to focus on the most frequent occuring problems

# In[ ]:


print(df.shape)
df=df[df['Complaint Type'].isin(df_comp.index)]
print(df.shape)


# Which borough had the largest number of complaints?

# In[ ]:


df_bor= df.groupby('Borough')[['Unique Key']].count().sort_values('Unique Key',ascending=True)
df_bor.plot(kind='barh',figsize=(15,10))
plt.xlabel('Number of Complains  $x10^6$', fontsize=14)
plt.ylabel('Borough',fontsize=14)
plt.show()


# It seems like Brooklyn has the highest number of complaints. But BRONX is also very close. There are also a lot of entries with unspecified boroughs. We will have to find what borough those zip numbers belong to.

# We are creating a dataframe that will have all the Zipcode towards the boroughs they were assigned to the most. That should be the correct borough for the zipcode

# In[ ]:


df_zip=df.groupby('Incident Zip')[['Borough']].agg(lambda x:x.value_counts().index[0])
df_zip.head()


# We are going to replace the entries with unspecified boroughs with the borough their Zipcode belongs to. We are using df.at instead of df.loc because this is faster and finishes in a reasonable amount of time.

# In[ ]:


for i,j in zip(df[df['Borough']=='Unspecified'].index,df[df['Borough']=='Unspecified']['Incident Zip']):
    if np.isnan(j):
        continue
    df.at[i,'Borough']=df_zip.at[j,'Borough']
    #print(type(j))
    
df.groupby('Borough')[['Unique Key']].count().sort_values('Unique Key').plot(kind='barh',figsize=(10,8))


# It seems like we have succesfully cleaned up the unspecified data.

# ### Let's see which address has the most complaints

# In[ ]:


#!pip install wordcloud
from wordcloud import WordCloud, STOPWORDS
print('Import Successfull')


# In[ ]:


from collections import Counter

count_dict = Counter(df['Incident Address'])
stopwords= set(STOPWORDS)
wc = WordCloud(background_color='white', max_words=20, stopwords=stopwords).generate_from_frequencies(count_dict)
#unique_string = ("").join(list(df['Incident Address'].astype('str')))
#wc = WordCloud(background_color='white', max_words=20, stopwords=stopwords).generate(unique_string)


plt.figure(figsize=(15,10))
plt.imshow(wc)
plt.axis('off')
plt.show()


# This gives us a good idea about the address where are the highest numbers of complaints came from. To get which street has the maximum number of complaints, we will do a sorting.

# In[ ]:


df.groupby('Incident Address')[['Unique Key']].count().sort_values('Unique Key',ascending=False).head()


# Now Let's get the full address of this

# In[ ]:


df[df['Incident Address']=='34 ARDEN STREET'][['Incident Address','Incident Zip','Borough']].head(1)


# ### The address where most number of complaints came from is 
# ### 34 ARDEN STREET, MANHATTAN 10040

# In[ ]:


df.groupby('Incident Zip')[['Unique Key']].count().sort_values('Unique Key',ascending=False).head()


# ### The zipcode where the most number of complaints came from is 
# ### 11226

# In[ ]:


df.groupby('Status')[['Unique Key']].count().sort_values('Unique Key',ascending=False).head()


# In[ ]:


df_pluto['bldgage']=2020-df_pluto['yearbuilt']
df_pluto.head()


# In[ ]:


df[df['Complaint Type']=='HEAT/HOT WATER'].head()


# In[ ]:


df_comp_count=df[df['Complaint Type']=='HEAT/HOT WATER'].groupby('Incident Address')[['Incident Address']].count()


# In[ ]:


df_comp_count.columns=['count of complaints']
df_comp_count['address']=df_comp_count.index
df_comp_count.head()


# In[ ]:


#df_comp_count.index=None
df_comp_count.reset_index(drop=True,inplace=True)
df_comp_count.head()


# In[ ]:


df_corr = pd.merge(df_comp_count,df_pluto,on='address')
df_corr.head()


# In[ ]:


#df_corr['alterage']= 2020-df_corr['yearalter1']
df_corr.drop(columns=['yearbuilt'],inplace=True)
df_corr.head()


# In[ ]:


data = df_corr.drop(columns=['address'])

null_data=data.isnull()

for i in null_data.columns:
    result=0
    for j in null_data[i]:
        if j: result+=1
    print(i," has ",result, " null values")



# We have a lot of null values. We have to deal with it before I try to do any machine learning on it. If we look carefully, they have a pattern of how many null values are in the columns. Maybe they have the null values for similar reason and we can replace the null values with similar approach.

# In[ ]:


data.describe()


# #plt.figure(figsize=(10,8))
# f, axes = plt.subplots(11,2,figsize=(8,30))
# for i in range(len(data.columns)):
#     sns.boxplot((data.iloc[:,i]),orient='vert',ax=axes[int(i/2),int(i%2)])
# #sns.boxplot((data['bldgdepth']),color='g',orient='vert',ax=ax)
# plt.show()

# f, axes = plt.subplots(11,2,figsize=(10,40))
# for i in range(len(data.columns)):
#     sns.distplot(data.iloc[:,i],kde=False, ax=axes[int(i/2),int(i%2)])
# #sns.boxplot((data['bldgdepth']),color='g',orient='vert',ax=ax)
# plt.show()

# sns.pairplot(data.T, size=2.5, corner=True)

# In[ ]:


for i in null_data.columns:
    result=0
    for k,j in enumerate(null_data[i]):
        if j: data.at[k,i]= data[i].mean()
            
            
null_data=data.isnull()
            
for i in null_data.columns:
    result=0
    for j in null_data[i]:
        if j: result+=1
    print(i," has ",result, " null values")


# In[ ]:




