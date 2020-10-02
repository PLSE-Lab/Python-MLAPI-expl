#!/usr/bin/env python
# coding: utf-8

# - <a href='#intro'>1. Introduction</a>  
# - <a href='#rtd'>2. Retrieving the Data</a>
#      - <a href='#ll'>2.1 Load libraries</a>
#      - <a href='#rrtd'>2.2 Read the Data</a>
# - <a href='#oot'>3. Overview of data</a> 
# - <a href='#dp'>4. Data preparation</a>
#      - <a href='#cfmd'> 4.1 Check for missing data and correct </a>
# - <a href='#de'>5. Data Exploration</a>
# 

# In[125]:


# - <a href='#s'>6. Summary/Conclusion</a> 


# ## <a id='intro'>1. Intoduction</a>
# Flaredown is an app that helps patients of chronic autoimmune and invisible illnesses improve their symptoms by avoiding triggers and evaluating their treatments. Each day, patients track their symptom severity, treatments and doses, and any potential environmental triggers (foods, stress, allergens, etc) they encounter.

# # <a id='rtd'>2. Retrieving the Data</a>

# ## <a id='ll'>2.1 Load libraries</a>

# In[126]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt # for plotting
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[127]:


import os
print(os.listdir("../input"))


# ## <a id='rrtd'>2.2 Read tha Data</a>

# In[128]:


df = pd.read_csv("../input/fd-export.csv")


# In[129]:


print("Size of Flaredown data or df is",df.shape)


# ## <a id='oot'>3 Overview of data</a>

# In[130]:


df.head()


# In[131]:


df.info()


# In[132]:


df.nunique()                     # number of unique counts in different parameters


# In[133]:


df.describe(include=["O"])


# # <a id='dp'>4. Data preparation</a>

# ### 'user_id' is alphanumerical and unique for distinct people, so let's use unique integer user ids to save memory

# In[134]:


df['user_id'] = pd.Categorical(df['user_id'])
df['user_id']= df.user_id.cat.codes


# In[135]:


df.head()


# 
# # <a id='cfmd'>4.1 Check for missing data</a>

# We can see above that there are many values in age column which are 0.0 which can bias our inference. So I am replacing these by NaN for consistency.

# In[136]:


df["age"] = df.age.replace(0.0,np.nan)


# In[137]:


df.head()


# In[138]:


df.age.describe()


# Here , minimum and maximum age are not valid, we need to clean it more .

# In[139]:


df[(df['age'] > 100)].sort_values(by='age',ascending=True).head(10)


# #### Since negative age and above 117 is not practically possible so replacing them by NaN.

# In[140]:


df[(df['age'] > 117) | (df['age'] < 0) ].shape  # number of columns to be replaced by NaN


# In[141]:


df = df.assign(age = lambda x: x.age.where(x.age.ge(0)))    # ALl negative ages replaced by NaN for consistency


# In[142]:


df = df.assign(age = lambda x: x.age.where(x.age.le(118)))  # All ages greater than 117 are replaced by NaN


# In[143]:


df[(df['age'] > 117) | (df['age'] < 0) ].shape  # as we can see they are replced


# In[144]:


df.age.describe()   # now age statistics makes more sense


# ##### As we can see above average age of users  is 32 years old with minimum 1 year and maximum 117 years old

# # <a id='de'>5. Data Exploration</a>

# In[145]:


print("Total numer of unique users are ",df.user_id.nunique())


# ### Categorization on the basis sex of users

# In[146]:


df.sex.value_counts()  # Total number of check-ins of differet sex categories


# In[147]:


df_sex_unique = pd.DataFrame([{'Number_of_Users' : df[df.sex=="doesnt_say"].user_id.nunique()}
                             ,{'Number_of_Users' : df[df.sex=="other"].user_id.nunique()}
                             ,{'Number_of_Users' :  df[df.sex=="male"].user_id.nunique()}
                             ,{'Number_of_Users' : df[df.sex=="female"].user_id.nunique()}
                             ], index=['Doesnt_say', 'Others', 'Male','Female'])
df_sex_unique.head()


# In[148]:


plt.figure(figsize=(10,6))
df_sex_unique.Number_of_Users.plot(kind='pie')


# ### Categorization on the basis of different 'trackable_type' 

# #### It makes more sense to categorize them using 'trackable _type'  to analyse further and obtain correlation between them.

# In[149]:


df.trackable_type.value_counts()


# In[150]:


df.trackable_type.value_counts().plot(kind='barh')


# ## Symptoms

# In[151]:


print("Total numer of unique symptoms ('trackable_name') tracked are",df[df.trackable_type=="Symptom"].trackable_name.nunique())


# In[152]:


df[df.trackable_type=="Symptom"].trackable_name.value_counts().head(10)  # Top 10 different symptoms traced


# In[153]:


plt.figure(figsize=(15,15))
sector_name = df[df.trackable_type=="Symptom"].trackable_name.value_counts().iloc[0:50]
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases registered')
plt.ylabel('Symptom Name')
plt.title(" Common symptoms registered")
plt.show()


# In[154]:


df1 = df.set_index(['user_id', 'age'])
df1.head()


# In[155]:


df1[df1.trackable_type == "Symptom"].trackable_name.head()


# In[156]:


df1[df1.trackable_type == "Treatment"].head(10)


# ## Weather Analysis

# In[157]:


print("Total numer of unique weather conditions ('trackable_name') are",df[df.trackable_type=="Weather"].trackable_name.nunique())


# In[158]:


df[df.trackable_type=="Weather"].trackable_name.value_counts()


# In[159]:


df[df.trackable_name=="temperature_min"].head()


# In[160]:


# df[df.trackable_name=="pressure"].trackable_value.unique()


# In[161]:


s_max = df[df.trackable_name=="temperature_max"].trackable_value
s_min = df[df.trackable_name=="temperature_min"].trackable_value


# In[162]:


max_temp = pd.to_numeric(s_max, errors='coerce')
min_temp = pd.to_numeric(s_min, errors='coerce')


# In[163]:


max_temp.describe()


# In[164]:


print (("Average maximum temperature recorded is") ,max_temp.describe()['mean'] )


# In[165]:


print (("Average mimimun temperature recorded is") ,min_temp.describe()['mean'] )


# In[166]:


#Pressure description
pd.to_numeric(df[df.trackable_name=="pressure"].trackable_value, errors='coerce').describe()


# In[167]:


#Humidity description
pd.to_numeric(df[df.trackable_name=="humidity"].trackable_value, errors='coerce').describe()


# In[168]:


# df[df.trackable_name=="precip_intensity"].trackable_value.unique()


# In[169]:


#Precipitation Intensity
pd.to_numeric(df[df.trackable_name=="precip_intensity"].trackable_value, errors='coerce').describe()


# ## Condition

# In[170]:


print("Total numer of unique conditions are",df[df.trackable_type=="Condition"].trackable_name.nunique())


# In[171]:


df[df.trackable_type=="Condition"].trackable_name.value_counts().head(10)


# In[172]:


# df[df.trackable_type=="Condition"].trackable_name.value_counts().iloc[0:30].plot(kind='bar')


# In[173]:


plt.figure(figsize=(15,15))
sector_name = df[df.trackable_type=="Condition"].trackable_name.value_counts().iloc[0:50]
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases registered')
plt.ylabel('Condition Name')
plt.title("Most Common Diseases Conditions registered")
plt.show()


# In[174]:


print("Total numer of unique Treatments are",df[df.trackable_type=="Treatment"].trackable_name.nunique())


# In[175]:


df[df.trackable_type=="Treatment"].trackable_name.value_counts().head(10)


# In[176]:


plt.figure(figsize=(15,15))
sector_name = df[df.trackable_type=="Treatment"].trackable_name.value_counts().iloc[0:50]
sns.barplot(sector_name.values, sector_name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases ')
plt.ylabel('Treatment Provided')
plt.title("Most Common Treatments Provided")
plt.show()


# ## Tags

# In[ ]:


print("Total numer of unique Tags are",df[df.trackable_type=="Tag"].trackable_name.nunique())


# In[ ]:


df[df.trackable_type=="Tag"].trackable_name.value_counts().head(10)


# In[ ]:


plt.figure(figsize=(15,15))
name = df[df.trackable_type=="Tag"].trackable_name.value_counts().iloc[0:50]
sns.barplot(name.values, name.index)
for i, v in enumerate(sector_name.values):
    plt.text(0.8,i,v,color='k',fontsize=10)
plt.xticks(rotation='vertical')
plt.xlabel('Number of cases ')
plt.ylabel('Treatment Provided')
plt.title("Most Common Treatments Provided")
plt.show()


# In[ ]:


from wordcloud import WordCloud

names = df[df.trackable_type=="Tag"].trackable_name.value_counts().iloc[0:500].index
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud)
plt.title("Wordcloud for Common Tags", fontsize=25)
plt.axis("off")
plt.show() 


# ## Food Habits

# In[ ]:


print("Total numer of unique Foods are",df[df.trackable_type=="Food"].trackable_name.nunique())


# In[ ]:


df[df.trackable_type=="Food"].trackable_name.value_counts().head(10)


# In[ ]:


# df[df.trackable_type=="Food"].trackable_name.value_counts().iloc[0:100].index


# In[ ]:


from wordcloud import WordCloud

names = df[df.trackable_type=="Food"].trackable_name.value_counts().iloc[0:100].index
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Food Taken by maximum people", fontsize=25)
plt.axis("off")
plt.show() 


# In[ ]:


# <a id='s'>5. Summary</a>


# In[ ]:




