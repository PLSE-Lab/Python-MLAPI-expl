#!/usr/bin/env python
# coding: utf-8

# Feel free to criticise the Visualizations and the analysis.<br> Any sort of feedback is appreciated. **Please upvote if you like kernel ! **:) Thank You! 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df = pd.read_csv('../input/master.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# **HDI for year** is mostly empty and I don't see any need for filling it since we aren't modelling and the **country-year** column is useless as both6 th6e information is present in seperate columns, so I'll be deleting them. 

# In[ ]:


df.drop(['HDI for year', 'country-year'],axis=1,inplace=True)


# In[ ]:


df['age'].value_counts()


# **Creating the vector map**

# In[ ]:


a_map = {
'75+ years' : 5,      
'55-74 years' : 4 ,    
'35-54 years' : 3,    
'25-34 years' : 2,    
'15-24 years' : 1,   
'5-14 years':  0     
 }


# In[ ]:


df['age'] = df['age'].map(a_map)


# ## Visualizations...

# In[ ]:


#Visualizations start here....
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('darkgrid')
sns.countplot(df['age'])
#df['age'].plot(kind = 'hist')


# In[ ]:


sns.countplot(df['sex'])


# So I have a feeling that developed countries have more suicides as compared to developing or underdeveloped countries (basically not developed)<br>
# So I'll be taking countries with **gdp_per_capita** greater than the median as developed (for my analysis).<br>
# * 0 : Not Developed
# * 1 : Developed

# In[ ]:


df['developed'] = 0
df.loc[ df['gdp_per_capita ($)'] > df['gdp_per_capita ($)'].median(), 'developed']  = 1


# In[ ]:


df.head()


# In[ ]:


df.describe()


# So I wanna plot graph which shows suicide rates (counts) in develoved countries and one in developing / underdeveloped countries<br>
# X axis : Year<br>
# Y axis : Number of Suicides

# In[ ]:


palette = {0 : 'red', 1: 'green'}


# In[ ]:


df_1 = df.groupby(['year','developed'],as_index=False).sum()
#df_1.drop(df_1[0])
plt.figure(figsize = (10,6)) 
plt.title('No. of suicides vs Year')
sns.lineplot(x = 'year', y ='suicides_no', data =df_1, hue='developed',palette = palette)
sns.scatterplot(x = 'year', y ='suicides_no', data =df_1, hue='developed', palette = palette)
#df_1.head()


# In[ ]:


df_1 = df.groupby(['year','developed'],as_index=False).mean()
df_1.head()
plt.figure(figsize = (10,6))
plt.title('Suicides/100k people vs Year')
sns.lineplot(x = 'year', y ='suicides/100k pop', data =df_1, hue='developed')
sns.scatterplot(x = 'year', y ='suicides/100k pop', data =df_1, hue='developed')


# * Okay this seems very interesting, the data from under-develepoed countries -- Seems like we could fit a Gaussian through it.<br><br>
# In fact, the years from 1993 to 2004 (approx) seem to have a rise in the number of suicides in the underdeveloped countries. I wasn't really able to find out why, if anyone has any insights, feel free to comment.

# In[ ]:


#A funtion which plots bar graphs for a given feature  and groups them by developed and not-developed 
def graph(feature):
    #define separate dataframes for developed and not developed  
    s_df = df[df['developed']==1].groupby([feature]).sum()['suicides_no']  #Developed
    d_df = df[df['developed']==0].groupby([feature]).sum()['suicides_no']  #Not Developed
    df1 = pd.DataFrame([s_df,d_df])
    df1.index = ['Developed','Not Developed']
    df1.plot(kind='bar') #we can stack them by using stacked = True
    plt.ylabel(feature)
    
    


# In[ ]:


graph('sex')
graph('age')


# * Again, its pretty clear that developed countries are home to a higher number of suicide rates.
# * In both scenarios, its the people in the age range of 35-55 years who have committed suicide in high numbers.(Highest, in fact). 
# * Not developed countries have a higher number of suicides only in the first category : 5-14 years. 
# * Higher number of males committed suicide as compared to females(also differing by around 1.7M men/people) 

# In[ ]:


def graph2(feature):
    #define separate dataframes for developed and not developed
    #value_counts() counts unique objects in a given feature
    s_df = df[df['developed']==1].groupby([feature])['suicides/100k pop'].mean()  #Developed
    d_df = df[df['developed']==0].groupby([feature])['suicides/100k pop'].mean()  #Not Developed
    df1 = pd.DataFrame([s_df,d_df])
    df1.index = ['Developed','Not Developed']
    df1.plot(kind='bar') #we can stack them by using stacked = True
    plt.ylabel(feature)
    #plt.show()


# In[ ]:


graph2('age')
graph2('sex')


# While comparing the suicide rates(no_suicides/100k) there doesn't seem to be much of a difference while comparing age groups and sex of the people, although still, the developed countries seem to have a higher rate.

# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize = (15,10))
df1 = df.groupby(['country','year'],as_index=False).sum()
plt.subplot(221)
plt.title('SuicideNos vs Population grouped by country and year')
sns.scatterplot(x = 'population', y = 'suicides_no', data = df1, alpha = 0.5)

plt.subplot(222)
plt.title('SuicideNos vs Population ')
sns.scatterplot(x = 'population', y = 'suicides_no', data = df, alpha = 0.5, hue = 'developed', palette = palette)

plt.subplot(223)
plt.title('Suicide Nos vs Gdp per Capita grouped by country and year')
sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides_no', data = df1, alpha = 0.5)

plt.subplot(224)
plt.title('SuicideNos vs Gdp per Capita')
sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides_no', data = df, alpha = 0.5, hue = 'developed', 
                palette = palette)


# Higher population in developed countries could be the reason for higher suicide numbers in them.

# In[ ]:


plt.figure(figsize = (15,10))
df1['suicides/100k pop'] = df1['suicides/100k pop']/12
df1['gdp_per_capita ($)'] = df1['gdp_per_capita ($)']/12

plt.subplot(221)
plt.title('Avg Suicide Rate vs Population grouped by country and year')
sns.scatterplot(x = 'population', y = 'suicides/100k pop', data = df1, alpha = 0.5)

plt.subplot(222)
plt.title('Avg Suicide Rate vs Population')
sns.scatterplot(x = 'population', y = 'suicides/100k pop', data = df, alpha = 0.5, hue = 'developed', palette = palette)

plt.subplot(223)
plt.title('Avg Suicide Rate vs Avg Gdp per capita grouped by country and year')
sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides/100k pop', data = df1, alpha = 0.5)

plt.subplot(224)
plt.title('Avg Suicide Rate vs Avg Gdp per capita')
sns.scatterplot(x = 'gdp_per_capita ($)', y = 'suicides/100k pop', data = df, alpha = 0.5, hue = 'developed',
                palette = palette)


# In[ ]:


plt.figure(figsize = (14,7))
plt.subplot(121)
sns.countplot(df.generation)
plt.subplot(122)
df['generation'].value_counts().plot.pie(shadow = True,explode=[0.15,0.15,0.15,0.15,0.15,0.15], autopct='%1.1f%%')


# In[ ]:


graph('generation')
graph2('generation')


# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(df1.corr(),annot=True)


# In[ ]:


df1 = df.drop(['developed'],axis=1)
sns.pairplot(df1,hue = 'generation',diag_kind = 'hist',palette = 'husl')


# In[ ]:




