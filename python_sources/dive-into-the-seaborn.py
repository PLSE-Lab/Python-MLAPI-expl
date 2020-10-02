#!/usr/bin/env python
# coding: utf-8

# # Dive Into The Seaborn

# Hello Kagglers!!
# 
# Here is the notebook for the basic tutorial for **Seaborn** and **Matplotlib** using Daily Temperature dataset.I used simple EDA techniques for the better visualisation of dataset.
# 
# If you like this notebook **PLEASE UPVOTE!!**

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/daily-temperature-of-major-cities/city_temperature.csv')
df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# Here we can see that **State** column has approx 50% null values,so we can drop it.Also we can see that the minimum value in **Year** column is 200, which has to be a dataset error and we can replace it easily.

# In[ ]:


df['Year'].unique()


# There are two errors: 200 & 201, which may be 2000 & 2010 respectively.

# In[ ]:


df = df.drop('State',axis=1)
df.loc[df['Year']==200,'Year']=2000
df.loc[df['Year']==201,'Year']=2010
df.head()


# # EXPLORATORY DATA ANALYSIS:
# 
# OK!! so lets start some visualisation of Average Mean Temperature of Earth.

# ### 1. AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS:
# 

# In[ ]:


s= df.groupby(['Region'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)
s.style.background_gradient(cmap='Purples')


# In[ ]:


plt.figure(figsize=(18,8))
sns.barplot(x='Region', y= 'AvgTemperature',data=s,palette='hsv_r')
plt.title('AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS')
plt.show()


# ### 2. YEAR-WISE AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS:
# 

# In[ ]:


a= df.groupby(['Year','Region'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)
a.head(20).style.background_gradient(cmap='Blues')


# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Year',y='AvgTemperature',hue='Region',data=a,palette='hsv')
plt.grid()
plt.title('YEAR-WISE AVERAGE MEAN TEMPERATURE OF DIFFERENT REGIONS')
plt.show()


# See how the mean temperature of **AFRICA** region increases drastically over the years.

# ### 3. VARIATION OF MAXIMUM TEMPERATURE OVER THE MONTHS:

# In[ ]:


b= df.groupby(['Region','Month'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)
b.head(20).style.background_gradient(cmap='Oranges')


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(x='Month', y= 'AvgTemperature',data=b,hue='Region',palette='hsv',saturation=.80)
plt.title('VARIATION OF MAXIMUM TEMPERATURE OVER THE MONTHS')
plt.show()


# ### 4. VARIATION OF MAXIMUM TEMPERATURE OVER THE YEARS:

# In[ ]:


c= df.groupby(['Region','Year'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)
c.head(20).style.background_gradient(cmap='Greens')


# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot(x='Year',y='AvgTemperature',data=c,hue='Region',palette='hsv_r',style='Region')
plt.title(' VARIATION OF MAXIMUM TEMPERATURE OVER THE YEARS')
plt.show()


# ### 5. VARIATION OF MEAN TEMPERATURE FOR TOP 20 COUNTRIES:
# 

# In[ ]:


c= df.groupby(['Country','City'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False).head(20)
c.style.background_gradient(cmap='Reds')


# In[ ]:


plt.figure(figsize=(8,10))
sns.barplot(x='AvgTemperature',y='City',data=c,palette='hsv_r')
plt.title('VARIATION OF MEAN TEMPERATURE FOR TOP 20 COUNTRIES')
plt.show()


# ### 6. VARIATION OF MEAN TEMPERATURE OVER THE MONTHS FOR EACH REGION:

# In[ ]:


africa=df[df['Region']=='Africa']
d= africa.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)


asia=df[df['Region']=='Asia']
e= asia.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)


mid_est=df[df['Region']=='Middle East']
p= mid_est.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)


n_amer=df[df['Region']=='North America']
q= n_amer.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)


eup=df[df['Region']=='Europe']
r= eup.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)


sth=df[df['Region']=='South/Central America & Carribean']
s= sth.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)


aus=df[df['Region']=='Australia/South Pacific']
m= aus.groupby(['Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)




# In[ ]:


plt.figure(figsize=(15,20))
plt.subplot(4,2,1)
sns.barplot(x='Month',y='AvgTemperature',data=d,palette='hsv')
plt.title('Africa')

plt.subplot(4,2,2,)
sns.barplot(x='Month',y='AvgTemperature',data=e,palette='hsv')
plt.title('Asia')


plt.subplot(4,2,3)
sns.barplot(x='Month',y='AvgTemperature',data=p,palette='hsv')
plt.title('Middle East')


plt.subplot(4,2,4)
sns.barplot(x='Month',y='AvgTemperature',data=q,palette='hsv')
plt.title('North America')


plt.subplot(4,2,5)
sns.barplot(x='Month',y='AvgTemperature',data=r,palette='hsv')
plt.title('Europe')


plt.subplot(4,2,6)
sns.barplot(x='Month',y='AvgTemperature',data=s,palette='hsv')
plt.title('South/Central America & Carribean')


plt.subplot(4,2,7)
sns.barplot(x='Month',y='AvgTemperature',data=m,palette='hsv')
plt.title('Australia/South Pacific')

plt.show()


# We can clearly see that **Africa** and **South/Central America & Carribean** haven't much variations over the months as compare to other regions.
# 

# ### 7. VARIATION OF MEAN TEMPERATURE IN INDIA:

# In[ ]:


ind=df[df['Country']=='India']
x= ind.groupby(['Year'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)
x.style.background_gradient(cmap='hsv')


# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Year',y='AvgTemperature',data=x,color='r')
plt.grid()
plt.title('Mean Temp. Variation in India') 
plt.show()


# Average temperature of India continuously decreases in recent years.

# ### 8.  VARIATION OF MAXIMUM TEMPERATURE IN DIFFERENT CITIES OF INDIA:

# In[ ]:


ind=df[df['Country']=='India']
x= ind.groupby(['City','Year'])['AvgTemperature'].max().reset_index().sort_values(by='AvgTemperature',ascending=False)
x.head(20).style.background_gradient(cmap='Blues')


# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Year',y='AvgTemperature',data=x,hue='City',style='City',markers=['o','*','^','>'])
plt.grid()
plt.title('Mean Temp. Variation Of Cities of India')
plt.show()


# ### 9. VARIATION OF MEAN TEMPERATURE OF CAPITAL DELHI:
# 

# In[ ]:



mask1=df['Country']=='India'
mask2=df['City']=='Delhi'

ind=df[mask1 & mask2 ]


y= ind.groupby(['Year','City','Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)
y.head(20).style.background_gradient(cmap='PiYG')


# In[ ]:


plt.figure(figsize=(15,12))
plt.subplot(2,1,1)
sns.barplot(x='Year',y='AvgTemperature',data=y,palette='hsv_r')
plt.title('Mean Temp. Variation Of Delhi(Yearly)')

plt.subplot(2,1,2)
sns.barplot(x='Month',y='AvgTemperature',data=y,palette='hsv')
plt.title('Mean Temp. Variation Of Delhi(Monthly)')

plt.show()


# ### 10. VARIATION OF MEAN TEMPERATURE OF INDIA IN YEAR 2020:
# 

# In[ ]:


mask1=df['Country']=='India'
mask2=df['Year']==2020

ind=df[mask1 & mask2 ]


k= ind.groupby(['Year','City','Month'])['AvgTemperature'].mean().reset_index().sort_values(by='AvgTemperature',ascending=False)
k.style.background_gradient(cmap='Greens')


# In[ ]:


plt.figure(figsize=(15,8))
sns.lineplot(x='Month',y='AvgTemperature',data=k,hue='City',style='City',markers=['*','o','<','>'])
plt.grid()
plt.title('Mean Temp. Variation Of Cities of India in 2020')
plt.show()


# That's all for this notebook.Hope you guys like it.
# 
# I tried to make this notebook as simple as i can.If you like this work **PLEASE UPVOTE** to keep me motivated to learn and write more.
# 
# 
# **THANK YOU FOR YOUR TIME!!**
