#!/usr/bin/env python
# coding: utf-8

# ## Insights from the Data:
# 1-Rajasthan is the most affected state in the Country and Bihar,Assam, Orissa bags the immediate next position to Rajasthan.
# 2-In most of the places in Rajasthan water quality is affected by Salinity content in water.
# 3-Water treatment plans in the states JAMMU AND KASHMIR,UTTARAKHAND,PUDUCHERRY & MANIPUR are effective
# 4-It can be understood that from the year 2009 to 2012 there is significant improvement in water quality overall
# 5-Water treatment plans followed in 2011 seems to have postive impact 
# 6-Apart from Nitrate, all the Quality Parameters followed a similar pattern through out the years. The presence of all these elements in water reduced significantly over years
# 7-Nitrate has a peak in 2011
# 8-It is found that Karnataka and Rajasthan states accounted more than 75% of total Nitrate content in 2011.
# 9-From 2010 to 2011, the amount of Nitrate got tripled in Karnataka and halved in Maharashtra.This might be because of numerous reasons like high use of inorganic fertilizers, pipeline leakages, Industrial pollution etc.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


#Loading Data
df = pd.read_csv("../input/IndiaAffectedWaterQualityAreas.csv",encoding='latin1')
df.head()


# ## Data Description

# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# In[ ]:


#Checking for Duplicates
Duplicates=df[df.duplicated(keep=False)]
Duplicates.shape


# In[ ]:


#Dropping Duplicates
df1=df.drop_duplicates(subset=['State Name', 'District Name', 'Block Name', 'Panchayat Name',
       'Village Name', 'Habitation Name', 'Quality Parameter', 'Year'], keep=False)
df1.shape


# In[ ]:


df1.describe()


# ## Iron is the most frequent element that causes water degradation in the country

# In[ ]:


df1['Quality Parameter'].value_counts()


# In[ ]:


import seaborn as sns
sns.countplot(df1['Quality Parameter'])


# ## Rajasthan is recorded in the first place to face water degaradation problems

# In[ ]:


df1['State Name'].value_counts()
#This may be because area wise Rajasthan is ranked highest in India


# In[ ]:


df1['Quality Parameter'].groupby(df1['State Name']).describe()


# In[ ]:


#Splitting the Year column to retain just the year
df1['year'] = pd.DatetimeIndex(df1['Year']).year
df1=df1.drop(columns='Year')


# ## Considering the top 4 affected states RAJASTHAN,BIHAR,ASSAM & ORISSA 

# In[ ]:


#Subsetting the data
df1_new=df1.loc[df['State Name'].isin(['RAJASTHAN','BIHAR','ASSAM','ORISSA'])] 
Subset_Data = df1_new[['State Name', 'Quality Parameter', 'year']]


# In[ ]:


#Assigning a numerical value to all the Quality Parameters
import sklearn
from sklearn.preprocessing import LabelEncoder
numbers = LabelEncoder()
Subset_Data['Quality'] = numbers.fit_transform(Subset_Data['Quality Parameter'].astype('str'))


# In[ ]:


State_Quality_Count = pd.DataFrame({'count' : Subset_Data.groupby( [ "State Name", "Quality","Quality Parameter"] ).size()}).reset_index()
State_Quality_Count.head()


# #### Shows the top 5 water degrading Quality parameters and their corresponding state

# In[ ]:


High_Quality_count = State_Quality_Count.sort_values(['count'], ascending=[False])
High_Quality_count.head()


# #### Further drilldown with year

# In[ ]:


State_Quality_Count_year = pd.DataFrame({'count' : Subset_Data.groupby( [ "State Name", "Quality","Quality Parameter","year"] ).size()}).reset_index()
State_Quality_Count_year


# In[ ]:


State_Quality_Count_year['rank']=State_Quality_Count_year.groupby(['State Name','Quality'])['count'].rank("dense", ascending=False)
State_Quality_Count_year.head()


# In[ ]:


Top_count=State_Quality_Count_year[State_Quality_Count_year['rank']==1]


# #### The top 5 states considered have encountered the highest water degradtion in the year of 2009 

# In[ ]:


import matplotlib.pyplot as plt
freq_plot = Top_count['year'].value_counts().plot(kind='bar',figsize=(9,5),title="Year with highest water degradation")
freq_plot.set_xlabel("Year")
freq_plot.set_ylabel("Frequency")
plt.show()


# #### Applying this concept to the entire data to see the worst and best year for water quality
# 2009 remained to be the year with highest degration where as 2011 has shifted to last position when plotted on entire data.

# In[ ]:


Subset_Data2 = df1[['State Name', 'Quality Parameter', 'year']]
Subset_Data2['Quality'] = numbers.fit_transform(Subset_Data2['Quality Parameter'].astype('str'))
SQT = pd.DataFrame({'count' : Subset_Data2.groupby( [ "State Name", "Quality","Quality Parameter","year"] ).size()}).reset_index()
SQT['rank']=SQT.groupby(['State Name','Quality'])['count'].rank("dense", ascending=False)
Top_count2=SQT[SQT['rank']==1]


# In[ ]:


freq_plot = Top_count2['year'].value_counts().plot(kind='bar',figsize=(9,5),title="Year with highest water degradation")
freq_plot.set_xlabel("Year")
freq_plot.set_ylabel("Frequency")
plt.show()


# #### Trend of Quality Parameters over the years
# Except Nitrate, all the Quality Parameters followed a similar pattern through out the years. The amount of all the elements resuced significantly over years.

# In[ ]:


Quality = pd.DataFrame({'count' : Subset_Data2.groupby( [ "Quality","year"] ).size()}).reset_index()
Quality.head()


# In[ ]:


Arsenic=Quality[Quality['Quality']==0]
Arsenic.plot('year','count',kind='scatter', layout=(5,5),title= "Arsenic distribution")
plt.show()


# In[ ]:


Fluoride=Quality[Quality['Quality']==1]
Fluoride.plot("year",'count',kind='scatter', layout=(5,5),title= "Fluoride distribution")
plt.show()


# In[ ]:


Iron=Quality[Quality['Quality']==2]
Iron.plot('year','count',kind='scatter', layout=(5,5),title= "Iron distribution")
plt.show()


# In[ ]:


Nitrate=Quality[Quality['Quality']==3]
Nitrate.plot('year','count',kind='scatter', layout=(5,5),title= "Nitrate distribution")
plt.show()


# In[ ]:


Salinity=Quality[Quality['Quality']==4]
Salinity.plot('year','count',kind='scatter', layout=(5,5),title= "Salinity distribution")
plt.show()


# #### Digging deep to know the reasons why Nitrate peaked its value in 2011 and in which states

# In 2011

# In[ ]:


State_Quality = pd.DataFrame({'count' : Subset_Data2.groupby( [ "State Name","Quality","year"] ).size()}).reset_index()
State_Nitrate_Quality=State_Quality[(State_Quality['year']==2011) & (State_Quality['Quality']==3)]
State_Nitrate_Quality.sort_values(['count'], ascending=[False])


# In 2010

# In[ ]:


State_Nitrate_Quality1=State_Quality[(State_Quality['year']==2010) & (State_Quality['Quality']==3)]
State_Nitrate_Quality1.sort_values(['count'], ascending=[False])


# #### Karnataka and Rajasthan has high amounts of Nitrate in 2011 accounting approximately 47% and 29% of the total .
# #### Also from 2010 to 2011, the amount of Nitrate got tripled in Karnataka and halved in Maharashtra.This might be numerous reasons for this increase like high use of inorganic fertilizers, pipeline leakages, Industrial pollution etc.
