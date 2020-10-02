#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('../input/2016 School Explorer.csv')
df.head()


# Missing value analysis

# In[ ]:


total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
df2 = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
df2


# In[ ]:


#School income estimate has a high percentage of missing values i.e. 31%
#Since school income is different for different schools, so if we have to use school income esstimate for any analysis,
#Then we have to omit the schools where value=NA


# Treating duplicate data

# In[ ]:


duplicate_bool=df.duplicated()
duplicate=df.loc[duplicate_bool == True]
print(duplicate)


# ANALYSIS
# Ques-1: What is the distribution between city and economic need index based on the presence of community school? Taking only top 500 schools based on economic need index

# In[ ]:


df1=df.sort_values(by=['Economic Need Index'],ascending=False)
df1=df1.iloc[:500,]
print(df1[['City','Economic Need Index']])
sns.stripplot(x="Economic Need Index", y="City",hue="Community School?",data=df1)


# The graph shows that most of the community schools are present in those areas where economic need index are high.Some cities like New York have different areas/schools with different economic need index

# Ques-2: What is the distribution of student attendance rate in community school in New York and Brooklyn?

# In[ ]:


subset=df.loc[(df.City.isin(['NEW YORK','BROOKLYN'])& (df['Community School?']=="Yes"),['Student Attendance Rate'])]
subset
subset['count']=subset.index
sns.barplot(x="count",y="Student Attendance Rate",data=subset)


# Student attendance rate is between 86 to 94 percent and most occuring values are 86 and 89%

# Ques-3:Finding the distribution of collaborative teachers rating having supportive environment%>70%

# In[ ]:



df.reset_index(inplace=True)
a=df.pivot_table(values = 'index', index =['Collaborative Teachers Rating'],columns=df['Supportive Environment %']>"70%", aggfunc = 'count')
print(a)
a.columns=['Lessthan70','Greaterthan70']
a['Collaborative Teachers Rating']=a.index
sns.barplot(x="Greaterthan70", y="Collaborative Teachers Rating",data=a,color="blue") 


# When supporting environment> 70%, most teachers are either exceeding or meeting target

# Ques 4: What is the association between average ELA profiency and average Math Proficiency based on the presence of community school?

# In[ ]:


sns.lmplot(x="Average ELA Proficiency", y="Average Math Proficiency", hue="Community School?", markers=["*","+"], palette="Set2", data=df)


# There is a positive linear relationship between Average ELA and average Math proficiency.The presence of community schools is marked by lower values of average ELA and average Math proficiency.

# Ques-5: What is the association between students chronically missing schools (being absent) and supportive environment rating?
# Ques 6: Show how rigorous instruction and collaborative teachers rating affect effective school leadership?

# In[ ]:


col1=['Rigorous Instruction %']
df[col1]=df[col1].replace({'\%':' '},regex=True)
df[col1]=df[col1].astype('float64')
col2=['Effective School Leadership %']
df[col2]=df[col2].replace({'\%':' '},regex=True)
df[col2]=df[col2].astype('float64')
col3=['Collaborative Teachers %']
df[col3]=df[col3].replace({'\%':' '},regex=True)
df[col3]=df[col3].astype('float64')
col4=['Percent of Students Chronically Absent']
df[col4]=df[col4].replace({'\%':' '},regex=True)
df[col4]=df[col4].astype('float64')
col5=['Supportive Environment %']
df[col5]=df[col5].replace({'\%':' '},regex=True)
df[col5]=df[col5].astype('float64')


# In[ ]:


a=df[col1]
b=df[col2]
c=df[col3]
d=df[col4]
e=df[col5]
result=pd.concat([a, b,c,d,e], axis=1)
#result
plt.figure(1)
sns.jointplot(x="Percent of Students Chronically Absent", y="Supportive Environment %",data=result)
plt.figure(2)
sns.jointplot(y="Effective School Leadership %", x="Rigorous Instruction %",data=result)
plt.figure(3)
sns.jointplot(y="Effective School Leadership %", x="Collaborative Teachers %",data=result)


# Supportive environment has little or no effect on the percentage of students chronically absent. Supportive environment % is between 80 and 100% and percent of students chronically absent is mostly between 0 and 50%.
# Both collaborative teachers rating and rigorous instruction have a positive and linear impact on effective school leadership

# Ques-7: Display the average ELA and math proficiency score for black/hispanic and asian/white dominant schools

# In[ ]:


#Display the Mean ELA and Math Scores for Black/Hispanic Dominant Schools
df[df['Percent Black / Hispanic'] >= '70%'][['Average ELA Proficiency','Average Math Proficiency']].mean()


# In[ ]:


#Display the Mean ELA and Math Scores for White/Asian Dominant Schools
df[df['Percent Black / Hispanic'] <= '30%'][['Average ELA Proficiency','Average Math Proficiency']].mean()


# In[ ]:


# Create New Column for Black/Hispanic Dominant Schools
df['Black_Hispanic_Dominant'] = df['Percent Black / Hispanic'] >='70%'

#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == True),'Average Math Proficiency'] , color='b',shade=True, label='Black/Hispanic Dominant School')
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == False),'Average Math Proficiency'] , color='r',shade=True, label='Asian/White Dominant School')
plt.title('Average Math Proficiency Distribution by Race')
plt.xlabel('Average Math Proficiency Score')
plt.ylabel('Frequency Count')


# In[ ]:


#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == True),'Average ELA Proficiency'] , color='b',shade=True, label='Black/Hispanic Dominant School')
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == False),'Average ELA Proficiency'] , color='r',shade=True, label='Asian/White Dominant School')
plt.title('Average ELA Proficiency Distribution by Race')
plt.xlabel('Average ELA Proficiency Score')
plt.ylabel('Frequency Count')


# Ques-8 : Finding the top 10 cities with most schools and community schools

# In[ ]:


city = df.groupby('City')['Zip'].count()
community=df[df["Community School?"]=='Yes'].groupby('City')['Zip'].count()
merge=pd.concat([city,community],axis=1)
merge.fillna(0, inplace=True)
merge.columns=['Number_of_schools','Number_of_CommunitySchools']
merge.sort_values(by=['Number_of_schools','Number_of_CommunitySchools'],ascending=False)
top_10=merge.iloc[:10,]
print(top_10)
top_10[['Number_of_schools','Number_of_CommunitySchools']].plot(kind='bar', stacked=True)


# Brooklyn has the most number of community and non community schools followed by Bronx
