#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading the Dataset

# In[ ]:


df = pd.read_csv(r"/kaggle/input/covid19updated2/raw_data_upd.csv",engine ='python')
df.head(10)


# In[ ]:


df.shape
print('The dataset contains', df.shape,' Rows and Columns')


# I excluded some of the columns like Sources of information, State code and Status Change Date  
# 
# Now the Dataset looks like as displayed below : 

# In[ ]:


df = df.drop(['Source_1','Source_2','Source_3','State code','Status Change Date','Unnamed: 19','Unnamed: 20','Unnamed: 21','Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25'],axis=1)
df.head()


# Using this dataset we will perform Data Exploration and Manipulation

# In[ ]:


df.shape


# In[ ]:


print('Now the updated dataset has', df.shape,'Rows and Columns')


# In[ ]:


print('The datatypes of the Parameters \n',df.dtypes)


# In[ ]:


print('All the variables in the dataset except Patient Number are Categorical ')


# # Univariate Analysis and Bivariate Analysis

# In[ ]:


print('Stats of Number of Patients    \n',df.describe())


# In[ ]:


df['Age Bracket'].value_counts()


# In[ ]:


print('Age bracket in the range of ',max(df['Age Bracket'].value_counts()),'are most affected, followed by the old aged')


# In[ ]:


t = df['Date Announced'].value_counts()
t.plot(kind='bar')


# In[ ]:


print('We can clearly see that number of cases went on multiplying every single day')
print('Maximum Number of cases',max(t),',were found on 13-04-2020')


# In[ ]:


(df['Gender'].value_counts()/len(df['Gender'])*100).plot(kind='bar')


# In[ ]:


print('Almost two and half times in number, males are affected than the number of women affected by the virus ')


# In[ ]:


print('Percentage of males(M) and females(F) infected in India\n',df['Gender'].value_counts()/len(df['Gender'])*100)


# In[ ]:


print('States with highest number of cases being detected\n',df['Detected State'].value_counts())


# In[ ]:


df['Detected State'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.title('Detected States')


# In[ ]:


print('Maharashtra is the most affected state with',max(df['Detected State'].value_counts()),'patients')


# In[ ]:


z= df['Current Status'].value_counts()
print('Current Status of the patients \n')
print('Total number of patients Hospitalized :',z.Hospitalized)
print('Total number of patients Recovered :',z.Recovered)


# In[ ]:


df['Current Status'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.title('Current Status')


# In[ ]:


z = df['Nationality'].value_counts()
print('Nationality of Coronavirus affected patients in India \n',z)


# In[ ]:


df['Nationality'].value_counts().plot.bar()
plt.ylabel('Count')
plt.title('Nationality')


# In[ ]:


print('Reasons for infection of virus \n')
df['Notes'].value_counts()


# People travelled to Delhi contributed to a huge no. of infections

# # People travelling to Delhi(STATS)

# In[ ]:


tem_df = df.loc[df['Notes']=='Travelled to Delhi']
print('Details of People travelling to Delhi and got affected')
tem_df


# In[ ]:


print('Totally',len(tem_df),'people got affected who travelled to Delhi')


# In[ ]:


print('The States that got affected')
z=tem_df['Detected State'].value_counts()
z


# In[ ]:


print('Totally', len(z),'States got affected') 


# In[ ]:


print('The States that got affected')
z=tem_df['Detected City'].value_counts()


# In[ ]:


print('Totally',len(z),'number of cities got affected')


# In[ ]:


print('Current Status of infected people:', tem_df['Current Status'].value_counts())


# In[ ]:


print('Number of Males(M) and females(F): \n',tem_df['Gender'].value_counts())


# In[ ]:


print('Dates on which these cases were found :\n',tem_df['Date Announced'].value_counts())


# # People travelling from abroad (STATS)

# In[ ]:


temp3_df = df.loc[(df['Nationality']=='India') & ((df['Notes']=='Travelled from Dubai') | (df['Notes']=='Travelled from UK'))]
print('Details of people travelling to India from abroad')
temp3_df


# In[ ]:


print('Total number: \n',temp3_df.shape)


# In[ ]:


print('Dates on which these cases were found: \n',temp3_df['Date Announced'])


# # General STATS

# In[ ]:


temp4_df = df.loc[df['Nationality']=='India']
z=temp4_df['Notes'].value_counts()
z


# In[ ]:


print('Totally there are',len(z),'different reasons why Indians got affected')


# In[ ]:


print('Current Status of these patients:\n',temp4_df['Current Status'].value_counts())


# In[ ]:


temp4_df['Current Status'].value_counts().plot.bar()
plt.ylabel('Total number')
plt.title('Current Status of Indian patients')


# Large number of Indian patients are Hospitalized, few are already recovered 

# In[ ]:


temp5_df = df.loc[df['Current Status']=='Deceased']
temp5_df['Age Bracket'].value_counts().plot.bar()
plt.xlabel('Age Bracket')
plt.ylabel('Total no. of people Deceased')
plt.title('Plot representing Deceased no. of Patients ')


# Older People in the age group above 60 are the highest to be Deceased

# In[ ]:


temp6_df = df.loc[df['Current Status']=='Recovered']
temp6_df['Age Bracket'].value_counts().plot.bar()
plt.xlabel('Age Bracket')
plt.ylabel('Total no. of people recovered')
plt.title('Plot representing recovery of Patients ')


# In[ ]:


temp7_df= df[['Age Bracket','Notes','Current Status']]
temp7_df


# In[ ]:


print('Details of people travelled from Wuhan\n',temp7_df.loc[temp7_df['Notes']=='Travelled from Wuhan'])


# In[ ]:


males = df[df['Gender']=='M']
females = df[df['Gender']=='F']

m=males['Current Status'].value_counts()/len(males)*100
f=females['Current Status'].value_counts()/len(females)*100
print('Percentage of Current Status of Male patients in India: \n',m)
print('\n')
print('Percentage of Current Status of Female patients in India: \n',f)


# In[ ]:


m.plot(kind='bar')
plt.xlabel('Percentage')
plt.title('Percentage of Current Status of Male patients in India:')


# In[ ]:


f.plot(kind='bar')
plt.xlabel('Percentage')
plt.title('Percentage of Current Status of Female patients in India:')


# # MAHARASHTRA

# In[ ]:


## Lets look for which districts are most affected in Maharashtra

temp_df=df.loc[df['Detected State']=='Maharashtra']
temp_df


# In[ ]:


print('Totally',len(temp_df),'people in Maharashtra got affected')


# In[ ]:


temp_df['Notes'].value_counts()


# In[ ]:


temp_df.replace({'Details Awaited': 'Details awaited'},inplace = True)
temp_df['Notes'].value_counts()


# In[ ]:


print('There is no clear information about how the virus affected so effectively in Maharashtra')
print('There are',len(temp_df['Notes'].value_counts()),'different number of reasons for infection')


# In[ ]:


temp_df['Detected District'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.title('Count in Districts of Maharashtra')


# In[ ]:


print('Mumbai District of Maharashtra with',max(temp_df['Detected District'].value_counts()),'is the most affected')


# In[ ]:


## Lets look for which city are most affected in Maharashtra

temp_df['Detected City'].value_counts().plot(kind='bar')
plt.ylabel('Count')
plt.title('Count in Cities of Maharashtra')


# In[ ]:


print('Mumbai City area with',max(temp_df['Detected City'].value_counts()),'is the most affected in Maharashtra')


# # KARNATAKA

# In[ ]:


## Lets look for which districts are most affected in Karnataka

temp2_df=df.loc[df['Detected State']=='Karnataka']
print('Details of patients in Karnataka')
temp2_df


# In[ ]:


print('Totally',len(temp2_df),'are affected in Karnataka')


# In[ ]:


temp2_df['Detected District'].value_counts().plot(kind='bar')
plt.ylabel('Count')


# # Data Manipulation 

# In[ ]:


df.describe()


# In[ ]:


w=df.isnull().sum()
print('Huge data is yet to be obtained, there are many empty fields in the dataset\n',w)


# In[ ]:


tem = ['Date Announced','Detected City','Gender','Detected District','Detected State','Current Status','Nationality']

for i in tem:
    print('--------------********-------------')
    print(df[i].value_counts())
    


# The Dataset is well maintained and there are no mistakes in the datset relating to spelling etc. Therefore there is no need to replace any attributes in any of the columns 

# Since there is no clear information on Details of how the virus was infected and other data about the patients, it would not be a good idea to drop the rows or columns since there will be loss of data.
# 

# # We are doing well so far #STAYHOME #STAYSAFE

# In[ ]:




