#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df=pd.read_csv('/kaggle/input/titanic/train.csv')
df


# loaded data into dataframe.
# Further analysing who has the better chance of survival.

# # Defining the data
# ### Name:
# Just name of the passenger
# 
# ### Passengerid:
# Id given to every individual passenger
# 
# ### Survived:
# 0=Didnt Survive, 1=Survived
# 
# ### Pclass:
# 1=1st class Passenger
# 2=2nd class Passenger
# 3=3rd class Passenger
# 
# ### Sex:
# Male or Female
# 
# ### Age:
# In years
# 
# ### SibSp:
# Number of siblings on board of that particular passenger
# 
# ### Parch:
# Number of parents or children on board of that particular passenger
# 
# ### Ticket:
# Ticket number
# 
# ### Fare:
# Amount the ticket is purchased for
# 
# ### Cabin:
# Cabin number the passenger is in
# 
# ### Embarked:
# City the passenger is from. S=SouthHampton, C=Cherbourg, Q=Queenstown.

# We are predicting the survival and here 'Name' doesnt Matter, Most of the cabin entries are 'NAN', Ticket numbers are of no use here.
# 
# they are of no use here, we can drop them.

# In[ ]:


df2=df.drop(['Name','Ticket','Cabin'],axis='columns')
df2


# lets see how every thing is correlated with Survival

# In[ ]:


def Correlation_with_survived(data):
    for i in data:
        print (df2[[i,'Survived']].groupby(i).describe())
    return 0


# In[ ]:


Correlation_with_survived(list(df2.columns[2:]))


# Here 'Age' and 'Fare' column is not very well defined and as we are grouping them by 'Survived' which has only 2 formats (1 and 0) so the mean is actually the percentage of people survived.

# # For 'Pclass'
# As you can see, 62% of people who were in 1st class have survived and only 24% of people who were in 3rd class have survived. So the titanic rescue team was a little parcial i'd say
# 
# # For 'Sex'
# 74% of females were saved where as only 18% of the males were saved.
# 
# # For 'SibSp' and 'Parch'
# These factors doesnt seem to have much affect on survival.
# 
# # For 'Fare'
# Now it also looks like it doesnt matter in survival but 3 people who paid the most (512$), all 3 of them survived. other than that, fare doesnt affect much on survival.
# 
# # For 'Embarked'
# Most of the people were from SouthHampton, only 33% of them survived but thats still around 225 people, where as 55% of people who embarked from Cherbourg survived, which is around 90 people.

# SibSp and Parch are basically the family members on board so lets make them a single column.
# 
# FOB=Family on board= siblings + parents/children

# In[ ]:


FOB=df2['SibSp']+df2['Parch']
df2['FOB']=FOB
df3=df2.drop(['SibSp','Parch'],axis='columns')
df3


# In[ ]:


sns.factorplot(x='Sex',y='Age',hue="Survived",data=df3,kind='swarm')


# As it shows here, most of the people on board are aged 18-50 and most of the females survived the crash where as most of the men did not.

# In[ ]:


sns.factorplot(x='Sex',y='Fare',hue="Survived",data=df3,kind='swarm')


# It doesnt look much of a factor here but it looks like the lesser the fare value was the more people died. 

# In[ ]:


sns.factorplot(x='Sex',y='Age',hue="Survived",data=df3,kind='swarm',col='Pclass')


# Most the people saved were from 1st class, out of which most were females. 3rd class has the most people but most of them didnt survive.

# In[ ]:


sns.factorplot(x='Sex',y='Age',hue="Survived",data=df3,kind='swarm',col='Embarked')


# A lot of people from SouthHampton survived, thats because most of the people were from SouthHampton,although the survival percentage is more of Cherbourg.

# In[ ]:


sns.factorplot(x="FOB",data=df3,kind='count',col='Survived',hue='Sex')


# This plot is basically showing that more female survived and FOB doesnt matter much for the survival.

# ### So in conclusion, i would like to say that if the person whose:
# ### Sex=Female, Age=18-50, Embarked= Cherbourg
# then you have high chance of survival. and also if you paid a lot of money for the ticket, then maybe you'd have survived(unlikely) and also there are always exception like almost all of the children who are younger than 5 yrs old survived.
# 
# ## At the end, what you do at the moment and place of crisis matters the most i think.
# # Thankyou
