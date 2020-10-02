#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd 
from pandas import Series, DataFrame


# In[ ]:



titanic_df = pd.read_csv('../input/train.csv')


# In[ ]:


titanic_df.head(3)


# In[ ]:



titanic_df.info()


# In[ ]:



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.catplot("Sex",kind="count",data=titanic_df)


# In[ ]:


sns.catplot("Sex",kind="count",hue="Pclass",data=titanic_df)


# In[ ]:


sns.catplot("Pclass",kind="count",hue="Sex",data=titanic_df)


# In[ ]:


def male_female_child(passenger):
        age,sex = passenger
        
        if age < 16:
            return 'child'
        else:
            return sex


# * I am adding a new colum called child <font color=blue > titanic_df ['person'] </font>into my titanic dataset to view the child ratio.

# In[ ]:


titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[ ]:


titanic_df[0:10]


# In[ ]:


sns.catplot('Pclass', kind ='count', hue ='person',data=titanic_df)


# In[ ]:


titanic_df['Age'].hist(bins=70)


# In[ ]:


titanic_df['Age'].mean()


# In[ ]:


titanic_df['person'].value_counts()


# * total aboard <font color=blue> male = 537 </font>, <font color=red>female = 271 </font>and <font color=green> child = 83.</font>

# In[ ]:


fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[ ]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# 
# * In what deck were the passenger on and how does this relate to class
# 

# In[ ]:


deck = titanic_df['Cabin'].dropna()


# In[ ]:


deck.head()


# In[ ]:


levels =[]
for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']
cabin_df = cabin_df.sort_values(['Cabin'])
sns.catplot('Cabin',kind='count',data=cabin_df,palette='winter_d')


# * Removing class T which is a unnecessary value here.<br><br>

# In[ ]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot('Cabin',kind='count',data=cabin_df,palette='cubehelix')


# In[ ]:


titanic_df.head(3)


# <br><br>
# #### Where did the passenger come from ?

# <br>
# Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)
# <br><br>
# Pclass <br>
# 1st = Upper<br>
# 2nd = Middle<br>
# 3rd = Lower<br>
# <br>
# 
# Its clearly understandable from the catplot below that in <font color='green'>Queenstown</font> most of the passengers were boarded in Lower class where as  most sold upper class tickets in <font color='blue'>Southampton city.<br><br>

# * In the violin plot below we are going to see the pessengers by their age, sex and where they come from i.e. Embarked and the class they were boarded. <br><br>

# In[ ]:



sns.catplot(x="Age", y="Embarked",data=titanic_df,hue="Sex",col="Pclass", col_wrap=3,height=4, aspect=1, dodge=True, palette="Set3",kind="violin", order=['C','Q','S'])


# In[ ]:


titanic_df.head(3)


# <br>
# 
# #### Who was alone and who was with their family?
# <br>

# In[ ]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[ ]:


titanic_df['Alone'].head()


# In[ ]:


titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[ ]:


titanic_df.head(3)


# In[ ]:


sns.catplot('Alone',kind='count',data=titanic_df,palette ='summer')


# * we can visualize form the above plot the most of passengers were aboard alone. <br>

# In[ ]:


titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})


# In[ ]:


sns.catplot('Survivor',data=titanic_df,kind='count',palette='Set1')


# In[ ]:


sns.catplot('Pclass','Survived', kind = "point",data=titanic_df)


# <br>* Most survived pessengers were in Fisrt Class<br><br>

# In[ ]:


sns.catplot('Pclass','Survived', hue='person', kind = "point",data=titanic_df)


# #### Regardless of their age and boarding class male passengers was always the one who has less survival rate.<br><br>

# In[ ]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[ ]:


sns.lmplot('Age','Survived',hue='Pclass',palette='winter',data=titanic_df)


# In[ ]:


generations = [10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[ ]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)


# In[ ]:


titanic_df.head(3)


# <br>
# 
# #### Did the deck have an affect on the passengers survival rate? 
# 
# 
# <br><br><vr>

# In[ ]:


titanic_df['Deck'] = cabin_df
sns.catplot('Survivor',col='Deck',col_wrap=4,data=titanic_df[titanic_df.Deck.notnull()],kind="count",height=3.3, aspect=.9,palette='rocket')


# <br> <br>
# 
# Yes for sure from <font color= 'green'> Deck A </font> the survicor rate is higher. <br>On the otherhand if we keep going lower and lower alphabetically into the Deck we can see the less number of survivor i.e. <font color='red'> Deck G </font>
# 
# <br>

# <br>
# 
# #### We can further ask a question, did having a family members increase the odds of having the crash?
# 
# <br>

# In[ ]:


sns.catplot('Alone',kind="count",hue='Survivor',data=titanic_df,palette='rocket')


# * From this catplot we can finally analyse that passengers who were having <br>family members must had been prioritized first to get into the lifeboat.

# ![](http://www.britannica.com/topic/Titanic/media/597128/201276)
# ##### RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in 1912, after colliding with an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of modern history's deadliest commercial marine disasters during peacetime. 

# ##### The wreck of the RMS Titanic is lying 3,900 metres (2.4 mi) at the bottom of the Atlantic Ocean, almost precisely under the location where she sank at April 15, 1912. The ship broke in two pieces, which came to rest 590 metres (approx. 650 yards) separated

# In[ ]:




