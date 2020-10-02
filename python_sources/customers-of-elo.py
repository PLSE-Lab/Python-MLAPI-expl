#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION 
# @ Team winner_winner_chicken_dinner <br/>
# [Smitha_Acha](https://www.kaggle.com/smithaachar)r and [Arjun_Singh ](https://www.kaggle.com/arjunsinghyadav2)
# ## ELO Is trying to predict the Customer Loyalty Score
# ## We will analyze the data here and try to come up with a strategy to use. 

# In[ ]:


from IPython.display import HTML

HTML('''
<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>
''')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load 
import matplotlib.pyplot as plt                   #For graphics
import seaborn as sns #For better looking graphics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading the Train data
train = pd.read_csv('../input/train.csv')


# # Training data overview

# In[ ]:


train.head()


# ### Total unique Customers

# In[ ]:


# Total number of unique customers
train.card_id.nunique()


# In[ ]:


# Getting the right format of date
train['first_active_month'] = pd.to_datetime(train['first_active_month'],format='%Y%m',infer_datetime_format=True)


# ### Data Types

# In[ ]:


# Checking the data type
train.info()


# In[ ]:


# Converting features to String type
train['feature_1'] = train['feature_1'].apply(lambda x:str(x))
train['feature_2'] = train['feature_2'].apply(lambda x:str(x))
train['feature_3'] = train['feature_3'].apply(lambda x:str(x))


# ## How has been the popularity of ELO across the years?

# In[ ]:


temp_series = train[['first_active_month','card_id']].groupby('first_active_month').aggregate({'card_id':'count'}).reset_index()
plt.plot(temp_series['first_active_month'],temp_series['card_id'])
plt.xlabel(('Time'))
plt.ylabel(('Count of Customers'))
plt.title('Popularity over time for ELO products')
plt.show()


# ### The number of customers have grown exponentially since the loyalty program started in the end of 2011
# #### 2018 hardly has any data, for now it maybe a good idea to drop it completely, we can come back it later.

# ## How has the outlook of all new customers been towards ELO products?

# In[ ]:


temp_series1 = train[['first_active_month','target']].groupby('first_active_month').aggregate({'target':'sum'}).reset_index()
plt.plot(temp_series1['first_active_month'],temp_series1['target'])
plt.xlabel('Time')
plt.ylabel('customers loyalty score')
plt.title('Outlook of ELO Products')
plt.show()


# ### We can see that it has been fairly negative, the uplift in 2018 is almost a false flag as this could be totally due to small sample with mostly positive score.

# # Type of Cards
# ### The train and test data has three loyalty card features
# 
# ### Feature 1 | Feature 2 | Feature 3
# ### The feature are annonamized but they need to be infered in oder to understand loyalty programs

# ## Two type of Feature 3 cards 
# ***To make things easy to relate to, I like to name stuff<br/>
# So from now on we will use following terminology<br/>
# Feature 3 - Label 1 is our Economy Card<br/>
# Feature 3 - Label 0 is our Premium Card<br/>***

# In[ ]:


temp_series4 = train[['feature_3','card_id']].groupby('feature_3').aggregate({'card_id':'count'}).reset_index()
temp_series4 = temp_series4.sort_values('card_id',ascending=False)
plt.bar(temp_series4['feature_3'],temp_series4['card_id'])
plt.xlabel(('Economy Card    &     Premium Card'))
plt.ylabel(('Count of unique card id'))
plt.title('Two type in Feature 3')
plt.show()


# ## Card feature 1 is of 5 category where 3 being the most popular label
# ## I wonder if feature 1 category are related to feature 3 in some way

# In[ ]:


temp_series2 = train[['feature_1','card_id']].groupby('feature_1').aggregate({'card_id':'count'}).reset_index()
temp_series2 = temp_series2.sort_values('card_id',ascending=False)
plt.bar(temp_series2['feature_1'],temp_series2['card_id'])
plt.xlabel(('Type of Feature 1'))
plt.ylabel(('count of card id'))
plt.title('Distribution of Feature 1')
plt.show()


# In[ ]:


# Where feature 3 is Economy what is feature 1
train[train['feature_3']=='1'].feature_1.unique()


# ## So Economy card has two categories
# ***Feature 1 - Label 5 - Copper<br/>
# Feature 1 - Label 3 - Bronze***

# In[ ]:


## And lets check Premium card for Feature 1 labels 
train[train['feature_3']=='0'].feature_1.unique()


# In[ ]:


# Frequncy of each feature 1 in Premium Cards
train[train['feature_3']=='0'].feature_1.value_counts()


# ## Premium Card has 3 Features
# *** Feature 1 - Label 1 - Platinum<br/> 
# Feature 1 - Label 2 - Silver<br/>
# Feature 1 - Label 3 - Gold***

# In[ ]:


temp_series3 = train[['feature_2','card_id']].groupby('feature_2').aggregate({'card_id':'count'}).reset_index()
temp_series3 = temp_series3.sort_values('card_id',ascending=False)
plt.bar(temp_series3['feature_2'],temp_series3['card_id'])
plt.xlabel('Silver      Gold       Platinum')
plt.ylabel('Count of card id')
plt.title('Premium Card Type in Feature 1')
plt.show()


# ### Feature 2 has 3 type of labels 1 is most popular type

# ## For Economy vs Premium Card 
# ## Let's visualize how the distribution has been for feature 2

# In[ ]:


# Frequncy of each feature 2 in Economy Cards
eco_fe2 = train[train['feature_3']=='1'].feature_2.value_counts().reset_index().rename(columns={'feature_2':'Economy'})
# Frequncy of each feature 2 in Premium Cards
prem_fe2 = train[train['feature_3']=='0'].feature_2.value_counts().reset_index().rename(columns={'feature_2':'Premium'})


# In[ ]:


#merge the two
eco_prem = pd.merge(eco_fe2,prem_fe2,on='index',how='inner')


# In[ ]:


ind = eco_prem.index.tolist()  # the x locations for the groups

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

yvals = eco_prem.Economy.tolist()
rects1 = ax.bar(ind, yvals, width= 0.25, color='r')
zvals = eco_prem.Premium.tolist()
rects2 = ax.bar(ind, zvals, width=0.25, color='g')

ax.set_ylabel('Frequency')
ax.set_xlabel('Feature_2')
ax.set_xticks(ind)
ax.set_xticklabels( ('1', '2', '3') )
ax.set_title('Feature 2 type distribution in Economy and Premium Card')
ax.legend(('Economy', 'Premium') )


# ## Feature two seems to be the only one common across all cutomers 
# So it has to be something related to the personality or difference in the consumer age<br/>
# Given label 3 in lowest frequecy:<br/>
# ***Lets say Label - 3 is Entreprenuers <br/>
# Label- 2 is Students <br/>
# Label -1 is Working Professionals***

# In[ ]:


train['year'] = train['first_active_month'].dt.year
tarin_back = train
train = train[train['year']!=2018]
# Loyalty Card behaviour by Students, Working Professionals and Entreprenuers
work_pro = train[train['feature_1']=='1'][['year','target']].groupby('year').aggregate({'target':'sum'}).reset_index()

stud = train[train['feature_1']=='2'][['year','target']].groupby('year').aggregate({'target':'sum'}).reset_index()
entre = train[train['feature_1']=='3'][['year','target']].groupby('year').aggregate({'target':'sum'}).reset_index()


# ### Finally lets look at the outlook of these three customer bases differently throughout the history of ELO

# In[ ]:


entre = entre.rename(columns={'target':'entrepreneur'})
work_pro = work_pro.rename(columns={'target':'working professional'})
stud = stud.rename(columns={'target':'student'})
entre_work = pd.merge(entre,work_pro,on='year',how='outer')
entre_work_stud = pd.merge(entre_work,stud,on='year',how='outer')
entre_work_stud.head()


# In[ ]:


entre_work_stud.set_index('year',inplace=True)


# In[ ]:


plt.figure(figsize=(15,7))
plt.xlabel('Target for New Joins by Feature 2')

ax1 = entre_work_stud['working professional'].plot(color='blue', grid=True, label='Working Professional')
ax2 = entre_work_stud['student'].plot(color='red', grid=True, secondary_y=True, label='Student')
ax3 = entre_work_stud['entrepreneur'].plot(color='green', grid=True, secondary_y=True, label='Entrepreneur')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()

plt.legend(h1+h2, l1+l2, loc=2)
plt.show()


# ### The Working Professionals have been quite negative about ELO service compared to Student and Entrepreneurs, Ideally we should treat them in two seperate groups and build seperate models.

# In[ ]:




