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


import matplotlib.pyplot as plt


# First read the file

# In[ ]:


trial_0=pd.read_csv('../input/labeledEligibilitySample1000000.csv')


# For some reason, read_csv doesnt work....something about the format is wrong, so try another way

# In[ ]:


trial=pd.read_table('../input/labeledEligibilitySample1000000.csv', header=None)


# read_table works fine. Let's look at some basic information

# In[ ]:


trial.shape


# One million......not dollars, but lines! only two columns?

# In[ ]:


trial.sample(10)


# According to the data description, one column is the label of qualification, the other just a short statement about the trial. Let's give them correct names.

# In[ ]:


clin_trial = pd.DataFrame(np.array(trial).reshape(1000000,2), columns=['label', 'describe'])


# In[ ]:


clin_trial['describe'].head(10)


# look carefully at the 'describe' column, there is a '.' in every statement to separate treatment of the trial from other description. how many kinds of values in 'label' column?

# In[ ]:


clin_trial['label'].unique()


# good! only two. then split the statement into two parts based on the dot '.'

# In[ ]:


clin_trial['study'], clin_trial['condition'] = clin_trial['describe'].str.split('.', 1).str


# In[ ]:


clin_trial['study'].unique()


# In[ ]:


clin_trial.head(10)


# In[ ]:


clin_trial=clin_trial.drop(['describe'], axis=1)


# Now the split columns are created. Let's make the label column more straight forward.: just 0 and 1

# In[ ]:


clin_trial['qualification']=clin_trial['label'].str.extract('(\d)', expand=True)


# In[ ]:


clin_trial=clin_trial.drop(['label'], axis=1)


# In[ ]:


clin_trial.sample(5)


# now the table seems clean enough. How many different trials are there, even though total number is one million?

# In[ ]:


clin_trial['study'].value_counts()


# totally there are more than 15,000 trials in the table. Some are similar trials with minor changes on names.

# So how can we look at these data reasonably? One way is to categorize into different cancer types, like lymphoma, breast cancer...
# Let's try select all the trials on lymphoma

# In[ ]:


clintrial_lymphoma=clin_trial.loc[clin_trial.condition.str.contains('\w*lymphoma')]


# In[ ]:


clintrial_lymphoma.shape


# There are nearly 230,000 trials of lymphoma! How about breast cancer?  

# In[ ]:


clintrial_breast=clin_trial.loc[clin_trial.condition.str.contains('.*reast')] # to avoid lower case and upper case error


# # Test start (D.M.)

# In[ ]:





# In[ ]:


unique_breast = clintrial_breast['condition'].unique()


# In[ ]:


print("breast unique: ", len(unique_breast))


# # Test End (D.M.)

# In[ ]:


clintrial_breast.shape


# About >10,000 trials on breast cancer. Then let's use lymphoma data as an example to do the analysis

# In[ ]:


clintrial_lymphoma['study'].value_counts()


#     The most popular treatment is Fludarabine, different from the total cancer dataset.

# let's find some keywords from the description column.

# In[ ]:


clintrial_lymphoma['words'] = clintrial_lymphoma.condition.str.split(' ')


# In[ ]:


clintrial_lymphoma.head(3)


# Now we have all the words singled out to count.

# In[ ]:


rows = list()
for row in clintrial_lymphoma[['study', 'words']].iterrows():
    r = row[1]
    for word in r.words:
        rows.append((r.study, word))

words = pd.DataFrame(rows, columns=['study', 'word'])


# In[ ]:


words.head(10)


# In[ ]:


words['word'] = words.word.str.lower()


# See which words are the most popular keywords

# In[ ]:


words['word'].value_counts().head(50)


# From the list,  we can see most of them are useless, like 'and', 'the'...but some are very important and informative for cancer treatment, like 'recurrent', 'stage_ii', 'stage_iii', 'follicular' and 'diffuse'.
# 
# So let's take a look at the relationship between keywords and 'study'
# 
# First let's make some columns to store the keywords information.

# In[ ]:


clintrial_lymphoma['recurrent'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*recurrent"), "recurrent","no")


# In[ ]:


clintrial_lymphoma.sample(10)


# In[ ]:


clintrial_lymphoma['stage_ii'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*stage ii"), "stage_ii","no")


# In[ ]:


clintrial_lymphoma['stage_iii'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*stage iii"), "stage_iii","no")


# In[ ]:


clintrial_lymphoma['stage_iv'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*stage iv"), "stage_iv","no")


# In[ ]:


clintrial_lymphoma['follicular'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*follicular"), "follicular","no")


# In[ ]:


clintrial_lymphoma['diffuse'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*diffuse"), "diffuse","no")


# In[ ]:


clintrial_lymphoma['hodgkin'] = pd.np.where(clintrial_lymphoma.condition.str.contains("\w*hodgkin"), "hodgkin","no")


# In[ ]:


clintrial_lymphoma.sample(50)


# In[ ]:


clintrial_lymphoma_select=clintrial_lymphoma.drop([ 'words'], axis=1)


# In[ ]:


import seaborn as sns


# In[ ]:





# In[ ]:


var = clintrial_lymphoma_select.groupby(['study']).qualification.value_counts()


# In[ ]:


var.shape


# In[ ]:


var.head(10)


# In[ ]:


var_q = clintrial_lymphoma_select.groupby('study')['qualification'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_q


# In[ ]:


var_r = clintrial_lymphoma_select.groupby('study')['recurrent'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_r


# most treatments cannot be applied to reccurent cases

# In[ ]:


var_iv = clintrial_lymphoma_select.groupby('study')['stage_iv'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_iv


# Stage four cancers have almost no popular treatments

# In[ ]:


var_ii = clintrial_lymphoma_select.groupby('study')['stage_ii'].value_counts().sort_values(ascending=False).head(50).plot(kind='bar', figsize=(15,5))
var_ii


# Stage ii cancers have much more ways to treat.

# In[ ]:



var1 = clintrial_lymphoma_select.groupby(['study']).qualification.value_counts()

var1.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], figsize=(15,6))


# The last graph gives an overview on the relatioship of studies and qualification. 
# 
# 

# With the text words, prediction based on the 'qualification' cannot be done. All the words have to be changed to number first. However after I tried, none of the  models I used gave an accuracy number bigger than 0.6. So more work need to be done on the vocabulary analysis.
# 
# 

# In[ ]:




