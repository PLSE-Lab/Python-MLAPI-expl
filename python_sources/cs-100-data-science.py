#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# # Loading and examining data

# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train_data[:10]


# In[ ]:


train_data.describe(include='all')


# # Basic analysis

# In[ ]:


women = train_data[train_data['Sex'] == 'female']['Survived']
rate_women = sum(women)/len(women)
print('% of women who survived:', rate_women)


# In[ ]:


men = train_data[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print('% of men who survived:', rate_men)


# In[ ]:


# alternative way of computing the above
train_data[['Sex', 'Survived']].groupby(['Sex']).mean()


# In[ ]:


train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# In[ ]:


# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)
train_data.corr()


# # Working with rows manually

# In[ ]:


women_count = 0
women_survived_count = 0
for idx, row in train_data.iterrows():
    if row['Sex'] == 'female':
        women_count += 1
        if row['Survived'] == 1:
            women_survived_count += 1
    
        
        
women_survived_count / women_count


# # Making predictions

# In[ ]:


predictions = []
for idx, row in test_data.iterrows():  
    if row ['Sex'] == 'female' and row ['Age']>= 1:
        predictions.append(1)
    else:
        predictions.append(0)
   
   
   
    


# In[ ]:


assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'


# In[ ]:


test_data['Survived'] = predictions


# In[ ]:


test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)

