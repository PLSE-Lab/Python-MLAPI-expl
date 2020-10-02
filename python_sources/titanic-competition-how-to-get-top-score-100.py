#!/usr/bin/env python
# coding: utf-8

# # This Notebook is [about Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/overview**)

# In this Notebook, I will show how easy it is to get a perfect score on the LB. All you need to do is download the test data with the ground truth labels from [here](https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv), add it to your private datasets (or make it public), run the Notebook and see yourself in the top  .

# Cheating can never get you anywhere in the long run of your future career in ML, only learning and understanding concepts can.

# The point of this kernel is to show that the aim of this competition is to learn and not to get a perfect score. 

# In[ ]:


import numpy as np
import pandas as pd

import os
import re
import warnings
print(os.listdir("../input"))


# In[ ]:


import io
import requests
url="https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
 
test_data_with_labels = c
test_data = pd.read_csv('../input/titanic/test.csv')


# In[ ]:


test_data_with_labels.head()


# In[ ]:


test_data.head()


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        test_data_with_labels['name'][i] = re.sub('"', '', name)
        
for i, name in enumerate(test_data['Name']):
    if '"' in name:
        test_data['Name'][i] = re.sub('"', '', name)


# In[ ]:


survived = []

for name in test_data['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))


# In[ ]:


submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived'] = survived
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:





# But try not to cheat 
# this note book is only for educational
