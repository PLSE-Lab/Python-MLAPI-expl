#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# In[ ]:


# only keep column PassengerId
submission_data = test_data.copy()[['PassengerId']]
# add columned Survived with default value 0 (False)
submission_data['Survived'] = 0


# In[ ]:


submission_data.to_csv('submission.csv', index=False)

