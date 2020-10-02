#!/usr/bin/env python
# coding: utf-8

# Here is code to randomly shuffle every ad_id list in submission file

# In[ ]:


import numpy as np
import pandas as pd
import zipfile


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission.head()


# In[ ]:


def getShuffledList(ad_id_string):
    tmp = np.array(ad_id_string.split())
    np.random.shuffle(tmp)
    return ' '.join(tmp)


# In[ ]:


submission['ad_id'] = submission['ad_id'].apply(lambda x: getShuffledList(x))


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission_shuffled.csv', index=False)


# In[ ]:


with zipfile.ZipFile('submission_shuffled.csv.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
    myzip.write('submission_shuffled.csv')


# In[ ]:




