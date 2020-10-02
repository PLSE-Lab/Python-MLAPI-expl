#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sub = pd.read_csv('/kaggle/input/herbarium-2020-fgvc7/sample_submission.csv')
sub.head()


# In[ ]:


sub.to_csv('bottom.csv', index=False)


# In[ ]:


import json, codecs
with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    train_meta = json.load(f)
    
with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',
                 encoding='utf-8', errors='ignore') as f:
    test_meta = json.load(f)


# In[ ]:


test_df = pd.DataFrame(test_meta['images'])
test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']


# In[ ]:


test_df.head()


# In[ ]:


sub = pd.DataFrame()
sub['Id'] = test_df.image_id
sub['Predicted'] = list(map(int, np.random.randint(1, 32000, (test_df.shape[0]))))

sub.to_csv('submission.csv', index=False)


# ## Fin.
