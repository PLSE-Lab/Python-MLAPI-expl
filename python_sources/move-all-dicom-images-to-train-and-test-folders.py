#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shutil
import os
import glob2


# In[ ]:


train_path = '../input/train/'
test_path = '../input/test/'


# In[ ]:


for filename in glob2.glob('../input/dicom-images-train/**/*.dcm'):
    fname = str(filename).split('/')[-1]
    print(fname)
    shutil.copy(str(filename), os.path.join(train_path, fname))


# In[ ]:


for filename in glob2.glob('../input/dicom-images-test/**/*.dcm'):
    fname = str(filename).split('/')[-1]
    print(fname)
    shutil.copy(str(filename), os.path.join(test_path, fname))

