#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.vision import *


# In[ ]:



data=ImageDataBunch.from_folder(path=Path('../input/chest_xray/chest_xray'),train="train",valid="val",test="test",size=512)


# In[ ]:


data.show_batch()


# In[ ]:


learner=cnn_learner(data,models.resnet18,metrics=accuracy)
learner.fit(5)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
interp.plot_top_losses(9, figsize=(10,10))


# In[ ]:


interp.plot_confusion_matrix()


# Validation data is too small

# In[ ]:


interp.plot_multi_top_losses()


# In[ ]:


ytrue=interp.y_true
ypred=interp.pred_class
from sklearn.metrics import classification_report
print(classification_report(y_true=ytrue,y_pred=ypred))


# Also only 1 false positive occurs.

# In[ ]:


one,label=learner.get_preds(ds_type=DatasetType.Test)
labels = np.argmax(one, 1)


# In[ ]:


labels


# Test is half ready

# In[ ]:


falan=data.label_list


# In[ ]:


falan.lists


# Tests Label is just empty

# In[ ]:


learner.summary()


# Okay somewhat it is work fine.

# In[ ]:




