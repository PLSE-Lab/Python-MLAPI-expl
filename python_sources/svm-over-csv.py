#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import f1_score
import os
print(os.listdir("../input"))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from IPython.display import clear_output as clear
import sklearn


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()


# In[ ]:


train_df = train_df.fillna(0)
test_df = test_df.fillna(0)


# In[ ]:


train_df_sorted = train_df.sort_values('Target')
plt.plot(train_df_sorted.Target.tolist())
plt.show()


# In[ ]:


train_label = np.array(train_df_sorted.Target.tolist())


# In[ ]:


train_data = []
for i in range(len(train_df_sorted)):
    li = train_df_sorted.iloc[i]
    li = li.drop('Target')
    li = li.drop('Id').tolist()
    for _ in range(len(li)):
        try:
            float(li[_])
        except ValueError:
            if li[_] == 'yes':
                li[_] = np.exp(1)
            else:
                li[_] = 0
    li = np.array(li, dtype='float64')
    li = np.log1p(np.abs(li))
    train_data.append(li)
train_data = np.array(train_data)
print('done')


# In[ ]:


test_Id = test_df['Id'].tolist()
test_data = []
for i in range(len(test_df)):
    li = test_df.iloc[i]
    li = li.drop('Id').tolist()
    for _ in range(len(li)):
        try:
            float(li[_])
        except ValueError:
            if li[_] == 'yes':
                li[_] = np.exp(1)
            else:
                li[_] = 0
    li = np.array(li, dtype='float64')
    li = np.log1p(np.abs(li))
    test_data.append(li)
test_data = np.array(test_data)
print('done')


# In[ ]:


clf_svc = svm.SVC(C=15.0)
clf_svr = svm.SVR(C=15.0)


# In[ ]:


clf_svc.fit(train_data.clip(0,100000000)[0:8000],train_label[0:8000])


# In[ ]:


clf_svr.fit(train_data.clip(0,100000000)[0:8000],train_label[0:8000])


# In[ ]:


plt.plot(clf_svc.predict(train_data.clip(0,100000000)))
#clf_svc.predict(train_data.clip(0,100000000))


# In[ ]:


plt.plot(np.round(clf_svr.predict(train_data.clip(0,100000000))).clip(1,4))


# In[ ]:


f1_score(clf_svc.predict(train_data.clip(0,100000000)), train_label,average='macro')


# In[ ]:


f1_score(np.round(clf_svr.predict(train_data.clip(0,100000000))).clip(1,4), train_label,average='macro')


# In[ ]:


pre_svc = clf_svc.predict(test_data.clip(0,100000000))
pre_svr = np.array(np.round(clf_svr.predict(test_data.clip(0,100000000))).clip(1,4),dtype='int64')


# In[ ]:


pre_svc_df = pd.DataFrame({'Id':test_Id, 'Target':pre_svc})
pre_svr_df = pd.DataFrame({'Id':test_Id, 'Target':pre_svr})


# In[ ]:


pre_svc_df.to_csv('submission_svc.csv',index=False)
pre_svr_df.to_csv('submission_svr.csv',index=False)

