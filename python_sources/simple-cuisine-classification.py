#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import os
print(os.listdir("../input"))
from sklearn.metrics import accuracy_score


# In[ ]:


test=pd.read_json("../input/test.json")
train=pd.read_json("../input/train.json")


# In[ ]:


print("Train",train.shape)
print("Test",test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### ** Checking for commom data present in two dataframes**

# In[ ]:


pd.merge(train,test,on='id',how='inner').shape[0]


# ### Transformation and Modeling

# In[ ]:


clf=Pipeline([('countV',CountVectorizer()),
          ('tf',TfidfTransformer()),
          ('lg',LogisticRegression())])


# In[ ]:


train['ingredients_new']=train['ingredients'].apply(lambda x:' '.join(x))


# In[ ]:


clf.fit(train['ingredients_new'],train['cuisine'])


# ### accuracy_score

# In[ ]:


accuracy_score(clf.predict(train.ingredients_new),train.cuisine)


# In[ ]:


test.head()


# ### **Submission file**

# In[ ]:


test['ing_new']=test.ingredients.apply(lambda x:' '.join(x))


# In[ ]:


type(test[['id']])


# In[ ]:



pre=pd.DataFrame(clf.predict(test['ing_new']),columns=['cuisine'])
pd.concat([test[['id']],pre],axis=1).to_csv('sub.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




