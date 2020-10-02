#!/usr/bin/env python
# coding: utf-8

# ## Solving MNIST problem with SVM
# 
# ## https://satya-python.blogspot.com/

# In[ ]:


#importing libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

import seaborn as sns


# In[ ]:


# Reading Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# **EDA**

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()


# In[ ]:


#train["label"].value_counts().sort_index()
sns.countplot(train["label"])


# In[ ]:


pixels = train.drop(["label"], axis=1)
target = train["label"]


# In[ ]:


pixels = pixels/255.0


# **Train & Test split**

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(pixels, target, test_size=0.25, random_state=2019)


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# **Model creation - SVM**

# In[ ]:


mdl = SVC(C=400, kernel='rbf', random_state=2019, gamma="scale", verbose=True)


# In[ ]:


mdl.fit(x_train, y_train)


# In[ ]:


predicted = mdl.predict(x_val)
predicted


# In[ ]:


print("accuracy", metrics.accuracy_score(y_val, predicted))    # accuracy


# In[ ]:


sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_val, predicted)), annot=True, cmap="YlGn", fmt='g')


# In[ ]:


test = test/255.0


# In[ ]:


y_pred = mdl.predict(test)


# In[ ]:


submission = {}
submission['ImageId'] = range(1,28001)
submission['Label'] = y_pred
submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Label']]
submission = submission.sort_values(['ImageId'])
submission.to_csv("submisision.csv", index=False)
print(submission['Label'].value_counts().sort_index())


# In[ ]:




