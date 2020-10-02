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
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("/kaggle/input/mobile-price-classification/train.csv")
data.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.countplot(data=data,x="price_range")


# ### Dataset is well balanced

# In[ ]:


sns.lmplot(data=data,x="clock_speed",y="ram",hue="price_range")


# ### Plot has some information but it's too crowded.

# In[ ]:


sns.countplot(data=data,x="wifi")


# In[ ]:


correlation=data.corr()
plt.figure(figsize=(16, 16))
sns.heatmap(correlation,annot=True)


# ### Ram and Price Range has strong correlation
# This map basically says, Bigger screen phones and bigger ram size size phones is has more price.

# 

# ### Create Data with Ram, px_height,px_width

# In[ ]:


x=data[["ram","px_height","px_width"]]
y=data["price_range"]
x.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rfc=RandomForestClassifier(max_depth=2)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
cross_val_score(rfc,x,y,cv=10)


# ### Can't Work with enough quality.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
cross_val_score(dtc,x,y,cv=10)


# ## Still can't have enough Accuracy

# ### We should use feature selection techniques

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector=SelectKBest(chi2,k=5)
allx=data.loc[:, data.columns != 'price_range']
x_best=selector.fit_transform(allx,y)


# In[ ]:


x_best.shape


# In[ ]:


rfc=RandomForestClassifier(max_depth=2)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
cross_val_score(rfc,x_best,y,cv=10)


# In[ ]:


dtc=DecisionTreeClassifier()
cross_val_score(dtc,x_best,y,cv=10)


# ### Decision Tree work Very good 

# In[ ]:


rfc=RandomForestClassifier(max_depth=2)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
cross_val_score(rfc,x_best,y,cv=5)


# ### Use Decision Tree in Data.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_best, y, test_size=0.25, random_state=1881)
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
ypred=dtc.predict(x_test)
import sklearn.metrics as metrik
print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))
print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))


# ### Works very good. What about Test Data?

# In[ ]:


test=pd.read_csv("/kaggle/input/mobile-price-classification/test.csv")
test.head()


# In[ ]:


test=data.loc[:, test.columns != 'id']
xt_best=selector.transform(test)


# In[ ]:


predictions=dtc.predict(xt_best)


# In[ ]:


set(predictions)


# In[ ]:


sifir=0
bir=0
for pred in predictions:
    if pred==0:
        sifir=sifir+1
    else:
        bir=bir+1
print("zero: "+str(sifir))
print("one:"+str(bir))


# ### With test data can't work well. It's not deployable.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_best, y, test_size=0.25, random_state=1881)
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
ypred=rfc.predict(x_test)
print(metrik.accuracy_score(y_true=y_test,y_pred=ypred))
print(metrik.confusion_matrix(y_true=y_test,y_pred=ypred))


# In[ ]:


predictions2=rfc.predict(xt_best)
print(set(predictions))


# In[ ]:


sifir=0
bir=0
for pred in predictions2:
    if pred==0:
        sifir=sifir+1
    else:
        bir=bir+1
print("zero: "+str(sifir))
print("one:"+str(bir))


# ### Maybe This test dataset has only 35 to 50 1 and no zeros but I don't think so.

# So Decision Tree is work better in Small dataset but medium dataset Decision Tree Lose his effectivness. 
# #### Thanks
