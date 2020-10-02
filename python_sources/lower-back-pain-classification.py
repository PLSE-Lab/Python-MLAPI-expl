#!/usr/bin/env python
# coding: utf-8

# Author: Kazi Amit Hasan
# 
# 
# Please upvote if you like it.

# 310 Observations, 13 Attributes (12 Numeric Predictors, 1 Binary Class Attribute - No Demographics)
# 
# Lower back pain can be caused by a variety of problems with any parts of the complex, interconnected network of spinal muscles, nerves, bones, discs or tendons in the lumbar spine. Typical sources of low back pain include:
# 
# The large nerve roots in the low back that go to the legs may be irritated
# The smaller nerves that supply the low back may be irritated
# The large paired lower back muscles (erector spinae) may be strained
# The bones, ligaments or joints may be damaged
# An intervertebral disc may be degenerating
# An irritation or problem with any of these structures can cause lower back pain and/or pain that radiates or is referred to other parts of the body. Many lower back problems also cause back muscle spasms, which don't sound like much but can cause severe pain and disability.
# 
# While lower back pain is extremely common, the symptoms and severity of lower back pain vary greatly. A simple lower back muscle strain might be excruciating enough to necessitate an emergency room visit, while a degenerating disc might cause only mild, intermittent discomfort.
# 
# This data set is about to identify a person is abnormal or normal using collected physical spine details/data.

# In[ ]:


import pandas as pd
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


del data["Unnamed: 13"]


# In[ ]:


data.describe()


# In[ ]:


data.rename(columns = {
    "Col1" : "pelvic_incidence", 
    "Col2" : "pelvic_tilt",
    "Col3" : "lumbar_lordosis_angle",
    "Col4" : "sacral_slope", 
    "Col5" : "pelvic_radius",
    "Col6" : "degree_spondylolisthesis", 
    "Col7" : "pelvic_slope",
    "Col8" : "direct_tilt",
    "Col9" : "thoracic_slope", 
    "Col10" :"cervical_tilt", 
    "Col11" : "sacrum_angle",
    "Col12" : "scoliosis_slope", 
    "Class_att" : "class"}, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data["class"].value_counts().sort_index().plot.barh()


# In[ ]:


data.corr()


# In[ ]:


sns.heatmap(data[data.columns[0:13]].corr(),annot=True,cmap='viridis',square=True, vmax=1.0, vmin=-1.0, linewidths=0.2)


# In[ ]:


X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2,random_state=47)


# In[ ]:


print(X_train.shape, y_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,auc

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


# In[ ]:


import matplotlib.pyplot as plt
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=.3)
plt.show()

print(classification_report(y_test,y_pred))


# In[ ]:




