#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Context
# 
# Although this dataset was originally contributed to the UCI Machine Learning repository nearly 30 years ago, mushroom hunting (otherwise known as "shrooming") is enjoying new peaks in popularity. Learn which features spell certain death and which are most palatable in this dataset of mushroom characteristics. And how certain can your model be?
# 
# ## Content
# 
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.
# 
# 
# ## Inspiration
# 
#  What types of machine learning models perform best on this dataset?
# Which features are most indicative of a poisonous mushroom?
# 
# 
# ## About this file
# 
# Attribute Information: (classes: edible=e, poisonous=p)
# 
# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 
# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# 
# bruises: bruises=t,no=f
# 
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# 
# gill-attachment: attached=a,descending=d,free=f,notched=n
# 
# gill-spacing: close=c,crowded=w,distant=d
# 
# gill-size: broad=b,narrow=n
# 
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 
# stalk-shape: enlarging=e,tapering=t
# 
# stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# 
# stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# veil-type: partial=p,universal=u
# 
# veil-color: brown=n,orange=o,white=w,yellow=y
# 
# ring-number: none=n,one=o,two=t
# 
# ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 
# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 
# population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 
# habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data=pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')


# In[ ]:


data.sample(10)


# In[ ]:


data.shape


# In[ ]:


data.isna().any()


# In[ ]:


y=data['class']
X=data.drop(['class'],axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


label_encoder=LabelEncoder()
X=X.apply(lambda x:label_encoder.fit_transform(x))
y=label_encoder.fit_transform(y)


# In[ ]:


X.head()


# In[ ]:


y


# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import SGDClassifier
classifier=SGDClassifier(alpha=0.01)
rfe=RFE(classifier,step=1,n_features_to_select=10)
rfe.fit(X,y)


# In[ ]:


rfe.ranking_


# In[ ]:


X=X.iloc[:,[3,5,6,7,9,10,11,12,16,17,19,20,21]]


# In[ ]:


X


# In[ ]:


one_hot_encoder=OneHotEncoder(sparse=True,drop='first')
X=one_hot_encoder.fit_transform(X)


# In[ ]:


X=X.toarray()


# In[ ]:


X.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=10,min_samples_leaf=100)
clf.fit(X_train,y_train)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.svm import SVC
clf1=SVC(C=0.1)
clf1.fit(X_train,y_train)


# In[ ]:


y_pred=clf1.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.linear_model import SGDClassifier
clf2=SGDClassifier(alpha=0.01)
clf2.fit(X_train,y_train)


# In[ ]:


y_pred=clf2.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf3=KNeighborsClassifier(n_neighbors=5)
clf3.fit(X_train,y_train)


# In[ ]:


y_pred=clf3.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




