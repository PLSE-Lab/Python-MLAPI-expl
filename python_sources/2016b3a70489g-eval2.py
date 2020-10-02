#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train_df=pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')


# In[ ]:


train_df.head()


# In[ ]:


#train_df = train_df[train_df['class']!=6]
#train_df = train_df[train_df['class']!=5]
#train_df = train_df[train_df['class']!=7]


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

corr = train_df.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


test_df=pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression  
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from imblearn.ensemble import BalancedRandomForestClassifier


# In[ ]:


#X_train = train_df.drop(['chem_2','chem_3','chem_7','attribute','class'],axis=1)
#X_train = train_df.drop(['id','class'],axis=1)
X_train=train_df[['chem_1','chem_2','chem_4','chem_6']]


# In[ ]:


y_train = train_df['class']


# In[ ]:


y_train.value_counts()


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


#clf=LogisticRegression(multi_class='multinomial',solver='newton-cg',class_weight='balanced',max_iter=50)
clf=RandomForestClassifier(n_estimators=2000,n_jobs=-1,max_depth=9)
#clf=XGBClassifier()
#clf=DecisionTreeClassifier(max_depth=15)
#clf = lgb.LGBMClassifier(max_depth=5,objective='multiclassova',num_class=4,learning_rate=0.1,n_estimators=50)
#clf=MultinomialNB()
#clf=BalancedRandomForestClassifier(n_estimators=50,max_depth=3,random_state=42)
#clf=ExtraTreesClassifier()


# In[ ]:


scores = cross_val_score(clf, X_train, y_train, cv=3)


# In[ ]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


#X_test=test_df.drop(['id'],axis=1)
X_test=test_df[['chem_1','chem_2','chem_4','chem_6']]


# In[ ]:


clf.fit(X_train,y_train)
preds=clf.predict(X_test)


# In[ ]:


sub=pd.read_csv('/kaggle/input/eval-lab-2-f464/sample_submission.csv')


# In[ ]:


sub = sub[0:0]


# In[ ]:


sub['id'] = test_df['id']


# In[ ]:


sub['class'] = preds


# In[ ]:


sub.head()


# In[ ]:


sub['class'].value_counts()


# In[ ]:


sub.to_csv('sub18.csv',index=False)


# In[ ]:


train_df = train_df[train_df['class']!=6]
train_df = train_df[train_df['class']!=5]


# In[ ]:


clf_2=RandomForestClassifier(n_estimators=2000,n_jobs=-1,max_depth=9).fit(X_train,y_train)


# In[ ]:


scores_2 = cross_val_score(clf_2, X_train, y_train, cv=3)


# In[ ]:


print("Accuracy: %0.2f (+/- %0.2f)" % (scores_2.mean(), scores_2.std() * 2))


# In[ ]:


#clf_2.fit(X_train,y_train)
preds_2=clf_2.predict(X_test)


# In[ ]:


sub['class'] = preds_2


# In[ ]:


sub.to_csv('sub20.csv',index=False)


# In[ ]:




