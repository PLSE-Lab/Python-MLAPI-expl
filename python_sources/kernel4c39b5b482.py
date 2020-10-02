#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# to import train and test of titanic dataset from kaggle
df = pd.read_csv("/kaggle/input/titanic/train.csv")


# In[ ]:


useless = ["PassengerId","Name","Ticket","Cabin","Embarked"]
clean_data = df.drop(useless,axis=1)


# In[ ]:


le = LabelEncoder()
clean_data["Sex"] = le.fit_transform(clean_data["Sex"]) 


# In[ ]:


clean_data = clean_data.fillna(clean_data["Age"].mean())


# In[ ]:


input_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare"]
out_cols = ["Survived"]

X = clean_data[input_cols]
Y = clean_data[out_cols]


# In[ ]:


counts = np.unique(Y["Survived"],return_counts = True)


# In[ ]:


sk_tree = DecisionTreeClassifier(criterion='entropy',max_depth=5)


# In[ ]:


sk_tree.fit(X,Y)


# In[ ]:


d = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


clean_test = d.drop(useless,axis=1)


# In[ ]:


le1 = LabelEncoder()
clean_test["Sex"] = le1.fit_transform(clean_test["Sex"]) 


# In[ ]:


clean_test = clean_test.fillna(clean_test["Age"].mean())


# In[ ]:


test_x = clean_test[input_cols]


# In[ ]:


pred = sk_tree.predict(test_x)


# In[ ]:


sid = 892
with open('titanic.csv','w') as f:
    s = "PassengerId,Survived\n"
    f.write(s)
    
for p in pred:
    with open('titanic.csv','a') as f:
        s = str(sid)+','+str(p)+'\n'
        f.write(s)
        sid+=1


# In[ ]:


rf = RandomForestClassifier(n_estimators=10,criterion='entropy')


# In[ ]:


rf.fit(X,Y)


# In[ ]:


ans = rf.predict(test_x)


# In[ ]:


rf.score(X,Y)


# In[ ]:


sk_tree.score(X,Y)


# In[ ]:


sid = 892
with open('titanic1.csv','w') as f:
    s = "PassengerId,Survived\n"
    f.write(s)
    
for a in ans:
    with open('titanic1.csv','a') as f:
        s = str(sid)+','+str(a)+'\n'
        f.write(s)
        sid+=1


# In[ ]:




