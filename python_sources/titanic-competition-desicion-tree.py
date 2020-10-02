#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


# **Dataframe Exploration**

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
train_df.head()


# In[ ]:


data_train = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
data_test = test_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


categorical_cols = [cname for cname in data_train.columns if
                    data_train[cname].nunique() <= 10 and
                    data_train[cname].dtype == 'object'
                   ]
numerical_cols = [cname for cname in data_train.columns if
                    data_train[cname].dtype in ['int64', 'float64']
                 ]


# In[ ]:


categorical_cols, numerical_cols


# **Data Cleaning**

# In[ ]:


data_train.Age = data_train['Age'].fillna(data_train['Age'].median())
data_test.Age = data_test['Age'].fillna(data_test['Age'].median())


# In[ ]:


data_test.Fare = data_test['Fare'].fillna(data_test['Fare'].median())


# In[ ]:


data_train.dropna(subset=['Embarked'], inplace=True)


# In[ ]:


data_train = pd.get_dummies(data_train)
data_test = pd.get_dummies(data_test)


# **Training**

# In[ ]:


data_train.columns


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data_train.drop('Survived', axis=1), data_train['Survived'], test_size=0.2)


# In[ ]:


model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)


# In[ ]:


model.score(x_test, y_test) # Accuracy score


# **Prediction**

# In[ ]:


output = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':model.predict(data_test)})
output


# In[ ]:


output.to_csv('my_submission.csv', index=False)


# **Tree Display**

# In[ ]:


from io import StringIO 
import pydotplus

out = StringIO()
tree.export_graphviz(model, out_file=out)

graph = pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png('titanic.png')


# In[ ]:




