#!/usr/bin/env python
# coding: utf-8

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or even more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and running Rapids locally on your own machine, then you shoudl [refer to the followong instructions](https://rapids.ai/start.html).

# In[ ]:


get_ipython().run_cell_magic('time', '', '# INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\nimport sys\n!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz\n!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz\nsys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[ ]:


import cupy, cudf, cuml
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = cudf.read_csv('/kaggle/input/titanic/train.csv')
test = cudf.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train = train.drop(columns= ['Name','Ticket','Cabin'])
test = test.drop(columns= ['Name','Ticket','Cabin'])


# In[ ]:


train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)
train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)
train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)
train['Gender'] = (train['Sex'] == 'male').astype(int)


# In[ ]:


test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)
test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)
test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)
test['Gender'] = (test['Sex'] == 'male').astype(int)


# In[ ]:


train = train.drop(columns= ['Embarked','Sex',])
test = test.drop(columns= ['Embarked','Sex',])


# In[ ]:


train.fillna(0,inplace=True)
test.fillna(0,inplace=True)


# In[ ]:


X = train.drop(columns = ['Survived'])
y = train['Survived'].astype('int32')


# In[ ]:


model = RandomForestClassifier(n_estimators = 100, max_depth = 6)
model.fit(X, y) 


# In[ ]:


yhat_train = model.predict(X, predict_model = 'CPU')
yhat_test = model.predict(test, predict_model = 'CPU') 


# In[ ]:


print(sum(y == yhat_train) / len(y)) 


# In[ ]:


submission = cudf.DataFrame({'PassengerId': test.PassengerId, 'Survived': yhat_test})
submission.to_csv('submission.csv', index = False) 


# In[ ]:




