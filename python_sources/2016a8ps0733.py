#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data1 = pd.read_csv("train_lab3.csv")


# In[ ]:


data2 = pd.read_csv("test_data.csv")


# In[ ]:


data2.info()


# In[ ]:


data1.head()


# In[ ]:


data1.describe()


# In[ ]:


data1.info()


# In[ ]:


data1.isnull().values.any()


# In[ ]:


data1.duplicated().sum()


# In[ ]:


data1['castleTowerDestroys'].unique()


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = data1.corr()
sns.heatmap(corr, center=0)


# In[ ]:


data = data1.drop(data1.columns[[0]], axis=1)
data_test = data2.drop(data2.columns[[0]], axis=1)


# In[ ]:


data_test.describe()


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data_test.info()


# In[ ]:





# In[ ]:


X=data.drop('bestSoldierPerc',axis=1)

y=data['bestSoldierPerc']
X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[ ]:


#             Decision Tree Model
#Finding optimum depth using Elbow Method
from sklearn.tree import DecisionTreeClassifier
error_rate_train = []
for i in range(1,50):
    dTree = DecisionTreeClassifier(max_depth=i)
    dTree.fit(X_train,y_train)
    pred_i = dTree.predict(X_train)
    error_rate_train.append(np.mean(pred_i != y_train))
    error_rate_test = []
for i in range(1,50):
    dTree = DecisionTreeClassifier(max_depth=i)
    dTree.fit(X_train,y_train)
    pred_i = dTree.predict(X_test)
    error_rate_test.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,50),error_rate_train,color='blue', linestyle='dashed')
test_score,=plt.plot(range(1,50),error_rate_test,color='red',linestyle='dashed')
plt.legend( [train_score,test_score],["Train Error","Test Error"])
plt.title('Error Rate vs. max_depth')
plt.xlabel('max_depth')
plt.ylabel('Error Rate')


# In[ ]:


# Fitting Decision Tree model to the Training set with max_depth 8
dTree = DecisionTreeClassifier(max_depth=100)
dTree.fit(X_train,y_train)


# In[ ]:


#   Mean Accuracy
dTree.score(X_test,y_test)


# In[ ]:


# Predicting the Test Results
y_pred=dTree.predict(X_test)


# In[ ]:


# Predicting the Test Results
y_dt=dTree.predict(data_test)
y_dt


# In[ ]:


soldierId = data_test['soldierId']
a = np.array(soldierId)
a = a.astype(str)


# In[ ]:


p = [a,y_dt]
p = p
df = pd.DataFrame(p)
df = df.transpose()
df.columns = ['soldierId','bestSoldierPerc']
df.astype(str)
df.info()


# In[ ]:


df


# In[ ]:


df.to_csv("solution111.csv", index = False)


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train=pd.read_csv('train_lab3.csv')
test=pd.read_csv('test_data.csv')


# In[ ]:


train.head(10)


# In[ ]:


test.head()


# In[ ]:


train['knockedOutSoldiers'] = train['knockedOutSoldiers'].fillna(0)


# In[ ]:


train['respectEarned'] = train['respectEarned'].fillna(np.mean(train['respectEarned']))


# In[ ]:


train['respectEarned'] = round(train['respectEarned'])


# In[ ]:


train_new.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.dtypes


# In[ ]:


train.shape


# In[ ]:


train_new = train


# In[ ]:


train_new.shape


# In[ ]:


train.shape


# In[ ]:


test.head()


# In[ ]:


test_new = test


# In[ ]:


test_new.shape


# In[ ]:


test_new.head()


# In[ ]:


test_new.shape


# In[ ]:


cols = ['soldierId', 'bestSoldierPerc']
cols_test = ['soldierId']


# In[ ]:


random_rows=list(np.random.random_integers(0, train.shape[0], 50000))
train_new = train.iloc[random_rows,:]


# In[ ]:


train_new.shape


# In[ ]:


X = train_new.drop(cols,axis=1)


# In[ ]:


X.shape


# In[ ]:


X_t = test.drop(cols_test,axis=1)


# In[ ]:


X_t.shape


# In[ ]:


X.head()


# In[ ]:


y = train_new['bestSoldierPerc']


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


X_train.shape


# In[ ]:


import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[ ]:


clsf = xgb.XGBRegressor(n_estimators=100, learning_rate=0.09, gamma = 0, subsample=1,
                         colsample_bytree=1, max_depth=15, min_child_weight = 1).fit(X_train, y_train, sample_weight=None)


# In[ ]:


predictions = clsf.predict(X_test)


# In[ ]:


np.mean(abs(predictions-y_test))


# In[ ]:


feature_importance=clsf.feature_importances_
len(feature_importance)
#list_nouse = []
#cols_of_value = []
for i in range(X.shape[1]):
   # if i < 25 :
    #    continue
    #if feature_importance[i] < 0.02:
     #   list_nouse.append(X.columns[i])
    #else:
     #   cols_of_value.append(X.columns[i])
    print(X.columns[i],':\t',feature_importance[i])


# In[ ]:


X_t.shape


# In[ ]:



test_new.shape


# In[ ]:


predtest = clsf.predict(X_t)


# In[ ]:


predtest


# In[ ]:



b=test.soldierId


# In[ ]:



sol=pd.DataFrame()


# In[ ]:



b.astype(float)


# In[ ]:


sol['soldierId']=b.astype(float)


# In[ ]:



pred= pd.DataFrame(predtest)


# In[ ]:



pred.head()


# In[ ]:


sol['bestSoldierPerc']=round(pred.iloc[:,0]).astype(int)


# In[ ]:


sol.head()


# In[ ]:


sol.shape


# In[ ]:



test.shape
#sol['soldierId']=sol['soldierId'].astype(str)


# In[ ]:


sol.info()


# In[ ]:


sol.to_csv('submission_file909.csv',index=False)

