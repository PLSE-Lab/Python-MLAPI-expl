#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


cd /kaggle/input/eval-lab-1-f464-v2/


# In[ ]:


train = pd.read_csv("train.csv")


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


print (train.isnull().any(axis = 0))
# print(train.isnull().sum())


# In[ ]:


train.fillna(train.mean(), inplace = True)
train.isnull().sum()


# In[ ]:


train.type.replace('new', 1, inplace = True)
train.type.replace('old', 0, inplace = True)


# In[ ]:


features = train[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 
           'feature9', 'feature10', 'feature11','type']]
X = features.values
y=train[['rating']].values
y = y.reshape(4547)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.legend_handler import HandlerLine2D

learning_rates = np.arange(0.01,0.5,0.01)
train_results=[]
test_results=[]

for eta in learning_rates:
    regressor = GradientBoostingRegressor(learning_rate = eta)  
    regressor.fit(X_train, y_train)
    
    train_pred = regressor.predict(X_train)
    train_pred = train_pred.round()
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    train_results.append(rmse_train)
    
    y_pred = regressor.predict(X_test)
    y_pred = y_pred.round()
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    test_results.append(rmse_test)

plt.title("learning_rates rmse's")
line1,= plt.plot(learning_rates, train_results, label="Traian rmse")
line2, = plt.plot(learning_rates, test_results, label="Test rmse")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("rmse values")
plt.xlabel("learning_rates")
plt.show


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.legend_handler import HandlerLine2D

n_estimators1 = [1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800, 1200, 3000]
train_results=[]
test_results=[]

for eta in n_estimators1:
    regressor = GradientBoostingRegressor(n_estimators = eta)  
    regressor.fit(X_train, y_train)
    
    train_pred = regressor.predict(X_train)
    train_pred = train_pred.round()
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    train_results.append(rmse_train)
    
    y_pred = regressor.predict(X_test)
    y_pred = y_pred.round()
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    test_results.append(rmse_test)

plt.title("n_estimators1 rmse's")
line1,= plt.plot(n_estimators1, train_results, label="Train rmse")
line2, = plt.plot(n_estimators1, test_results, label="Test rmse")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("rmse values")
plt.xlabel("n_estimators1")
plt.show


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.legend_handler import HandlerLine2D

train_results=[]
test_results=[]
max_depths = np.linspace(1, 32, 32, endpoint=True)

for eta in max_depths:
    regressor = GradientBoostingRegressor(max_depth = eta)  
    regressor.fit(X_train, y_train)
    
    train_pred = regressor.predict(X_train)
    train_pred = train_pred.round()
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    train_results.append(rmse_train)
    
    y_pred = regressor.predict(X_test)
    y_pred = y_pred.round()
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    test_results.append(rmse_test)

plt.title("max_depths rmse's")
line1,= plt.plot(max_depths, train_results, label="Train rmse")
line2, = plt.plot(max_depths, test_results, label="Test rmse")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("rmse values")
plt.xlabel("max_depths")
plt.show


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.legend_handler import HandlerLine2D

train_results=[]
test_results=[]
max_features = list(range(1,12))
                    
for eta in max_features:
    regressor = GradientBoostingRegressor(max_features = eta)  
    regressor.fit(X_train, y_train)

    train_pred = regressor.predict(X_train)
    train_pred = train_pred.round()
    rmse_train = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    train_results.append(rmse_train)

    y_pred = regressor.predict(X_test)
    y_pred = y_pred.round()
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    test_results.append(rmse_test)

plt.title("max_features rmse's")
line1,= plt.plot(max_features, train_results, label="Train rmse")
line2, = plt.plot(max_features, test_results, label="Test rmse")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("rmse values")
plt.xlabel("max_features")
plt.show


# In[ ]:


X_train = X
y_train = y


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor(learning_rate=0.2, n_estimators=2000, max_depth=9, max_features=3)
regressor.fit(X_train, y_train)


# In[ ]:


test = pd.read_csv("test.csv")
test.isnull().sum()
test.fillna(test.mean(),inplace = True)
test.type.replace('new', 1, inplace = True)
test.type.replace('old', 0, inplace = True)


# In[ ]:


X_test = test[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 
               'feature9', 'feature10', 'feature11','type']].values
#predicting using model1
y_pred = regressor.predict(X_test)
y_pred = y_pred.round()
pandasDF = pd.DataFrame({"id" : test.id, "rating" :y_pred})
print(pandasDF)
pandasDF.to_csv("output_gdb.csv", index=False)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

regressor_GDB = GradientBoostingRegressor(learning_rate=0.2, n_estimators=2000, max_depth=9)
regressor_GDB.fit(X_train, y_train)
#predicting using model2
y_pred = regressor_GDB.predict(X_test)
y_pred = y_pred.round()
pandasDF = pd.DataFrame({"id" : test.id, "rating" :y_pred})
print(pandasDF)
pandasDF.to_csv("gdb_output.csv", index=False)

