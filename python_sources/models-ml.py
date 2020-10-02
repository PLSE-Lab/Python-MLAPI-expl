#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

data = pd.read_csv('../input/Air.csv')
data.head()

y = data.Pressure
X = data.drop('Pressure',axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.2)
 
models = [LinearRegression(),
          RandomForestRegressor(n_estimators=100, max_features='sqrt'),
          KNeighborsRegressor(n_neighbors=6),
          SVR(kernel='linear')
	      SVR(kernel='rbf', C=1e3, gamma=0.1)

	      SVR(kernel='poly', C=1e3, degree=2)
	    ]
 
TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
        m = str(model)
        tmp['Model'] = m[:m.index('(')]
        model.fit(X_train, y_train)
        tmp['R2_Price'] = r2_score(y_test, model.predict(X_test))
        TestModels = TestModels.append([tmp])
        print(model.score(X_test,y_test))
 
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()


# In[ ]:


'../input/Air.csv'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




