#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


red = pd.read_csv("../input/winequality-red.csv")


# In[ ]:


red.tail()


# In[ ]:


red.info()


# In[ ]:


#red['quality'] = red['quality'].astype('category')


# In[ ]:


red.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set(xlabel = 'total sulfur dioxide',
      ylabel = 'free sulfur dioxide')
ax.scatter(red['total sulfur dioxide'], red['free sulfur dioxide'], c='r')
plt.show()


# In[ ]:


#red['quality'].describe()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = red.corr()
sns.heatmap(corr, xticklabels = corr.columns.values,
           yticklabels=corr.columns.values)


# In[ ]:


red.columns


# In[ ]:


red['quality'].hist()


# In[ ]:


red['total sulfur dioxide'].hist()


# In[ ]:


red['fixed acidity'].hist()


# In[ ]:


red['volatile acidity'].hist()


# In[ ]:


red['citric acid'].hist()


# In[ ]:


red['residual sugar'].hist()


# In[ ]:


red['chlorides'].hist()


# In[ ]:


red['free sulfur dioxide'].hist()


# In[ ]:


from sklearn.model_selection import train_test_split

X = red.iloc[:,0:11]
y = np.ravel(red.quality)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                  y,
                                                  test_size=0.33,
                                                  random_state = 42)


# In[ ]:


X.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(164, input_dim= 11, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))


# In[ ]:


model.compile(loss= 'mse',
             optimizer='rmsprop',
             metrics=['mse'])
model.fit(X_train, y_train, epochs=50, batch_size=1,verbose=1)


# In[ ]:


mse_value, mae_value = model.evaluate(X_test, y_test, verbose=0)

print(mse_value)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import r2_score

r2_score(y_test, y_pred)


# In[ ]:




