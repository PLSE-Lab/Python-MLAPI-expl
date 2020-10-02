#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


bottle = pd.read_csv("/kaggle/input/calcofi/bottle.csv")
bottle.columns


# In[ ]:


degree = bottle.T_degC


# In[ ]:


salt=bottle.Salnty


# In[ ]:


data = pd.concat([degree,salt], axis = 1, ignore_index=True)


# In[ ]:


data.rename(columns={0:"sicaklik", 1: "tuzluluk"}, inplace=True)


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


data.isna().sum()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.corr()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
data.plot(kind="scatter", x="tuzluluk", y="sicaklik", alpha= 0.5, color="blue")
plt.xlabel('tuzluluk')
plt.ylabel('sicaklik')
plt.show()


# In[ ]:


x=data[["tuzluluk"]]
y=data["sicaklik"]


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state= 0)


# In[ ]:


model = LinearRegression().fit(x_train,y_train)  # ornek modeli tanimliyoruz


# In[ ]:


y_pred = model.predict(x_test)
y_pred


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
r2_score(y_test, y_pred)


# In[ ]:


model.predict([[33.4400]])


# In[ ]:


f,ax = plt.subplots(figsize = (20,20)) # resmin buyuklugunu ayarlar
sns.heatmap(bottle.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
#annot True uzerindeki degerleri yazar
#linewidth cizgilerin kalinligi
#fmt virgullu degerleri ayarlar

plt.show();


# In[ ]:


aray = np.arange(len(y_test))
plt.plot(aray, y_pred, color="red" )  
plt.plot(aray, y_test, color="blue",alpha=0.5)

plt.show();


# In[ ]:


# Plot outputs
plt.plot(x_test, y_test,  color='black')
plt.plot(y_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:


# from sklearn.preprocessing import PolynomialFeatures

# polynomial_regression = PolynomialFeatures(degree = 100)  # degree ust ifade eder. 

# x_polynomial = polynomial_regression.fit_transform(x)

# linear_regression2 = LinearRegression()

# linear_regression2.fit(x_polynomial,y)

# y_head2 = linear_regression2.predict(x_polynomial)

# plt.plot(x_polynomial, y_head2, color="red", label="poly")


# In[ ]:


r2_score(y_head2,y)


# In[ ]:





# In[ ]:





# In[ ]:




