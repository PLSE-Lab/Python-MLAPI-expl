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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, skew


# In[ ]:


color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


# In[ ]:


data = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')


# In[ ]:


test=pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')


# In[ ]:


data= data.drop('id', axis=1)
data= data.drop('b10', axis=1)
data= data.drop('b2', axis=1)
data= data.drop('b5', axis=1)
data= data.drop('b6', axis=1)
    # data= data.drop('b9', axis=1)
data= data.drop('b12', axis=1)
data= data.drop('b16', axis=1)
data= data.drop('b17', axis=1)
data= data.drop('b19', axis=1)
data=data.drop('b21',axis=1)
data= data.drop('b26', axis=1)
data= data.drop('b18', axis=1)
data= data.drop('b24', axis=1)
data= data.drop('b36', axis=1)
data= data.drop('b41', axis=1)
data= data.drop('b44', axis=1)
data= data.drop('b45', axis=1)
data= data.drop('b54', axis=1)
data= data.drop('b58', axis=1)
data= data.drop('b61', axis=1)
data= data.drop('b64', axis=1)
data= data.drop('b68', axis=1)
data= data.drop('b71', axis=1)
data= data.drop('b72', axis=1)
    # data= data.drop('b79', axis=1)

data= data.drop('b81', axis=1)
data= data.drop('b83', axis=1)

data= data.drop('b86', axis=1)
data= data.drop('b88', axis=1)
data= data.drop('b92', axis=1)
#     # #data= data.drop('label', axis=1)
#       #data= data.drop('b2', axis=1)
# #data= data.drop('b11', axis=1)
#   data= data.drop('b10', axis=1)
#  #data= data.drop('b12', axis=1)
#     #data= data.drop('b13', axis=1)
# #data= data.drop('b26', axis=1)
# #data= data.drop('b36', axis=1)
#     #data = data.drop('id', axis=1)  
data= data.drop('b1', axis=1)
# #data= data.drop('b20', axis=1)
# #data= data.drop('b16', axis=1)
# #data= data.drop('b40', axis=1)
# #data= data.drop('b39', axis=1)
# #data= data.drop('b44', axis=1)
# #data= data.drop('b72', axis=1)
# #data= data.drop('b54', axis=1)
# #data= data.drop('b58', axis=1)
# #data= data.drop('b61', axis=1)
# #data= data.drop('b62', axis=1)
# #data= data.drop('b63', axis=1)
# #data= data.drop('b64', axis=1)
# #data= data.drop('b65', axis=1)
# #data= data.drop('b68', axis=1)
# #data= data.drop('b69', axis=1)
# #data= data.drop('b71', axis=1)
# #data= data.drop('b75', axis=1)
# #data= data.drop('b87', axis=1)
# #data= data.drop('b88', axis=1)
data= data.drop('b89', axis=1)
# #data= data.drop('b81', axis=1)
# #data= data.drop('b86', axis=1)
# #data= data.drop('b6', axis=1)
# #data= data.drop('b17', axis=1)
# #data= data.drop('b18', axis=1)
# #data= data.drop('b19', axis=1)
# #dats= data.drop('b20', axis=1)
# #data= data.drop('b21', axis=1


# In[ ]:


test= test.drop('id', axis=1)
test= test.drop('b10', axis=1)
test= test.drop('b2', axis=1)
test= test.drop('b5', axis=1)
test= test.drop('b6', axis=1)
    # data= data.drop('b9', axis=1)
test= test.drop('b12', axis=1)
test= test.drop('b16', axis=1)
test= test.drop('b17', axis=1)
test= test.drop('b19', axis=1)
test=test.drop('b21',axis=1)
test= test.drop('b26', axis=1)
test= test.drop('b18', axis=1)
test= test.drop('b24', axis=1)
test= test.drop('b36', axis=1)
test= test.drop('b41', axis=1)
test= test.drop('b44', axis=1)
test= test.drop('b45', axis=1)
test= test.drop('b54', axis=1)
test= test.drop('b58', axis=1)
test= test.drop('b61', axis=1)
test= test.drop('b64', axis=1)
test= test.drop('b68', axis=1)
test= test.drop('b71', axis=1)
test= test.drop('b72', axis=1)
    # data= data.drop('b79', axis=1)

test= test.drop('b81', axis=1)
test= test.drop('b83', axis=1)

test= test.drop('b86', axis=1)
test= test.drop('b88', axis=1)
test= test.drop('b92', axis=1)
#     # #data= data.drop('label', axis=1)
#       #data= data.drop('b2', axis=1)
# #data= data.drop('b11', axis=1)
#   data= data.drop('b10', axis=1)
#  #data= data.drop('b12', axis=1)
#     #data= data.drop('b13', axis=1)
# #data= data.drop('b26', axis=1)
# #data= data.drop('b36', axis=1)
#     #data = data.drop('id', axis=1)  
test= test.drop('b1', axis=1)
# #data= data.drop('b20', axis=1)
# #data= data.drop('b16', axis=1)
# #data= data.drop('b40', axis=1)
# #data= data.drop('b39', axis=1)
# #data= data.drop('b44', axis=1)
# #data= data.drop('b72', axis=1)
# #data= data.drop('b54', axis=1)
# #data= data.drop('b58', axis=1)
# #data= data.drop('b61', axis=1)
# #data= data.drop('b62', axis=1)
# #data= data.drop('b63', axis=1)
# #data= data.drop('b64', axis=1)
# #data= data.drop('b65', axis=1)
# #data= data.drop('b68', axis=1)
# #data= data.drop('b69', axis=1)
# #data= data.drop('b71', axis=1)
# #data= data.drop('b75', axis=1)
# #data= data.drop('b87', axis=1)
# #data= data.drop('b88', axis=1)
test= test.drop('b89', axis=1)
# #data= data.drop('b81', axis=1)
# #data= data.drop('b86', axis=1)
# #data= data.drop('b6', axis=1)
# #data= data.drop('b17', axis=1)
# #data= data.drop('b18', axis=1)
# #data= data.drop('b19', axis=1)
# #dats= data.drop('b20', axis=1)
# #data= data.drop('b21', axis=1


# In[ ]:


#data=data.drop("id",axis=1)
corr = data.corr()
plt.subplots(figsize=(30,30))
sns.heatmap(corr, vmax=0.9, square=True)


# In[ ]:


sns.distplot(data['label'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(data['label'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('Label distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(data['label'], plot=plt)
plt.show()


# In[ ]:


y = data["label"]
y


# In[ ]:


data= data.drop('label', axis=1)


# In[ ]:


X=data


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=42)  #Checkout what does random_state do


# In[ ]:


def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / len(actual)
    return np.sqrt(mean_error)


# In[ ]:


#TODO
from sklearn.preprocessing import StandardScaler

numerical_features=X_train.columns[:-7]

scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])  
test[numerical_features]=scaler.transform(test[numerical_features])
 

# It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc

X_train[numerical_features].head(100)


# In[ ]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


regr = RandomForestRegressor(random_state=0,verbose=1).fit(X_train, y_train)


# In[ ]:


y_pred1=regr.predict(X_val)


# In[ ]:


regr.score(X_val, y_val)


# In[ ]:


rmse_metric(y_val.values,y_pred1)


# In[ ]:


print(X_train.columns)


# In[ ]:


test.columns


# In[ ]:


y_pred=regr.predict(test)


# In[ ]:


datafr=pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor


# In[ ]:





# In[ ]:


reg1 = BaggingRegressor(base_estimator=RandomForestRegressor(random_state=0,verbose=1),
                        n_estimators=10, random_state=0,verbose=1).fit(X_train, y_train)


# In[ ]:


# y_pred2=reg1.predict(X_val)


# In[ ]:


# reg1.score(X_val, y_val)


# In[ ]:


# rmse_metric(y_val.values,y_pred2)


# In[ ]:


# y_prede=reg1.predict(test)


# In[ ]:





# In[ ]:


from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor

estimators = [
    ('gr', RandomForestRegressor(random_state=0,verbose=1)),
    ('svr', HistGradientBoostingRegressor(random_state=0,verbose=1))
]
reg = VotingRegressor(
    estimators=estimators
)


# In[ ]:


reg.fit(X_train, y_train)


# In[ ]:


y_pred_5=reg.predict(X_val)


# In[ ]:


rmse_metric(y_val.values,y_pred_5*0.5+y_pred1*0.5)


# In[ ]:


yyy=reg.predict(test)


# In[ ]:


y_fin=0.5*yyy+0.5*y_pred


# In[ ]:


pdd=[]
print(len(y_fin))
for i in range(0,len(y_fin)):
    pdd.append([datafr.loc[i,'id'],y_fin[i]])


# In[ ]:


out = pd.DataFrame(pdd, columns=['id', 'label'],)
out.head(53175)


# In[ ]:


out.to_csv('2017A7PS0104G.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




