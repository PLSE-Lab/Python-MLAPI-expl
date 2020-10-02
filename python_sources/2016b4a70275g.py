#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')


# Fill NaN

# In[ ]:


data.fillna(data.mean(),inplace=True)
#type_code = {'old':0,'new':1}
#data['type']=data['type'].map(type_code)


# Describe data

# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


#data[data['type']=='old'].describe()


# In[ ]:


#data[data['type']=='new'].describe()


# In[ ]:


corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
mask

# # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12,9))

# # Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# Get X and Y

# In[ ]:


cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
       'feature6', 'feature7', 'feature8', 'feature9',  'feature10',
       'feature11','type']
X = data[cols]
X = pd.get_dummies(data=X,columns=['type'])
y = data['rating']


# In[ ]:


corr = X.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
mask

for i in range(0,X.shape[1]):
    mask[i][i]=True
# # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12,9))

# # Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)


# In[ ]:


#from sklearn.ensemble import GradientBoostingRegressor
#grad = GradientBoostingRegressor()

#grad.fit(X_train,y_train)
#y_pred = grad.predict(X_test)
#y_pred=np.rint(y_pred)
#err_rms = (sum((y_test-y_pred)**2)/len(y_test))**(0.5)
#print("RMSE ={}".format(err_rms))


# In[ ]:


#model
from sklearn.ensemble import ExtraTreesRegressor
et = ExtraTreesRegressor(n_estimators=2000)
et.fit(X,y)
#y_pred = rand_for.predict(X_test)
#y_pred=np.rint(y_pred)
#err_rms = (sum((y_test-y_pred)**2)/len(y_test))**(0.5)
#print("RMSE ={}".format(err_rms))


# In[ ]:


#model
from sklearn.ensemble import RandomForestRegressor

rand_for = RandomForestRegressor(n_estimators=4000)
rand_for.fit(X,y)
#y_pred = rand_for.predict(X_test)
#y_pred=np.rint(y_pred)
#err_rms = (sum((y_test-y_pred)**2)/len(y_test))**(0.5)
#print("RMSE ={}".format(err_rms))


# # Final Submission

# In[ ]:


test_data=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
final= pd.DataFrame(test_data['id'])
test_data.fillna(test_data.mean(),inplace=True)
test_data = test_data[cols]
test_data = pd.get_dummies(data=test_data,columns=['type'])


# In[ ]:


y_pred = et.predict(test_data)
y_pred=np.rint(y_pred)


# In[ ]:


y_pred


# In[ ]:


final['rating']=y_pred
final.head()


# In[ ]:


final.to_csv('predicted.csv',index=False)


# In[ ]:




