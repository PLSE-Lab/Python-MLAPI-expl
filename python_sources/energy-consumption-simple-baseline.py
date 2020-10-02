#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read Data

# In[ ]:


train = pd.read_csv('../input/infopulsehackathon/train.csv', index_col='Id')
test = pd.read_csv('../input/infopulsehackathon/test.csv', index_col='Id')

train.head()


# # Data Preprocesing

# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


X = train.drop(columns = 'Energy_consumption')
y = train['Energy_consumption']


# In[ ]:


ohe = OneHotEncoder()
ohe_cols = train.loc[:,train.dtypes == 'object'].columns

ohe_data = pd.DataFrame(ohe.fit_transform(train[ohe_cols]).toarray(), dtype=int)
train = pd.concat([train.drop(columns = ohe_cols), ohe_data], axis=1)

ohe_data = pd.DataFrame(ohe.transform(test[ohe_cols]).toarray(), dtype=int)
X_test = pd.concat([test.drop(columns = ohe_cols), ohe_data], axis=1)

train.head()


# In[ ]:


X = train.drop(columns = 'Energy_consumption')
y = train['Energy_consumption']


# # Validation

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# In[ ]:


ridge = Ridge(alpha=100, random_state=42)
mse = -cross_val_score(ridge, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-100).mean()
mae = -cross_val_score(ridge, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-100).mean()
r2 = cross_val_score(ridge, X, y, scoring='r2', cv=5, n_jobs=-100).mean()
print(f'mean_squared_error : {mse}\nmean_absolute_error : {mae}\nr2 : {r2}')


# # Adversarial Validation

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ### Adversarial Dataset creation

# In[ ]:


X_adv = X.append(X_test).reset_index(drop=True)
y_adv = pd.Series(0, index= X_adv.index)
y_adv.iloc[X.shape[0]:] = 1

skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=42)


# ### Adversarial result

# In[ ]:


log_reg = LogisticRegression(C=10, random_state=42)
cross_val_score(log_reg, X_adv, y_adv, scoring='roc_auc', cv=skf.split(X_adv, y_adv), n_jobs=-1).mean()


# # Fit

# In[ ]:


ridge.fit(X,y)


# # Prediction

# In[ ]:


prediction = ridge.predict(X_test)
prediction[prediction < 0] = 0


# # Vizualize

# In[ ]:


print('Train target distribution')
y.hist(bins=30)

plt.show()

print('Test Prediction distribution')
pd.Series(prediction).hist(bins=30);


# # Submission

# In[ ]:


sub = pd.read_csv('../input/infopulsehackathon/sample_submission.csv', index_col='Id')
sub['Energy_consumption'] = prediction
sub.to_csv('submission.csv')
sub


# # Conclusion <br>
# In this kernel we have implemented simple linear model approach. Nevertheless MSE loss is quite a large number, but R2 score shows ~0.61, which means that our model is adequate. The last, but not the least: adversarial validation result shows 0.5 auc, which means that our cv result is lb-consistent.
