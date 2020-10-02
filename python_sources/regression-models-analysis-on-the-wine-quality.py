#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# # Getting Data and doing some sanity checks

# In[13]:


wine = pd.read_csv("../input/winequality-red.csv",sep=',')


# In[14]:


wine.head()


# In[15]:


wine.isnull().sum()


# In[16]:


wine.info()


# In[17]:


wine.describe()


# ## There are no null values in dataset. It's pretty clean dataset. No data cleaning has done. And all the data is having numerical values. But we should standardize the data before using into models. Because most of the algorithms assume that all the features are centered around zero and have same variance. We will do that in later stages.

# ## Going to check how the features are correlated with each other. Here am just doing for White wine data. In case if we have lots of features in the dataset. It is best practise to check which features are more correlated with the target variable. The model will give more reliable output when we pass significant features into the model.

# In[18]:


import seaborn as sns


# In[19]:


corr = wine.corr()
fig, ax = plt.subplots(figsize = (10,10))
g= sns.heatmap(corr,ax=ax, annot= True)
ax.set_title('Correlation between variables')


# In[20]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# ## Here am taking all the features as predictors but we can eliminate pH, free sulfur dioxide, residual sugar features from predictor list. Becase they are not much correlated with target variable "Quality".

# In[21]:


y = wine.quality
X = wine.drop('quality',axis = 1)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


train_x,test_x,train_y,test_y = train_test_split(X,y,random_state = 0, stratify = y)


# In[24]:


from sklearn import preprocessing


# ## Here am using standarscaler function to standardize the data. We can use other methods also to do this. But this function will make sure the test data also standardized based on training data mean. 

# In[25]:


scaler = preprocessing.StandardScaler().fit(train_x)
train_x_scaled = scaler.transform(train_x)


# In[26]:


test_x_scaled = scaler.transform(test_x)


# ## Cross validating the most common regression models to find which algorithm works better

# In[27]:


from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


# In[28]:


models = []
models.append(('DecisionTree', DecisionTreeRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('GradienBoost', GradientBoostingRegressor()))
models.append(('SVR', SVR()))
names = []


# In[29]:


for name,model in models:
    kfold = model_selection.KFold(n_splits=5,random_state=2)
    cv_results = model_selection.cross_val_score(model,train_x_scaled,train_y, cv= kfold, scoring = 'neg_mean_absolute_error')
    names.append(name)
    msg  = "%s: %f" % (name, -1*(cv_results).mean())
    print(msg)


# ## Regression models can be validated using Mean Absolute Error(MAE). The less the mae value the better the model works. From above results we can decide that Random Forest model works better than other models for our data.
# ## For classification models , we will use accuracy score and confusion matrix to validate across the models. Accuracy score should be high for best results.

# In[30]:


model = RandomForestRegressor()
model.fit(train_x_scaled,train_y)
pred_y = model.predict(test_x_scaled)


# In[31]:


mean_absolute_error(pred_y,test_y)


# In[32]:


test_y.head()


# In[33]:


pred_y


# In[26]:


get_ipython().run_line_magic('pinfo', 'RandomForestRegressor')


# ## Below am going to find the parameter n_estimator, to tune the model for better result. The same way we can find other paramters also. 

# In[34]:


def get_mae_rf(num_est, predictors_train, predictors_val, targ_train, targ_val):

    # fitting model with input max_leaf_nodes
    model = RandomForestRegressor(n_estimators=num_est, random_state=0)

    # fitting the model with training dataset
    model.fit(predictors_train, targ_train)

    # making prediction with the test dataset
    preds_val = model.predict(predictors_val)

    # calculate and return the MAE
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


# In[37]:


plot_mae = {}
for num_est  in range(2,50):
    my_mae = get_mae_rf(num_est,train_x_scaled,test_x_scaled,train_y,test_y)
    plot_mae[num_est] = my_mae


# In[38]:


plt.plot(list(plot_mae.keys()),plot_mae.values())
plt.show()


# * ## The mae is less after 36. Lets check how much performance is improved by passing this parameter into the model

# In[39]:


model = RandomForestRegressor(n_estimators=36)
model.fit(train_x_scaled,train_y)
pred_y = model.predict(test_x_scaled)


# In[40]:


mean_absolute_error(pred_y,test_y)

