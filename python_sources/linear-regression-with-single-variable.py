
# coding: utf-8

# ## Linear Regression with Single Variable

# In[27]:


# Libraries
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns


# In[28]:


# Import datasets
training_data = pd.read_csv('../input/linear-regression/train.csv')
# training_data = training_data.fillna(training_data.mean())
testing_data = pd.read_csv('../input/linear-regression/test.csv')


# In[29]:


training_data = training_data.dropna()


# In[30]:


# Describe datasets
training_data.describe()
# testing_data.describe()


# In[31]:


# Plot Training Data
figure_dim = (20, 12)
fig, ax = plt.subplots(1, 2,  figsize=figure_dim)
sns.regplot(x='x', y='y', data=training_data, marker='o', color='blue', ax=ax[0]).set_title('Training Data')
sns.regplot(x='x', y='y', data=testing_data, marker='o', color='green', ax=ax[1]).set_title('Test Data')
plt.show()


# In[32]:


# Assign Xs and Ys
X_Train = training_data.iloc[:, :-1].values
Y_Train = training_data.iloc[:, :1].values
X_Test = testing_data.iloc[:, :-1].values
Y_Test = testing_data.iloc[:, 1:].values


# In[33]:


# Init Linear Regressor
model = LinearRegression()
model.fit(X=X_Train, y=Y_Train)


# In[34]:


Y_Pred = model.predict(X=X_Test)


# In[37]:


mean_squared_error(y_pred=Y_Pred, y_true=Y_Test)
