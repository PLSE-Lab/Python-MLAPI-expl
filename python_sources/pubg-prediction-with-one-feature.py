#!/usr/bin/env python
# coding: utf-8

# # <center>PUBG Prediction with One Feature</center>
# ### <center>by Bon Crowder</center>

# ***

# ## Preliminary Code

# Here are the dependencies:

# In[ ]:


import pandas as pd
import numpy as np


# And then I add my personal prettifiers (not required).

# In[ ]:


pd.options.display.max_columns = 60
pd.options.display.float_format = lambda x: f' {x:,.2f}'
import warnings
warnings.filterwarnings("ignore")


# ## Choose Feature

# ### Load the PUBG Dataset

# In[ ]:


game = pd.read_csv('../input/train_V2.csv', index_col='Id')
game.head()


# In[ ]:


game.describe()


# In[ ]:


game.info()


# In[ ]:


game.isna().sum()


# The only missing value is a single value in `winPlacePerc`. Let's take a look at it.

# In[ ]:


filt = game['winPlacePerc'].isna()
game[filt]


# In[ ]:


filt2 = game['maxPlace'] == 1
game[filt2]


# Based on the information in the variable descriptions as well as the correlation of `maxPlace` to `winPlacePerc`, I think filling in the missing values for `winPlacePerc` with 0 (total loser) seems appropriate. 

# In[ ]:


game = game.fillna(0)


# Just to confirm we're all done with missing values:

# In[ ]:


game.isna().sum().sum()


# In order to decide which single feature to use, I can look at the correlation of `winPlacePerc` to the other features. I've highlighted the minimums and I'll manually look at the maximums.

# In[ ]:


game.corr().style.format("{:.2%}").highlight_min()


# Looks like `killPlace` is the most dramatically and negatively correlated. So I'll start with that one.
# 
# Also, `weaponsAcquired` and `walkDistance` are both strongly and *positively* correlated, so those may be the two to check out next.

# ## Choosing a Model for Use on killPlace

# I'm going to set up the feature set and target first. Then I'll look at a handful of possible models to see which might work the best.

# In[ ]:


X = game['killPlace'].values.reshape(-1, 1)
X[:10]


# In[ ]:


y = game['winPlacePerc'].values
y[:10]


# #### Linear Regression

# I'll start with linear regression and check to see what the cross validation score is to see if that might be a good model.

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cvs_lr = cross_val_score(lr, X, y, cv=15)
cvs_lr.mean(), cvs_lr.std()


# That's kinda pitiful. Let's see if we can do better with a different model.

# #### Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()


# In[ ]:


cvs_dtr = cross_val_score(dtr, X, y)
cvs_dtr.mean(), cvs_dtr.std()


# Ah! That's much better. 
# 
# Outside of this notebook, I ran cross validation on the Random Forest Regressor, but the difference was almost zero. So for this notebook, we'll stick with DTR.

# ## Training the Decision Tree Regressor

# We already have `dtr` instantiated as our Decision Tree Regressor. Now it's time to train it.

# In[ ]:


dtr.fit(X,y)


# Now it's time to bring in the test data and take a peek.

# In[ ]:


game_test = pd.read_csv('../input/test_V2.csv', index_col='Id')
game_test.head()


# In[ ]:


game_test.isna().sum().sum()


# Looks good from here. So let's focus on the important part: `killPlace`.

# In[ ]:


X_test = game_test['killPlace'].values.reshape(-1, 1)
X_test[:10]


# Now to get our predictions:

# In[ ]:


predictions = dtr.predict(X_test).reshape(-1,1)


# We can put those in a dataframe:

# In[ ]:


dfpredictions = pd.DataFrame(predictions, index=game_test.index).rename(columns={0:'winPlacePerc'})
dfpredictions.head(15)


# And then create the output that can be sent to Kaggle.

# In[ ]:


dfpredictions.to_csv('submission.csv', header=True)


# And our result is...
