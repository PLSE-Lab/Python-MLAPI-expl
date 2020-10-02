#!/usr/bin/env python
# coding: utf-8

# **Kaggle Competition 1: Banking Predictions**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt


# **Getting Started**
# Run the necessary imports.
# 
# Read in the files. Get summary statistics on the dataset, both categorical and numerical features. 
# 
# Separate the numerical data from both the testing and training sets into their own dataframes.

# In[ ]:


bank_test = pd.read_csv('../input/bank-test.csv')
bank_train = pd.read_csv('../input/bank-train.csv')
bank_train_num = bank_train[['id', 'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
bank_test_num = bank_test[['id', 'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
results = bank_train[['y']]


# In[ ]:


bank_train.describe()


# In[ ]:


bank_test.describe()


# In[ ]:


bank_test.describe(include=['O'])


# Import the necessary packages to create several models

# In[ ]:


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


# Start by trying a **logistic regression**, using only the numerical features -- this seemed like a good way to get a baseline we could start from. Including only the numerical features was easier than handling all of the categorical features, and allowed us to get a model quickly. Additionally, we weren't sure how useful the categorical features would be after visualizing the data in a separate notebook.

# In[ ]:


lr = LogisticRegression()
lr_fit = lr.fit(bank_train_num, results)
lr_predict = lr.predict(bank_train_num)

#generate predictions on the test set from the logistic regression
lr_predict_test = lr.predict(bank_test_num)

print(classification_report(results, lr_predict))


# Try running a lasso regression to improve on the model.
# 
# This would automatically perform some feature selection if performed correctly, and might be more accurate in the long run than the logistic regression as it avoids overfitting the training data.  Unfortunately, this didn't work the way we wanted it to and wasn't a better model than the logistic regression.
# 

# In[ ]:


lasso = Lasso(alpha=5)
lasso_fit = lasso.fit(bank_train_num, results)
lasso_predict = lasso.predict(bank_train_num)

for i in range(len(lasso_predict)):
    if lasso_predict[i] < 0.7:
        lasso_predict[i] = 0
    else:
        lasso_predict[i] = 1

print(classification_report(results, lasso_predict))


# Try a ridge regression to improve on the model.
# 
# Similar to the lasso regression, we thought that this would help with feature selection. While ridge regression was better than lasso, it still didn't improve on the logistic model.

# In[ ]:


ridge = Ridge(alpha=5)
ridge_fit = ridge.fit(bank_train_num, results)
ridge_predict = ridge.predict(bank_train_num)

for i in range(len(ridge_predict)):
    if ridge_predict[i] < 0.7:
        ridge_predict[i] = 0
    else:
        ridge_predict[i] = 1

print(classification_report(results, ridge_predict))


# **Random Forest**
# 
# Try a random forest to improve on the logistic regression.
# 
# A random forest should be a more accurate model than logistic regression-- additionally, the ability to get feature importances from each feature is valuable when choosing which features to include in the model. After running the first random forest with all numerical features included, we dropped all numerical features except the top four on the "feature importance list." Neither of these models improved upon the logistic regression. 

# In[ ]:


# try a random forest
from sklearn.ensemble import RandomForestClassifier
rand_for = RandomForestClassifier(n_estimators = 500, random_state = 40)
rand_for_fit = rand_for.fit(bank_train_num, results)


# In[ ]:


#predict values based on the random forest
rand_for_predict_test = rand_for.predict(bank_test_num)


# In[ ]:


#run feature importance
feat_imp = pd.DataFrame(rand_for.feature_importances_, index=bank_train_num.columns)
print(feat_imp)

#edit the features used in the random forest based on these results
bank_train_num2 = bank_train[['age', 'campaign', 'pdays', 'euribor3m', 'nr.employed']]
bank_test_num2 = bank_test[['age', 'campaign', 'pdays', 'euribor3m', 'nr.employed']]

rand_for_fit2 = rand_for.fit(bank_train_num2, results)
rand_for_predict2 = rand_for.predict(bank_test_num2)


# **Adding the Categorical Features**
# 
# Add the categorical features back into a separate dataframe. Replace them with dummy variables.
# 
# Run a logistic regression with all of the features. (In this case, this didn't improve the model).

# In[ ]:


bank_train_d = pd.get_dummies(bank_train)
bank_test_d = pd.get_dummies(bank_test)

#bank_train_d = bank_train_d.drop('poutcome',axis=1)
#bank_test_d = bank_test_d.drop('poutcome',axis=1)
bank_train_d = bank_train_d.drop('duration',axis=1)
bank_test_d = bank_test_d.drop('duration',axis=1)
bank_train_d = bank_train_d.drop('default_yes', axis=1)


bank_train_d = bank_train_d.drop('y',axis=1)

print(bank_test.shape)
print(bank_test_d.shape)

#print(bank_train_d.shape)
#print(bank_test_d.shape)

lr_fit_dummies = lr.fit(bank_train_d, results)
lr_predict_dummies = lr.predict(bank_train_d)

#print(classification_report(results, lr_predict_dummies))

lr_predict_d = lr.predict(bank_test_d)


# In[ ]:


from sklearn.linear_model import RidgeClassifier
ridge2 = RidgeClassifier(alpha=5)
ridge_fit2 = ridge2.fit(bank_train_d, results)
ridge_predict2 = ridge2.predict(bank_train_d)

for i in range(len(ridge_predict2)):
    if ridge_predict2[i] < 0.7:
        ridge_predict2[i] = 0
    else:
        ridge_predict2[i] = 1

print(classification_report(results, ridge_predict2))

ridge_predict_test2 = ridge2.predict(bank_test_d)
print(bank_test_d.shape)


# Use a random forest on the new dataframe, including the categorical variables.

# In[ ]:


rand_for_fit_d = rand_for.fit(bank_train_d, results)
rand_for_predict_d = rand_for.predict(bank_test_d)

#feat_imp = pd.DataFrame(rand_for.feature_importances_,index = bank_train_d.columns,columns=['importance']).sort_values('importance',ascending=False)
#print(feat_imp)

print(bank_test_d.shape)


# Based on feature importance, do another random forest with only features with an importance higher than 0.02 (this is the same approach as before, but this time categorical variables are included). This did not improve the model.

# In[ ]:


bank_train_d2 = bank_train_d[['id', 'age', 'campaign', 'euribor3m', 'nr.employed', 'cons.conf.idx', 'poutcome_success']]
bank_test_d2 = bank_test_d[['id', 'age', 'campaign', 'euribor3m', 'nr.employed', 'cons.conf.idx', 'poutcome_success']]

rand_for_fit_d2 = rand_for.fit(bank_train_d2, results)
rand_for_predict_d2 = rand_for.predict(bank_test_d2)


# For the linear ridge regression in R, the occupation variable was converted into 12 levels of dummy variables and the remaining categorical variables were encoded numerically (ex. monday=0, tuesday=1, march=0, april=1, divorced=-1, single=0, married=1). 
# 
# ```{r}
# library(ridge)
# linRidgeMod <- linearRidge(y ~ ., data = finalset)
# predicted <- predict(linRidgeMod, Xt)
# predicted1 = as.data.frame(predicted)
# predicted1[predicted1$predicted > .7,] = 1
# predicted1[predicted1$predicted <= .7,] = 0
# predicted1 = cbind(Xt2$id, predicted1)
# colnames(predicted1) = c("id","Predicted")
# table(predicted1$Predicted)
# ```

# In[ ]:


print(bank_test_d.shape)
submission2 = pd.concat([bank_test_d.id, pd.Series(ridge_predict_test2)], axis = 1)
submission2.columns = ['id', 'Predicted']
submission2.to_csv('submission.csv', index=False)
#submission['id'].astype('int64')
print(len(submission2['id']))
print(len(submission2['Predicted']))

