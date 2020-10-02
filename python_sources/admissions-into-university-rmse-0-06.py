#!/usr/bin/env python
# coding: utf-8

# Task at hand is to predict the likelihood of admission

# In[ ]:


from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np


# We are dealing with labelled data that is continuous so our model will be a supervised regression model

# <h2> Get the Data

# In[ ]:


df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


# indexing on Serial No.
df.set_index('Serial No.', inplace = True)


# <h2> Explore Data

# In[ ]:


df.head()


# The data has 7 total attributes:
# 1. GRE Score (340 max)
# 2. TOEFL Score (120 max)
# 3. University Rating (5 max)
# 4. Statement of Purpose (5 max)
# 5. Letter of Reccomendation (5 max)
# 6. Undergraduate GPA (10 max)
# 7. Research Experience (1 max)

# In[ ]:


# luckily we're only dealing with numbers and clean up of NaN values/invalid values is unecessary
df.info()


# In[ ]:


df.hist(bins = 50, figsize = (20, 15))
plt.show()


# In[ ]:


scatter_matrix(df)
plt.show()


# From first glance it seems that the chance of admission has a strong correlation with:
# 1. CGPA
# 2. LOR
# 3. SOP
# 4. TOEFL score
# 5. GRE score 
# 
# and a weak correlation with:
# 1. University rating 
# 2. Research
# 
# Let's check the correlation coefficient between the different attributes:

# In[ ]:


corr_matrix = df.corr()
corr_matrix["Chance of Admit "].sort_values(ascending = False)


# It seems like the TOEFL score, GRE score and CGPA have the highest correlation with university admittance. However, let's look at how all attributes relate to each other

# In[ ]:


corr_matrix


# From a quick glance, it appears that all attributes are related, either strongly or weakly, to other attributes. Fortunately, this should not affect the predictive power of our machine learning model. 
# 
# Now creating the training and testing set. Because of our extremely small dataset sample (500 entries total), we will split on the training and testing set on a 3:1 ratio to prevent overfitting of our model.

# In[ ]:


train_set, test_set = train_test_split(df, test_size = 0.25, random_state = 42)
# training set
X_train = train_set.values[:,0:7]
y_train = train_set.values[:,7]

# test set
X_test = test_set.values[:,0:7]
y_test = test_set.values[:,7]


# <h2> Preparing Data

# both min-max and StandardScaler will be used and compared. Note that both scalers are fit on the training data set and then applied to the test set as well. 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

minmax_scaler = MinMaxScaler(feature_range = (0, 1)).fit(X_train)
standard_scaler = StandardScaler().fit(X_train)

# Min-Max 
X_train_MM = minmax_scaler.transform(X_train)
X_test_MM = minmax_scaler.transform(X_test)

# Standard Scaler
X_train_ST = standard_scaler.transform(X_train)
X_test_ST = standard_scaler.transform(X_test)


# <h2> Model Creation

# Several models will be compared:
# 1. Ridge Regression
# 2. Lasso Regression
# 3. Elastic Net
# 4. SVM
# 5. Decision Tree

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor 

models = []
models += [['Ridge', Ridge(alpha = 0.9, solver = "cholesky")]]
models += [['Lasso', Lasso(alpha = 1)]]
models += [['Elastic Net', ElasticNet(alpha = 0.1, l1_ratio = 0.25)]]
models += [['SVM', LinearSVR()]]
models += [['Tree', DecisionTreeRegressor()]]


# <h4> Min Max
#     

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kfold = KFold(n_splits = 5, random_state = 42)
result_MM =[]
names = []

for name, model in models:
    cv_score = -1 * cross_val_score(model, X_train_MM, y_train, cv = kfold, scoring = 'neg_root_mean_squared_error')
    result_MM +=[cv_score]
    names += [name]
    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))


# From this, it seems like Ridge linear regression is the best choice (RMSE = 0.061186) with SVM in close second place (RMSE = 0.061156). Note that the constant -1 was multiplied to the cross_val_score simply to make the RMSE positive

# <h4> Standard Scaler

# In[ ]:


result_ST =[]
for name, model in models:
    cv_score = -1 * cross_val_score(model, X_train_ST, y_train, cv = kfold, scoring = 'neg_root_mean_squared_error')
    result_MM +=[cv_score]
    print('%s: %f (%f)' % (name,cv_score.mean(), cv_score.std()))


# Once again, Ridge and SVM are the top two models with respective RMSE scores of 0.060759 and 0.061416. Using both Min Max and StandardScaler, Ridge Regression seems to have the least amount of error in both cases so let us continue with this model

# <h2> Evaluation

# In[ ]:


# training the models
Ridge_model_MM = Ridge(alpha = 0.9, solver = "cholesky").fit(X_train_MM, y_train)
Ridge_model_ST = Ridge(alpha = 0.9, solver = "cholesky").fit(X_train_ST, y_train)

# getting predictions
predictions_MM = Ridge_model_MM.predict(X_test_MM)
predictions_ST = Ridge_model_ST.predict(X_test_ST)


# Now evaluating our models

# In[ ]:


from sklearn.metrics import mean_squared_error
print("Ridge, Min Max: " + str(np.sqrt(mean_squared_error(y_test, predictions_MM))))
print("Ridge, Standard Scaler: " + str(np.sqrt(mean_squared_error(y_test, predictions_ST))))


# Using Standard Scaler results in a lower RMSE score, so our final model to choose to predict university admission is Ridge Regression model <br>
# 
# P.S. This is my first ever Kaggle submission, please give me comments/advice/areas to improve on!
