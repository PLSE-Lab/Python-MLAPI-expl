#!/usr/bin/env python
# coding: utf-8

# This is still the original task: predict university admission. This time, only neural networks will be used for personal practice.

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


# taking the same steps with the data as the other file...

# <h2> Get the Data

# In[ ]:


df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


# indexing on Serial No.
df.set_index('Serial No.', inplace = True)


# <h2> Data exploration/cleaning

# In[ ]:


df.head()


# In[ ]:


# All of our given features
columns = df.columns.values
columns


# Note that the chance of admission and letter of reccomendation column have an additional whitespace at the end of their respective column names. This might lead to issues/confusion, so first removing these whitespaces

# In[ ]:


col_fix = []
for col in columns:
    col_fix += [col.strip()]
df.columns = col_fix
df.columns.values


# There is not much else to do in terms of data cleaning, this dataset was made nicely on purpose. This means we can go straight into analysis and exploration of the data

# In[ ]:


scatter_matrix(df)
plt.show()


# The research category is extremely binary and may not be as useful to include as a feature in our model. Let's look more in depth into it to determine if we should remove it.
# 
# Majority of the features appear to be linearly correlated with each other, which could potentially be an issue in the context of multicollinearity. This will be addressed after dealing with the research column.

# In[ ]:


df.plot(x = 'Research', y = 'Chance of Admit', kind = 'scatter')


# It appears that people that have research generally have a higher chance of admissions so we will keep this as a feature.
# 
# Now let us determine if multicollinearity will be an issue. The variance inflation factor (VIF) will be used to determine the significance of the effect of multicollinearity. Note that this was ignored in the other version.

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_calc(features):
    vif = pd.DataFrame()
    vif['features'] = features.columns
    vif['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    return vif


# In[ ]:


X_temp = df.copy()
X_temp.insert(0,'Intercept', 1)
X_temp.drop(columns = ['Chance of Admit'], inplace = True)
vif = vif_calc(X_temp)
vif


# Using the general guideline for VIF[1] GRE, CGPA and TOEFL will be further analyzed to see if any corrections need to be made. <br>
# [1] https://online.stat.psu.edu/stat462/node/180/#:~:text=As%20the%20name%20suggests%2C%20a,much%20the%20variance%20is%20inflated.&text=The%20general%20rule%20of%20thumb,of%20serious%20multicollinearity%20requiring%20correction.

# <h3> GRE, CGPA, TOEFL analysis

# It makes logical sense that these 3 features have a high variance inflation factor. People who have a higher GPA will most likely score higher on the TOEFL and GRE, because these 2 examinations are based in acadeamics. <br>
# 
# The main difference between the GRE and TOEFL is that the GRE is used to see how well a student can take graduate-level coursework and TOEFL is used to measure the participant's skill in English [2].
# 
# 
# [2]https://www.prepscholar.com/toefl/blog/gre-and-toefl/#:~:text=The%20TOEFL%20and%20GRE%20are,TOEFL%20measures%20English%20language%20skills.&text=It's%20possible%20you%20may%20have%20to%20take%20both%20exams.

# In[ ]:


temp = df[['GRE Score', 'TOEFL Score', 'CGPA', 'Chance of Admit']]
temp.corr(method = 'pearson')


# From the correlation matrix, it seems that there is a high correlation (>0.8) between all three features. However, since the TOEFL score has the lowest correlation coefficient to the Chance of admission, we will drop the TOEFL score. 
# 
# **NOTE**: a low correlation coefficient does not necessarily mean that the feature is unrelated to the target. It simply means that there is a *weak linear relationship*. It's entirely possible for that feature to be related to the target, albeit it would be a higher dimension relationship. 

# In[ ]:


df.drop(columns = ['TOEFL Score'], inplace = True)
df.head()


# In[ ]:


X_temp = df.copy()
X_temp.insert(0,'Intercept', 1)
X_temp.drop(columns = ['Chance of Admit'], inplace = True)
vif = vif_calc(X_temp)
vif


# While the VIF has gone down, it is still high for GRE and CGPA. It would be best if we can merge these 2 columns into one. Let's divide each column by their respective max score, then multiply those 2 values to obtain our metric for academic potential/ability

# In[ ]:


df['GRE Score'] = df['GRE Score'].divide(340)
df['CGPA'] = df['CGPA'].divide(10)
df.head()


# In[ ]:


df['Academics'] = df['GRE Score'] * df['CGPA']
df.head()


# We can now drop GRE and CGPA

# In[ ]:


df.drop(columns = ['GRE Score', 'CGPA'], inplace = True)
df.head()


# In[ ]:


X_temp = df.copy()
X_temp.insert(0,'Intercept', 1)
X_temp.drop(columns = ['Chance of Admit'], inplace = True)
vif = vif_calc(X_temp)
vif


# We can now see that our VIF scores are all satisfactory

# <h2> Model Preparation

# Now that we've cleaned our data and condensed several features, we can now begin to separate our data to put into our model

# In[ ]:


from sklearn.model_selection import train_test_split
X = df.copy()
X.drop(columns = ['Chance of Admit'], inplace = True)
y = df['Chance of Admit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[ ]:


# scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# <h3> Model Creation

# Because of the small # of available samples, we will be using 2 Dense layers and use iterated k fold with shuffling to create a more accurate model. To avoid overfitting, we will also be using l1 regularization.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
def model_create():
    model = Sequential()
    model.add(Dense(20, kernel_initializer='normal', activation = 'relu', input_dim = 5))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mae'])
    return model


# Using Kfolds

# In[ ]:


from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

estimator = KerasRegressor(build_fn = model_create, epochs = 500, batch_size = 500, verbose=0)
kfold = KFold(n_splits=15)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Baseline: %f (%f) MSE" % (results.mean(), results.std()))


# <h2> Final Evaluation

# Now that we have our model (*estimator*) we can evaluate it on our test data set

# In[ ]:


estimator.fit(X_train, y_train)
predictions = estimator.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("Final mean squared error: %f      R^2 value: %f" %(mean_squared_error(y_test, predictions), r2_score(y_test, predictions)))


# Note that there are 3 main areas that we can choose to tune our model:
# 1. in our function *model_create*, we can add/remove more layers and tweak the parameters of those layers to obtain a better model. Note that we have a limited sample size so a large # of nodes is not required
# 2. we can change the # of epochs to use
# 3. we can choose the # of folds to use (i.e. change value of k for kfolds)
# 
# To determine how well our tweaks to our model are, we want a low final mean squared error and a high (max 100) R^2 value.
