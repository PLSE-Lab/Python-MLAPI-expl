#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load Libraries

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import libraries for data transformation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Import classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc



import time
import warnings
warnings.filterwarnings("ignore")
print(os.listdir("../input"))


# # 1. Exploratory Data Analysis

# In[ ]:


df  = pd.read_csv('../input/mushrooms.csv')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df['class'].value_counts(normalize = True) # Check Class Balance


# In[ ]:


analysis = [df.shape, df.columns, df.info(), df.isnull().sum()]
for j in analysis:
    print(j,'\n----------------------------------------------------------------------------\n')


# There are 8124 rows and 23 columns in the dataset of which one is lalbel. The dataset is clean with no missing data. The class variable has two unique classes (P for poisonous and E for edible). The class label looks very balanced with similar number of mushrooms classified under each label. The entire dataset is categorical (text data), we will have to convert them to nominal type.

# ### **1.1. Visualize cross tab output

# In[ ]:


fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25),sharey=True)
fig.subplots_adjust(hspace=1.2, wspace=0.6)


for ax, col in zip(axes[0], df.iloc[:,1:].columns):
    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)
    plt.xlabel(col, fontsize=18)
    #ax.set_title(col)
for ax, col in zip(axes[1], df.iloc[:,5:].columns):
    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)
    plt.xlabel(col, fontsize=18)
    #ax.set_title(col)
for ax, col in zip(axes[2], df.iloc[:,9:].columns):
    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)
    plt.xlabel(col, fontsize=18)
    #ax.set_title(col)
for ax, col in zip(axes[3], df.iloc[:,13:].columns):
    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)
    plt.xlabel(col, fontsize=18)
    #ax.set_title(col)
for ax, col in zip(axes[4], df.iloc[:,17:].columns):
    pd.crosstab(df[col], df['class']).plot.bar(stacked=True, ax = ax)
    plt.xlabel(col, fontsize=18)
    #ax.set_title(col)


#handles, labels = ax.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center')
plt.tight_layout()


# * From the above cross tab plots we can see that some of the group in the variables show clear demarcation between classes for poisonous and edible for example : in odour , f class belongs to poisonous and m to edible
# * However most of the others share the groups. 
# * Veil typehas only group. This variable is not useful in our analysis, we can remove it.
# * veil-colour, ring-number and gill-attachment also seems to exhibit same behaviour as Veil-type.Only f class exists we can validate it below with cross tab.

# ### 1.2. Analyze redundant columns

# In[ ]:


df.groupby(['gill-attachment','ring-number','veil-color'])['class'].value_counts(normalize=True).unstack()


# As observed, most of the classes in these groups are redundant since the entire group belongs to one class. Lets remove all these variables --> 'gill-attachment', 'ring-number', 'veil-color' and 'veil-type'. These variables will not help us classify data into either classes.

# # 2. Data Transformation

# ### 2.1. Drop variables

# In[ ]:


data = df.drop(['gill-attachment','ring-number','veil-color','veil-type'], axis=1)
data.shape


# Now we have 19 variables including one class label. Next we need to address catgeorical variables, one of the most popular way of handling categorical variable is one-hot encoding.

# ### 2.2. Tranform X and y

# In[ ]:


y = data.iloc[:,0]
data_X = data.drop('class', axis=1)


# 2.2.1. **One-Hot encoding for X vars and Label encoding for y var**
# 
# In this method each category value is converted into a new column and assigned a 1 or 0 (True/False) value to the column. This has the benefit of not weighting a value improperly. However this can increase the dimensions exponentially depending on the number of categories and categorical columns we are dealing with.

# In[ ]:


# Label Encoding
le=LabelEncoder()
y=le.fit_transform(y)

#One-Hot encoding
X = pd.DataFrame()
for variables in data_X:
    dummy_var = pd.get_dummies(data_X[variables],prefix=variables)
    X = pd.concat([X,dummy_var], axis=1)

X.head()


# In[ ]:


X.shape


# The number of variables increased from 18 to 107 with one-hot encoding. Since all the variables were categorical, we one-hot encoded all of them. Hence there is no need for standarding the data.

# ### 2.3. Split data into training and testing

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)


# The data is now ready for modelling, we will try different models like linear, non-linear, parametric, non-parametric etc in this project.

# # 3. Data Mining

# ## Parametric Models
# [In statistics, a parametric model or parametric family or finite-dimensional model is a particular class of statistical models. Specifically, a parametric model is a family of probability distributions that has a finite number of parameters.](https://en.wikipedia.org/wiki/Parametric_model)
# 
# In this project we will run hyparameter tuning for all the parametric models. scikit-learn has a package called GridSearchCV, which takes the dictionary of hypermeters and provides the best parameter.

# ### Logistic Regression
# [In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).](https://en.wikipedia.org/wiki/Logistic_regression)
# 

# #### 1. Find estimator parameters

# In[ ]:


lr = LogisticRegression()
lr.get_params().keys()


# #### 2. Tune hyperparameters

# In[ ]:


start = time.clock()

parameters = {'solver':('liblinear', 'newton-cg', 'lbfgs', 'sag'), 
              'C': np.logspace(1,0.1,10),
             'max_iter': [200],
             'n_jobs':[-1]}

clf = GridSearchCV(lr, parameters, cv=5, verbose=1)
best_model = clf.fit(X, y)
  
print('Time taken (in secs) to tune hyperparameters: {:.2f}'.format(time.clock() - start))


# #### 3. Best model accuracy

# In[ ]:


y_pred = clf.predict(X_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print(clf.best_estimator_)
print('\n---------------------------------------------------------------------------------')
print('Confusion matrix of logistic regression classifier on test set: \n{}'.format(metrics.confusion_matrix(y_test,y_pred)))
print('\n---------------------------------------------------------------------------------')
print('Accuracy of logistic regression classifier on test set: {:.10f}'.format(clf.score(X_test, y_test)))
print('\n---------------------------------------------------------------------------------')
print('ROC of logistic regression classifier on test set: {:.10f}'.format(roc_auc))


# ### Plot ROC Curve

# In[ ]:


plt.figure()
lw = 2
plt.plot(false_positive_rate[2], true_positive_rate[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)'% roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

