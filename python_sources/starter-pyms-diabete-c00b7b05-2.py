#!/usr/bin/env python
# coding: utf-8

# # Ridge Regression, Logistic Regression and Decision Tree : Diabete
# In this notebook I analyze a dataset related to Diabete. In particulare I'm interested to tell if a person has diabete depending from other features, like glucose.
# I tried to use Ridge Regression width different alphas and degrees, Logistic Regression with different degrees, and Decision Tree with different criteria and limiting depths.
# 
# Obviously, because we want a Binary Classifier, using Ridge Regression will not give good results. I added it too because I think it's interesting too see how inefficace is to use this algorithm for a classification problem.
# In particular, with Ridge Regression we consider that a person doesn't have diabete if and only if the prediction is lower than 0.5.
# 
# ## Intro to logistic regression
# Logistic regression is a statistical method for predicting binary classes. It is a special case of regression where the target variable is categorical in nature (discrete). In particular, this algorithm predicts the probability of occurrence of a binary event and it needs a function that has a codomain from 0 to 1 (obviously), for example the inverse of [logit function](https://en.wikipedia.org/wiki/Logit).
# In order  to estimate the parameters of the features, this algorithm uses maximum likelihood.

# In[ ]:


import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# First of all, I analyze the dataset. We can see there are 768 samples with 9 features (including the target: diabete).

# In[ ]:


nRowsRead = None # specify 'None' if want to read whole file
df1 = pd.read_csv('/kaggle/input/diabete.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'diabete.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(5)


# In[ ]:


df1.describe()


# Here there is a function to plot the correlation matrix. This can be useful to see visually which features are more correlated: for example (and obviously) diabete and glucose.

# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()
    
plotCorrelationMatrix(df1, 8)


# # Applying the algorithms
# In the following lines of code I prepare the train and test dataset.

# In[ ]:


y = df1.pop('diabete')
x = df1
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=5)


# Here there is the core. We can see in the plot the different models and the respective accuracy (correct predictions / number of predictions).

# In[ ]:


# CONFIG
alphas = [0, 0.004, 0.2, 1, 5, 20, 100]
criteria = ["entropy","gini"]
MAX_DEPTH = 10
MIN_DEG = 1
MAX_DEG = 5
res = list()

for depth in range(1, MAX_DEPTH+1):
    for c in criteria:
        model_tree = DecisionTreeClassifier(criterion=c, max_depth=depth)
        model_tree.fit(X_train, Y_train)
        predictions = model_tree.predict(X_test)
        res.append((f"Decision tree with criteria={c} and max_depth={depth}",accuracy_score(Y_test, predictions)))
for deg in range(MIN_DEG, MAX_DEG+1):
    model_tree = make_pipeline(PolynomialFeatures(degree=deg), LogisticRegression())
    model_tree.fit(X_train, Y_train)
    predictions = model_tree.predict(X_test)
    res.append((f"Logistic regression with degree={deg}",accuracy_score(Y_test, predictions)))
for deg in range(MIN_DEG, MAX_DEG+1):
    for a in alphas:
        model_reg = make_pipeline(PolynomialFeatures(degree=deg), Ridge(alpha=a))
        model_reg.fit(X_train, Y_train)
        predictions = list(map(lambda p: 0 if p>0.5 else 1, model_reg.predict(X_test)))
        res.append((f"Ridge regression with degree={deg} and alpha={a}",accuracy_score(Y_test, predictions)))

res.sort(key=lambda a: a[1])
plt.figure(figsize=(10,10))
plt.title("Accuracy of different models")
plt.plot(range(0,len(res)), list(map(lambda a: a[1], res)), 'bo-', linewidth=2, markersize=4)
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.show()


# In[ ]:


# Legend
for i in range(len(res)):
    print(f"Model {i}: {res[i]}")


# # Conclusion 
# Obviously we can notice that Ridge Regression has very bad results.
# 
# Decision Tree is good in particolar using gini criteria for information gain. If we put high max_depth the tree reduces the accuracy probably due the overfitting.
# 
# Logistic Regression is very good, but with high degrees there is overfitting (for example with degree>3).
