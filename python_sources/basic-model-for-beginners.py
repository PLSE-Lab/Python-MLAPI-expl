#!/usr/bin/env python
# coding: utf-8

# # 1. Import the necessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt

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


# # 2. Load the csv file in a dataframe

# In[ ]:


try:
    df_iris = pd.read_csv('../input/iris/Iris.csv')
    print("Successfully loaded Iris.csv")
    
except Exception as e:
    print(e)


# In[ ]:


print("Null values in Iris dataset \n\n", df_iris.isnull().sum())


# # 3. View the dataset
# 
#     - 3.1 First few records in the dataset
#     - 3.2 Total number of Rows and Columns loaded
# 

# In[ ]:


print(df_iris.head(10))         

print("\nRows and columns in the Iris dataset : ", df_iris.shape)                 # shows rows and columns


# In[ ]:


print(df_iris.info())   # check basic info of the dataset


#     - 3.3 See the statistical summary of the data

# In[ ]:


print("Iris Statistical Summary \n", df_iris.describe())


# - 3.4 See the distinct number of Iris Species, which need to be predicted, and their distribution in the dataset

# In[ ]:


print("Distinct Species of Iris are : ",df_iris.Species.unique())

print("\n", df_iris.groupby('Species').size())


# # 4. Visualize the data

# In[ ]:


# 4.1 Create Univariate Plots

# box and whisker plot

plt.boxplot([df_iris.SepalLengthCm, df_iris.SepalWidthCm, df_iris.PetalLengthCm, df_iris.PetalWidthCm],
           labels=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
plt.show()


# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='Species',y='PetalLengthCm',data=df_iris,
                order=['Iris-virginica','Iris-versicolor','Iris-setosa'],
                linewidth=2.5,orient='v',dodge=False)


# In[ ]:


#scatter plot of Petal Length & Width
get_ipython().run_line_magic('matplotlib', 'inline')
sns.FacetGrid(df_iris,hue='Species',height=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()


# In[ ]:


#scatter plot of Sepal Length & Width
get_ipython().run_line_magic('matplotlib', 'inline')
sns.FacetGrid(df_iris,hue='Species',height=5).map(plt.scatter,'PetalLengthCm','PetalWidthCm').add_legend()


# In[ ]:


# Consolidated view of the scatter plot of Sepal and Petal - Length & Width

sns.set(style="ticks", color_codes=True)
#iris = sns.load_dataset("iris")
get_ipython().run_line_magic('matplotlib', 'inline')

#iris = sns.load_dataset("iris")
sns.pairplot(df_iris.iloc[:,1:6], hue="Species")


# We drop off the id column, since it adds no value to our analysis

# In[ ]:


df_iris.drop('Id', axis=1, inplace=True)
print(df_iris.columns)


# # 5. Find correlation between columns
# 
# **Observation** : 
#  - SepalLengthCm & SepalWidthCm do not have a high correlation
#  - PetalLengthCm & PetalWidthCm have a very high correlation
#  - PetalLengthCm & SepalLengthCm also have a high correlation
#  

# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(df_iris.corr(),annot=True,cmap='cubehelix_r')    # heatmap of correlation matrix calculted by(iris.corr())
plt.show()


# # 6. Modeling
# 
# *     6.1 - split dataset into train & validation

# In[ ]:


X = df_iris.iloc[:,0:4]
y = df_iris.iloc[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


print(X_train.shape, X_validation.shape, Y_train.shape, Y_validation.shape)


# We will use stratified 10-fold cross validation to estimate model accuracy.
# 
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

# * **6.2 Build Model**
# 
# Let's try a few simple models
# We will try to build the below models, and evaluate which one works best
# 
# * * Logistic Regression (LR)
# * * Linear Discriminant Analysis (LDA)
# * * K-Nearest Neighbors (KNN).
# * * Classification and Regression Trees (CART).
# * * Gaussian Naive Bayes (NB).
# * * Support Vector Machines (SVM).

# In[ ]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# # 7. Predict
# 
# 
# * 7.1 Make Predictions
# 
# Since SVM has the best accuracy, we will use SVM to do the predictions

# In[ ]:


model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("Predicted the class of Iris for",len(predictions), "records")


# 
# 
# * 7.2 Evaluate the result of prediction

# In[ ]:


# Evaluate predictions
print("SVM Model accuracy on validation set is : ", accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# Now we will to create a model without the highly correlated PetalLengthCm column
# 

# In[ ]:


X = df_iris.loc[:,['SepalLengthCm','SepalWidthCm', 'PetalWidthCm']]
y = df_iris.loc[:,'Species']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("SVM Model accuracy on validation set is : ", accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
                   


# The results look quite the same, and we have a simpler model (with lesser attributes)

# # 8. Ensemble
# 
# Till now, we have evaluated only single models
# 
# We will now try a simple ensemble - xgboost
# 

# In[ ]:


#### xgboost

import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train, Y_train)

print('Accuracy of xgb classifier is {:.5f} on training data'.format(xgb_clf.score(X_train, Y_train)))
print('Accuracy of xgb classifier is {:.5f} on validation data'.format(xgb_clf.score(X_validation, Y_validation)))


# Though the validation acuracy did not improve much, normally the ensemble models provide better results than a single model
