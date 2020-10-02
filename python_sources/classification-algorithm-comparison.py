#!/usr/bin/env python
# coding: utf-8

# # Comparison of Various Machine Learning Module Using UCI Heart Disease Data

# **In this karnel compare result of various machine learning module using the scikit-learn package**

# **Importing the required libaries**

# In[ ]:


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from  sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
plt.style.use('seaborn')
warnings.filterwarnings('ignore')


# In[ ]:


file_name = '../input/heart.csv'


# In[ ]:


data_df = pd.read_csv(file_name)


# In[ ]:


data_df.head()


# Columns
# age: age in years
# sex: (1 = male; 0 = female)
# cp: chest pain type
# trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# chol:serum cholestoral in mg/dl
# fbs:(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# restecg:resting electrocardiographic results
# thalach:maximum heart rate achieved
# exang:exercise induced angina (1 = yes; 0 = no)
# oldpeak:ST depression induced by exercise relative to rest
# slope:the slope of the peak exercise ST segment
# ca:number of major vessels (0-3) colored by flourosopy
# thal:3 = normal; 6 = fixed defect; 7 = reversable defect
# target:1 or 0 

# ### Checking for Null Values
# If there are any null values we need to handle them. To learn about various ways to handle the missing values go through [this](http://www.dipeshpoudel.com.np/2018/08/16/foreplay-before-doing-data-part-i-handling-missing-data/) blog post written by me.

# In[ ]:


pd.DataFrame(data_df.isna().sum(),columns=['null_count'])


# Since there are no missing values we do not need to handle them

# In the given dataset all the data are in the numeric form but they are not continuous data. In this particular case the description of each column is given and we can use it to determine which is a numerical column is and which is a categorical. If this information is not provided then we can use take a look at the distribution plot of each column to determine if the column is categorical or numerical in nature.

# In[ ]:


for col in data_df.columns:
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.distplot(data_df[col])
    plt.show()


# In[ ]:


data_df['target'].value_counts().plot(kind='bar')
plt.title("Target Frequency")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()


# From the observation of the charts above we need to convert some of the columns to categorical.

# In[ ]:


data_df.dtypes


# In[ ]:


dtype_map={"sex":"category","cp":"category","fbs":"category","restecg":"category","exang":"category",
           "slope":"category","ca":"category","thal":"category","target":"category"}


# In[ ]:


data_df = data_df.astype(dtype_map)


# In[ ]:


sns.pairplot(data_df)


# In[ ]:


#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data_df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


X = data_df.drop(columns=['target'])
y = data_df['target']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.25)


# In[ ]:


# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[ ]:


# prepare models
models = []
models.append(("LogisticRegression",LogisticRegression()))
models.append(("SVC",SVC()))
models.append(("LinearSVC",LinearSVC()))
models.append(("KNeighbors",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("RandomForest",RandomForestClassifier()))
rf2 = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0, max_features=None)
models.append(("RandomForest2",rf2))
models.append(("MLPClassifier",MLPClassifier()))
# evaluate each model in turn
results = []
names = []
seed=0
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    cross_val_result = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print("Print the Corss Validation Result {}".format(name))
    print(cross_val_result)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    plot_confusion_matrix(cm=cm, classes=[0,1])
    acc_score = accuracy_score(y_test,y_pred)
    print("Accuracy Score of {} is {}".format(name,acc_score))


# # Conclusion

# From the analysis above we can reach the following conclusion:
# 1. The cross validation score shows that Random Forest is the best model for this dataset since the average accuarcy from cross validation is high and the standard deviation (value inside bracket) is low.
# 2. In terms of accuracy Multi Layer Precprton (MLP) is the best model.

# In[ ]:




