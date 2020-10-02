#!/usr/bin/env python
# coding: utf-8

# # In this kernel I am going to classify the data in 3 categories **Fail, Medium and Good**  
# If student final grade (0-20) is below 5 then fail, above 15 then Good else in Medium category
# 
# ## And then will predict which students are lying in which category and should be more focused. Will find out students which may fail so that attention can be paid to those failing students

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
sns.set_color_codes("pastel")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
os.listdir('../input')


# In[ ]:


# I'm use only student-mat.csv
data = pd.read_csv('../input/student-mat.csv')


# # Data Overview

# In[ ]:


print("G3 range: Min={}, Max={}".format(data["G3"].min(), data["G3"].max()))
data.head(5)


# **Creating a new column G3_class with values - Fail, Medium and Good**  
# If student final grade is below 5 then fail, above 15 then Good else in Medium category
# 
# *In last print the new column - Scroll horizonatally in last to see the new values*

# In[ ]:


def create_g3_class(data):
    return ["Fail", "Medium", "Good"][0 if data["G3"] <= 5 else 1 if data["G3"] <= 15 else 2]

data["G3_class"] = data.apply(lambda row: create_g3_class(row), axis=1)
data.head()
data.info()


# # Final Grade Prediction

# Output target of this dataset is **Final Grade**. Let's use some regression model to predict it. I'll limit myself to 4 simple regression models (without searching the best parameters): decision tree regression, linear regression, lasso and ridge regression.

# ## With G1 and G2 test results features

# In[ ]:


from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

from sklearn.model_selection import cross_val_score


# In[ ]:


y = data['G3_class']
X = data.drop(['G3', 'G3_class'], axis=1)


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


names = ['RandomForestClassifier', 'NaiveBayes' , 'DecisionTreeClassifier', 'XGBClassifier']

clf_list = [RandomForestClassifier(),
            MultinomialNB(),
            DecisionTreeClassifier(),
           xgb.XGBClassifier()]


# In[ ]:


clf_scores = {}
for name, clf in zip(names, clf_list):
    clf_scores[name]= cross_val_score(clf, X, y, cv=5).mean()
    print(name, end=': ')
    print(clf_scores[name])


# let's look at feature importances.

# ## Feature Importances

# In[ ]:


best_classifier = sorted(clf_scores, key=clf_scores.get, reverse=True)[0]
best_classifier


# In[ ]:


clf = clf_list[names.index(best_classifier)]
clf.fit(X, y)


# Print main features contributing to top 1% for selection

# In[ ]:


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    if(importances[indices[f]] >= 0.01):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))


# We can see the top selection criteria is results achieved in last tests done by students.  
# Other than test resuts, students should also be told to focus on their - 
# * Attendance
# * number of past class failures 

# Let's look at scores of models, trained without G1 and G2 features (Test results).

# ## Without G1 and G2 features

# In[ ]:


X = data.drop(['G3', 'G2', 'G1', 'G3_class'], axis=1)


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


clf_scores = {}
for name, clf in zip(names, clf_list):
    clf_scores[name]= cross_val_score(clf, X, y, cv=5).mean()
    print(name, end=': ')
    print(clf_scores[name])


# In[ ]:


best_classifier = sorted(clf_scores, key=clf_scores.get, reverse=True)[0]
best_classifier


# In[ ]:


clf = clf_list[names.index(best_classifier)]
clf.fit(X, y)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    if(importances[indices[f]] >= 0.01):
        print("%d. Feature %s (%f)" % (f + 1, X.columns.values[indices[f]], importances[indices[f]]))


# If test resuts are ignored then students should also be told to focus on their - 
# 
# 1. Attendance
# 2. Time spent on Outing 
# 3. Student with less and more age should focus more on studies
# 4. number of past class failures 
# 5. Freetime after school
# 6. Alcohol consumption
# 7. studytime
# 8. Maintenance of health
# 9. Student with less educated mothers shoud focus more on studies

# **Stritified Split the data in training, test **

# In[ ]:


from sklearn.model_selection import train_test_split
X = data.drop(['G3_class'], axis=1)
X = pd.get_dummies(X)     #Convert to categorical
y = data['G3_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, stratify=y)
X_train= X_train.drop(['G3'], axis=1)
import copy
X_test_withG3 = copy.deepcopy(X_test)    #will be used in end to display actual G3 score
X_test= X_test.drop(['G3'], axis=1)


# In[ ]:


import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


clf = clf_list[names.index(best_classifier)]
print("using classifer: %s"%best_classifier)
# clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)
# print(clf.classes_)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, normalize=True, title='Normalized confusion matrix')

plt.show()


# **As we can see in above confusion matrix there are a lot of wrong predictions (above and below diagnal).**  
# *Most probable reason of this is bias-ness in input data.* Let us verify this below

# In[ ]:


plt.figure()
plt.boxplot(data['G3'], notch=True, sym='gD', vert=False)
plt.title('G3 (final grade) score distribution in dataset')
plt.show()


# G3 output class is unevenly distributed. Above box plot shows that most values are in range approx 8-13.    
# Below count plot also shows uneven distribution.  

# In[ ]:


p = sns.countplot(data['G3_class']).set_title('G3 class distribution')


# To counter this issue we need to oversample the data in classes which are having less samples.   
# imblearn package is used to do this.

# In[ ]:


from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import CondensedNearestNeighbour, AllKNN, OneSidedSelection, RandomUnderSampler
from imblearn.ensemble import BalanceCascade, EasyEnsemble

from collections import Counter
# X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
X_resampled, y_resampled = AllKNN(sampling_strategy=['Medium']).fit_resample(X_train, y_train)
# X_resampled, y_resampled = SMOTETomek().fit_resample(X_train, y_train)  #sampling_strategy='minority'
# X_resampled, y_resampled = EasyEnsemble().fit_resample(X_train, y_train)
# X_resampled = X_resampled[0] ; y_resampled = y_resampled[0]
print(sorted(Counter(y_resampled).items()))


# Apply PCA to find out most relevant features

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X_resampled) #pd.get_dummies(data)
principalDf_train = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
principalComponentstest = pca.fit_transform(X_test) #pd.get_dummies(data)
principalDf_test = pd.DataFrame(data = principalComponentstest
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[ ]:


clf = clf_list[names.index(best_classifier)]
print("Using classifier: %s"%best_classifier)
# clf = xgb.XGBClassifier()
# clf = DecisionTreeClassifier()  #Using for demo and consistency
clf.fit(principalDf_train, y_resampled)

y_pred = clf.predict(principalDf_test)
y_pred_prob = clf.predict_proba(principalDf_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

np.set_printoptions(precision=2)
# print(clf.classes_)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=clf.classes_, normalize=True, title='Normalized confusion matrix')

plt.show()


# Much better prediction results after over and under sampling   
# # Let us show the students which are in danger zone

# In[ ]:


y_test_list = list(y_test)
X_test_withG3_list = list(X_test_withG3['G3'])
for idx, item in enumerate(y_pred):
    if(item == 'Fail'):
        print("Student {} \t [Actual Failed?: {}  \tG3: {}]".format(idx, y_test_list[idx], X_test_withG3_list[idx]))


# # Few points to note and justification
# ## * Undersampling is used on Fail class students as these are the students which should be focused
# ## * The data is not accurate thereby resulting some incorrect prediction in above results
# ## * **The results above contain failed students and a lot of students with Medium G3 scores also. The choice was to reduce the Medium G3 scorers and miss some Failed students too.  Important point is no/minimum Failed students are missed (As seen in last confusion matrix), So I resampled focusing on Failed classes.     It is important to focus on NOT MISSING ANY FAILING STUDENT, and focusing on Medium students will make them only better.**
