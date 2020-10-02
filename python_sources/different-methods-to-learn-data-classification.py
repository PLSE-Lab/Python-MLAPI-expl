#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Import sklearn methods
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report,                             confusion_matrix 
from sklearn.feature_selection import GenericUnivariateSelect, f_classif, SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Import sklearn classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct

#Import XGBOOST classifier and importance plot method
from xgboost import XGBClassifier, plot_importance

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Import data
dataframe = pd.read_csv("../input/Dataset_spine.csv")
dataframe = dataframe.drop('Unnamed: 13', axis=1)
dataframe['Class_att'].replace({"Abnormal": 0, "Normal": 1}, inplace=True)
dataframe['Class_att'].unique()
dataframe.head()


# In[ ]:


#Dataframe description
dataframe.describe()


# In[ ]:


#Check for damaged samples
dataframe.info()


# In[ ]:


# Split into data and target
target = dataframe['Class_att']
data = dataframe.drop("Class_att", axis=1)
del dataframe
data.head()


# In[ ]:


# Correlation matrix
print(data.shape)
data.corr()


# In[ ]:


#Plotted for better visualizing
plt.figure(figsize=(9,9))
sns.heatmap(data.corr(), cbar=True, square=True, annot=True,
            xticklabels = data.columns.tolist(), yticklabels = data.columns.tolist())


# In[ ]:


# Turn data and target values into numpy array
names = data.columns.get_values()
data = data.as_matrix()
target = target.as_matrix()


# In[ ]:


# Analysing data - feature selection - GenericUnivariateSelect -> using SelectKBest method 
selector = GenericUnivariateSelect(f_classif,mode="k_best", param=4)
fit = selector.fit(data,target)


# In[ ]:


figsize = (9,9)

weights = fit.scores_
ind = np.arange(len(names))
plt.figure(figsize=figsize)
plt.bar(ind, weights, orientation = 'vertical')
plt.xticks(ind+0.4, names)
plt.xlabel("Features names")
plt.ylabel("Weights")
plt.title("Feature importance")

for i, w in enumerate(weights):
    num = "%.4f" % w
    plt.text(i+0.4, w + 1.5, num, ha='center', color='blue', fontweight='bold')

plt.show()


# In[ ]:


# Feature selection - XGBoost feature importance method
xgsel = XGBClassifier(n_estimators=400)
xgsel.fit(data, target)

plt.figure(figsize = figsize)
plot_importance(xgsel)
threshold = xgsel.feature_importances_
threshold = sorted(threshold)
print(threshold)


# In[ ]:


# Using feature importance form XGBoost classifier to classify dataset 
# with different algorithms
cv = StratifiedShuffleSplit(test_size=0.2)

names = [
    "SVC",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "SGDClassifier",
    "KNeighborsClassifier",
    "GaussianProcessClassifier",
    "AdaBoostClassifier"
]

classifiers = [
    SVC(C=2.0, tol=1e-4),
    DecisionTreeClassifier(max_features="log2"),
    ExtraTreesClassifier(n_estimators=100, bootstrap=True, n_jobs=-1),
    SGDClassifier(loss="perceptron", penalty="elasticnet", n_iter=10, shuffle=True, n_jobs=-1),
    KNeighborsClassifier(n_neighbors=2, n_jobs=-1),
    GaussianProcessClassifier(),
    AdaBoostClassifier(ExtraTreesClassifier())
]

clf_values = []

print_values = False

for name, clf in zip(names, classifiers):
    bucket = []
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print("Using classifier:",name)
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    for thresh in threshold:
        
        data_point = {'name':name, "n_features":0, "mean":0, "std":0}
        
        selector = SelectFromModel(xgsel, threshold=thresh, prefit=True)
        X = selector.transform(data)
        X = StandardScaler().fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3)
        
        results = cross_val_score(clf, X, target, cv=cv)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        delic = f1_score(y_test, y_pred)
        
        if print_values:
        
            print("Thresh: %.3f, Number of features: %d"%(thresh, X.shape[1]))
            print("Mean: %.3f, Std: %.3f" %(results.mean()*100, results.std()))
            print("Accuracy: %.3f, F1 Score: %.3f"%(accuracy*100, delic))
            print(clf)
            print("\n#######################################\n")
        
        data_point['n_features'] = X.shape[1]
        data_point['mean'] = results.mean()*100
        data_point['std'] = results.std()
        bucket.append(data_point)
        
    clf_values.append(bucket)


# In[ ]:


# Using feature importance from GenericUnivariateSelect to classify dataset 
# with different algorithms 
clf_values_2 = []

print_values = False

for name, clf in zip(names, classifiers):
    bucket = []
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print("Using classifier:",name)
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    for i in range(12):
        
        data_point = {'name':name, "n_features":0, "mean":0, "std":0}
        
        selector = GenericUnivariateSelect(f_classif, mode='k_best', param=i+1)
        X = selector.fit_transform(data,target)
        X = StandardScaler().fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3)
        
        results = cross_val_score(clf, X, target, cv=cv)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        delic = f1_score(y_test, y_pred)
        
        if print_values:
        
            print("Number of features: %d" %  X.shape[1])
            print("Mean: %.3f, Std: %.3f" %(results.mean()*100, results.std()))
            print("Accuracy: %.3f, F1 Score: %.3f"%(accuracy*100, delic))
            #print(clf)
            print("\n#######################################\n")
        
        data_point['n_features'] = X.shape[1]
        data_point['mean'] = results.mean()*100
        data_point['std'] = results.std()
        bucket.append(data_point)
        
    clf_values_2.append(bucket)


# In[ ]:


#Quick lookup at the data - GUS featuer selection
for i in range(len(clf_values_2[0])):
    print(clf_values_2[0][i])


# In[ ]:


#GenericUnivariateSelect feature selecion - mean accuracy
plot_data = []
legend = []
x_axis = [i+1 for i in range(12)]

plt.figure(figsize=(9,9))

for j in range(len(clf_values_2)):
    #print(clf_values_2[j][0]['name'])
    legend.append(clf_values_2[j][0]['name'])
    for i in range(len(clf_values_2[j])):
        plot_data.append(clf_values_2[j][i]['mean'])
    plt.plot(x_axis, plot_data)
    plot_data = []

x_ticks = [i for i in range(14)]

plt.xticks(x_ticks)
plt.xlabel("Number of features")
plt.ylabel("Mean accuracy")
plt.title("GUS feature selection\nMean accuracy plot")
plt.legend(legend, loc=0)
plt.show()


# In[ ]:


#GenericUnivariateSelect feature selecion - standard deviation
plot_data = []
legend = []
x_axis = [i+1 for i in range(12)]

plt.figure(figsize=(9,9))

for j in range(len(clf_values_2)):
    #print(clf_values_2[j][0]['name'])
    legend.append(clf_values_2[j][0]['name'])
    for i in range(len(clf_values_2[j])):
        plot_data.append(clf_values_2[j][i]['std'])
    plt.plot(x_axis, plot_data)
    plot_data = []

x_ticks = [i for i in range(14)]

plt.xticks(x_ticks)
plt.xlabel("Number of features")
plt.ylabel("Standard deviation ")
plt.title("GUS feature selection\nStandard deviation accuracy plot")
plt.legend(legend, loc=0)
plt.show()


# In[ ]:


#Quick lookup at the data - XGBoost selection
for i in range(len(clf_values[0])):
    print(clf_values[0][i])


# In[ ]:


#Sort points by number of features
for i in range(len(clf_values)):
    clf_values[i] = sorted(clf_values[i], key=lambda k: k['n_features'])


# In[ ]:


#Quick check
for i in range(len(clf_values[0])):
    print(clf_values[0][i])


# In[ ]:


#XGBoost feature selecion - mean accuracy
plot_data = []
legend = []
x_axis = [i+1 for i in range(12)]

plt.figure(figsize=(9,9))

for j in range(len(clf_values)):
    #print(clf_values[j][0]['name'])
    legend.append(clf_values[j][0]['name'])
    for i in range(len(clf_values[j])):
        plot_data.append(clf_values[j][i]['mean'])
    plt.plot(x_axis, plot_data)
    plot_data = []

x_ticks = [i for i in range(14)]

plt.xticks(x_ticks)
plt.xlabel("Number of features")
plt.ylabel("Mean accuracy")
plt.title("XGBoost feature selection\nMean accuracy plot")
plt.legend(legend, loc=0)
plt.show()


# In[ ]:


#XGBoost feature selecion - standard deviation
plot_data = []
legend = []
x_axis = [i+1 for i in range(12)]

plt.figure(figsize=(9,9))

for j in range(len(clf_values_2)):
    #print(clf_values_2[j][0]['name'])
    legend.append(clf_values_2[j][0]['name'])
    for i in range(len(clf_values_2[j])):
        plot_data.append(clf_values_2[j][i]['std'])
    plt.plot(x_axis, plot_data)
    plot_data = []

x_ticks = [i for i in range(14)]

plt.xticks(x_ticks)
plt.xlabel("Number of features")
plt.ylabel("Standard deviation")
plt.title("XGBoost feature selection\nStandard deviation plot")
plt.legend(legend, loc=0)
plt.show()


# That is it for now! Again sorry for not using XGBoost classifier. It couldn't execute few times(probably not enough memory or processor power). If you want anything more or you want to express your attitude towards this notebook send me a message or post comment. I will be very happy if this script would be helpful ;)

# In[ ]:




