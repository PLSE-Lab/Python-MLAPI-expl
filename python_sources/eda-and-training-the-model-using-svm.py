#!/usr/bin/env python
# coding: utf-8

#  <h1><center><span style="color:red">Breast Cancer Classification</span> </center></h1>

#  <h3><center>There are two main classifications of tumors. One is known as benign and the other as malignant. A benign tumor is a tumor that does not invade its surrounding tissue or spread around the body. A malignant tumor is a tumor that may invade its surrounding tissue or spread around the body.</center></h3>
#  
#   <h3><center>- This notebook is used to classify a breast cancer patient by wheather it is malignant or benign</center></h3>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# ## 1.0 Analyzing the dataset to understand the data

# In[ ]:


data = pd.read_csv('../input/data.csv')


# In[ ]:


data=data.drop('Unnamed: 32',axis=1)


# In[ ]:


data.head()


# ### 1.1 Null values

# In[ ]:


data.apply(lambda x: sum(x.isnull()))


# ### 1.2 Unique Values

# In[ ]:


print(data.shape[0])
data.apply(lambda x : len(x.unique()))


# ### 1.3 Comparing the valus in the dependant variable

# In[ ]:


plt.figure(1)
data['diagnosis'].value_counts(normalize=True).plot.bar( title= 'dependant variable')


# ### 1.4 Analyzing the variables, after normalizing the variables and after log transforming the variable

# In[ ]:


def dist(variable):
    plt.subplot(222)
    ax1=plt.subplot(221)
    sns.distplot(data[variable]);
    ax2=plt.subplot(222)
    sns.distplot(np.log1p(data[variable]));
    ax2=plt.subplot(223)
    mms = MinMaxScaler()
    sns.distplot(mms.fit_transform(data[variable].values.reshape(-1,1)))


# ### 1.5 Analyzing the correlation of variables 

# In[ ]:


fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(data.corr(),cmap=sns.diverging_palette(220, 20, as_cmap=True))


# ### 1.6 Excluding one variable that is correlated

# In[ ]:


data = data.drop('radius_mean',axis=1)
data = data.drop('perimeter_mean',axis=1)
data = data.drop('area_mean',axis=1)
data = data.drop('perimeter_worst',axis=1)
data = data.drop('area_worst',axis=1)
data = data.drop('radius_se',axis=1)
data = data.drop('perimeter_se',axis=1)


# ### 1.7 Analyzing the distribution of each numerical variable 

# In[ ]:


dist('smoothness_mean')


# In[ ]:


dist('texture_mean')


# In[ ]:


dist('compactness_mean')


# In[ ]:


dist('concavity_mean')


# In[ ]:


dist('concave points_mean')


# In[ ]:


dist('symmetry_mean')


# In[ ]:


dist('fractal_dimension_mean')


# In[ ]:


dist('texture_se')


# In[ ]:


dist('area_se')


# In[ ]:


dist('smoothness_se')


# In[ ]:


dist('compactness_se')


# In[ ]:


dist('concavity_se')


# In[ ]:


dist('concave points_se')


# In[ ]:


dist('symmetry_se')


# In[ ]:


dist('fractal_dimension_se')


# In[ ]:


dist('radius_worst')


# In[ ]:


dist('texture_worst')


# In[ ]:


dist('smoothness_worst')


# In[ ]:


dist('compactness_worst')


# In[ ]:


dist('concavity_worst')


# In[ ]:


dist('concave points_worst')


# In[ ]:


dist('symmetry_worst')


# In[ ]:


dist('fractal_dimension_worst')


# ### 1.8 Create a copy of the dataset to predict the model performance before and after remove outliers

# In[ ]:


data_outliers_removed = data.copy()


# In[ ]:


ax = sns.boxplot(y="texture_mean",  data=data_outliers_removed, linewidth=2.5)
description = data_outliers_removed.texture_mean.describe()
Q1 = description[4]
Q3 = description[6]
outliers_low = Q1 - (1.5 * (Q3-Q1))
outliers_high = Q3 + (1.5 * (Q3-Q1))
print(outliers_low,outliers_high)


# In[ ]:


numerical = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# In[ ]:


for i in numerical:
    if i in data.columns:
        print (i + ' : ' + str(data_outliers_removed[i].mean()))


# In[ ]:


for i in numerical:
    if i in data.columns:
        description = data_outliers_removed[i].describe()
        Q1 = description[4]
        Q3 = description[6]
        outliers_low = Q1 - (1.5 * (Q3-Q1))
        outliers_high = Q3 + (1.5 * (Q3-Q1))
        median = data_outliers_removed[i].median()
        temp_high = data_outliers_removed[i]>outliers_high
        temp_low = data_outliers_removed[i]>outliers_low
        data_outliers_removed.loc[temp_high == True,i]= median
        data_outliers_removed.loc[temp_low == True,i]= median


# In[ ]:


for i in numerical:
    if i in data.columns:
        print (i + ' : ' + str(data_outliers_removed[i].mean()))


# ### 1.9 Converting numerical variables to the log of each variable

# In[ ]:


for i in numerical:
    if i in data.columns:
        data_outliers_removed[i] = np.log1p(data_outliers_removed[i])
        data[i] = np.log1p(data[i])


# ### 1.10 Removing the ID variable

# In[ ]:


data = data.drop('id',axis=1)
data_outliers_removed = data_outliers_removed.drop('id',axis=1)


# ## 2.0 Model Training and Prediction

# ### 2.1 Using Logistic Regression to predict the model's preformance

# In[ ]:


def prediction(x,y,regressor):
    le = LabelEncoder()
    y=le.fit_transform(y)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    return accuracy_score(y_test,y_pred)    


# In[ ]:


LR = LogisticRegression()


# ### 2.2  performance without outliers

# In[ ]:


prediction(data_outliers_removed.drop('diagnosis',axis=1),data['diagnosis'],LR)


# ### 2.3 Performance with outliers

# In[ ]:


prediction(data.drop('diagnosis',axis=1),data['diagnosis'],LR)


# ### 2.4 Using support vector machine by using different parameters

# In[ ]:


le = LabelEncoder()
x = data.drop('diagnosis',axis=1)
y = le.fit_transform(data['diagnosis'])
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    #print(accuracy_score(y_pred = y_pred, y_true = y_test),precision_score(y_pred = y_pred, y_true = y_test),recall_score(y_pred = y_pred, y_true = y_test))

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality


# In[ ]:


x = data.drop('diagnosis',axis=1)
y = le.fit_transform(data['diagnosis'])
clf = SVC(kernel='linear', C=1000)
scores = cross_val_score(clf, x, y, cv=10, scoring='accuracy')
scores.mean()


# ## 3.0 Comparing different techniques and optimizing the algorithm

# ### 3.1 Using cross validation and fitting SVM, Decision Tree Classifier

# In[ ]:


x = data.drop('diagnosis',axis=1)
y = le.fit_transform(data['diagnosis'])
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=4)
metrics = pd.DataFrame(index = ['accuracy','precision','recall'],
                       columns = ['Tree','SVM'])
def crossval(model,parameters):
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train,y_train)
    y_pred = clf.best_estimator_.predict(X_test)
    accuracy = np.average(cross_val_score(clf, X_test, y_test, scoring='accuracy'))
    precision = np.average(cross_val_score(clf, X_test, y_test, scoring='precision'))
    recall = np.average(cross_val_score(clf, X_test, y_test, scoring='recall'))
    f1= np.average(cross_val_score(clf, X_test, y_test, scoring='f1'))
    if model==svm:
        metrics.loc['accuracy','SVM'] = accuracy
        metrics.loc['precision','SVM'] = precision
        metrics.loc['recall','SVM'] = recall
    if model==tree:
        metrics.loc['accuracy','Tree'] = accuracy
        metrics.loc['precision','Tree'] = precision
        metrics.loc['recall','Tree'] = recall
    return accuracy,precision,recall,f1,clf.best_estimator_,metrics


# In[ ]:


svm = SVC()
tree= DecisionTreeClassifier()
parameters = {'kernel':('linear', 'rbf'), 'C':(1,10,100),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
accuracy,precision,recall,f1,model,metrics = crossval(svm,parameters)
print(metrics)
parameters = {'max_depth':(1,6,12,15)}
accuracy,precision,recall,f1,model,metrics = crossval(tree,parameters)
print(metrics)


# ### 3.2 Accuracy, Precision and Recall values of SVM and Decision Tree Classifier

# In[ ]:


fig,ax = plt.subplots(figsize = (10,5))
metrics.plot(kind='barh', ax=ax)


# ### Since We are dealing with two different types of cancers and not True or False, I believe that our prediction has a good accuracy score and can be improved further

#  <h1><center><span style="color:red">END</span> </center></h1>
