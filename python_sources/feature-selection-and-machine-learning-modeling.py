#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import preprocessing 


# In[ ]:


file = '../input/mushrooms.csv'
m_data = pd.read_csv(file)


# In[ ]:


m_data.head()


# We can simply LabelEncode all the features with categorical data.

# In[ ]:


m_data = m_data.apply(preprocessing.LabelEncoder().fit_transform)


# In[ ]:


m_data.head()


# In[ ]:


m_data.describe()


# we can observe that, we don't have missing values on the data.

# # Feature selection.
# We are going to see if the column to predict (class) have a high correlation with some other features, sorting it.

# In[ ]:


df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'class']


# we can see that "gill-size" are high correlated with class, we can simply search for features that are high correlated with gill-size, that can give us clues for redundant features that can give error or more training time to our predict models.

# In[ ]:


df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'gill-size']


# ## Method #1 Feature Selection KBest scores.
# SelectKBest class give us an idea of how important the features are, we give to the class the chi2 score function that stats of non-negative features for classification tasks.

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


y=m_data['class']
x=m_data.drop(['class'], axis=1)

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=11)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# ## Method #2 Model feature_importances.
# We can simply build a simply model an have a look for feature importances a weighted method for feature selection.

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

etmodel = ExtraTreesClassifier()
etmodel.fit(x,y)
feat_importances = pd.Series(etmodel.feature_importances_, index=x.columns).sort_values(kind="quicksort", ascending=False).reset_index()
print(feat_importances)


# ## Method #3 RFE (Recursive Feature Elimination).
# This method workd by recursively removing attributes and building a model on those features remaining, we choose that the method show us the 10 more relevant features in this dataset.
# The methods used for this tasks are:
# support_: Shows a boolean list that give what features to choose.
# ranking_: Shows a list with a ranking from 1 (important features) to N with N the number of features in the dataset - Number1 features.

# In[ ]:


# feature extraction
model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(x, y)
print("Num Features: {}".format(fit.n_features_))
print("Selected Features: {}".format(fit.support_))
print("Feature Ranking: {}".format(fit.ranking_))


# In[ ]:


x.columns


# In[ ]:


df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'gill-attachment']


# In[ ]:


df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'veil-color']


# In[ ]:


df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_all_corr[df_all_corr['Feature 1'] == 'ring-number']


# Gill-attachment feature don't have correlation with other feature that has relevation so we can put this as a 
# feature.
# Veil-color is a redundant feature because we have gil-attachment.

# The features that i have choose are from all the three method plus previously having removed the features with high correlation transitively.

# In[ ]:


features = ['gill-color', 'gill-attachment', 'ring-type', 'ring-number', 'gill-size', 'bruises', 'stalk-root',
            'gill-spacing', 'habitat', 'spore-print-color', 'stalk-surface-above-ring', 'class']

data_prefinal = m_data[features]


# # Machine learning modeling.
# we are going to compare some models with this data and choose those with good classification_report.

# In[ ]:


import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

x_all = data_prefinal.drop(['class'], axis=1)
y_all = data_prefinal['class']

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.33, random_state=23)


# In[ ]:


x_all.shape


# This function try to automate the comparing process between models and different metrics both in trainning and validation data.

# In[ ]:


def print_score(classifier,x_train,y_train,x_val,y_val,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(x_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(x_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(x_train))))
        res = cross_val_score(classifier, x_train, y_train, cv=10, n_jobs=-1, scoring='balanced_accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_val,classifier.predict(x_val))))
        print('Classification Report:\n{}\n'.format(classification_report(y_val,classifier.predict(x_val))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_val,classifier.predict(x_val))))


# We furthermore gridSearch for parameter tunning.
# For example in the first model Support Vector Machine i choose a invariant linear kernel but with search for Cs and gammas.

# In[ ]:


svcmodel = svm.SVC(kernel='linear', gamma='scale').fit(x_train, y_train)
svprediction = svcmodel.predict(x_val)

def svc_param_selection(X, y, nfolds):
    Cs = [0.001]
    gammas = [0.1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel = 'linear'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print(grid_search.best_params_) 
    return grid_search.best_estimator_

sv_best = svc_param_selection(x_train, y_train, 10)
sv_prediction = sv_best.predict(x_val)


# In[ ]:


print_score(sv_best, x_train, y_train, x_val, y_val, train=True)


# In[ ]:


print_score(sv_best, x_train, y_train, x_val, y_val, train=False)


# We see that the global score are high for both scores train and validation.

# In[ ]:


rf = RandomForestClassifier()
rfmodel = rf.fit(x_train, y_train)
prediction = rfmodel.predict(x_val)


# In[ ]:


print_score(rfmodel, x_train, y_train, x_val, y_val, train=True)


# In[ ]:


print_score(rfmodel, x_train, y_train, x_val, y_val, train=False)


# With this model we are reach the perfect classification accuracy and Confusion Matrix and validation scores support our hypothesis.

# In[ ]:


lrmodel = LogisticRegression()
lrmodel = lrmodel.fit(x_train, y_train)


# In[ ]:


print_score(lrmodel, x_train, y_train, x_val, y_val, train=True)


# In[ ]:


def knn_param_selection(X, y, nfolds):
    n_neighbors = [1, 2, 3, 4, 5, 6, 7]
    weights = ['distance', 'uniform']
    metric = ['euclidean', 'manhattan']
    param_grid = {'n_neighbors': n_neighbors, 'weights' : weights, 'metric': metric}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    print(grid_search.best_params_) 
    return grid_search.best_estimator_

knn_best = knn_param_selection(x_train, y_train, 10)


# In[ ]:


print_score(knn_best, x_train, y_train, x_val, y_val, train=True)


# In[ ]:


print_score(knn_best, x_train, y_train, x_val, y_val, train=False)


# We are gonna test a Neural Network for classfication purposes, we build our model with an input of 11 (number of features), and hidden layers with 15 and Dropout of 0.2 for generalization purposes, the output layer must be a sigmoid to see the correct values between 0 and 1.
# We compile with binary_crossentropy for binary classification and optimize with adamax, we can grid search the hidden layer nodes or the optimizer or simply the Dropout parameter, but the classification accuracy are good for this classification problem and doesn't need tunning.

# In[ ]:


nnmodel = Sequential()
nnmodel.add(Dense(15, input_dim = 11, activation='relu'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(15, activation='relu'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(15, activation='sigmoid'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(15, activation='relu'))
nnmodel.add(Dropout(0.2))
nnmodel.add(Dense(1, activation='sigmoid'))

nnmodel.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])


# In[ ]:


nnmodel.fit(x_train, y_train, batch_size=8100, epochs=2000)


# In[ ]:


y_pred=nnmodel.predict(x_val)
y_pred=(y_pred>0.5)


# In[ ]:


print(confusion_matrix(y_val, y_pred))


# In[ ]:


print(classification_report(y_val, y_pred))


# In[ ]:




