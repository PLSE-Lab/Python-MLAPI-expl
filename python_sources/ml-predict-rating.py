#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


app_df=pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


app_df.head(10)


# In[ ]:


app_df.shape


# # Clean and Transfrom data
#     
# ### Keep columns maybe related to Rating, and clean them for example delete Nan value and change data type. Besides, it needs to transfrom some data like date and categroies for further model tuning.

# In[ ]:


#App, Category, Content rating, Genres, Current Ver and Android Ver are consided as less relative to Rating
#and Type & Price are repesent same meaning, so just keep one, use Price can also show whether it is free or not.
#we keep Category temporarily for later usage as groupby() function.
drop_list=['App','Content Rating', 'Genres', 'Current Ver', 'Android Ver','Type']
app_df2=app_df.drop(drop_list,axis=1)
app_df2.head()


# In[ ]:


app_df2.isna().sum()


# In[ ]:


app_df2.groupby('Category')['Rating'].mean()


# In[ ]:


#The mean of Rating with different Category is similar so just fill the null value with the mean of Rating
#instead of drop them.
app_df2.fillna(app_df2['Rating'].mean(),inplace= True)
app_df2.isna().sum()


# In[ ]:


#it also finds a abnormal value in Caategory = 1.9 and relative Rating is 19.0
app_df2[app_df2['Category']=='1.9']


# In[ ]:


#delete this abnormal recorde 
app_df2=app_df2[app_df2.Rating!=19.0]


# In[ ]:


#Here, we assume the purpose is to predict whether a possible rating is positive or not. 
#we can set that rating higher than 4 as a positive rating transformed as 1 and others as 0, 

app_df2.Rating=np.where(app_df2.Rating<4,0,1)


# In[ ]:


app_df2.Rating.value_counts()


# In[ ]:


#change date type to numerical.
app_df2.dtypes


# In[ ]:


#transform values in Installs to numerical by applying function delete '+' and ','
app_df2['Installs']=app_df2['Installs'].astype('str')
app_df2['Installs'] = app_df2['Installs'].apply(lambda x : x.strip('+').replace(',', ''))


# In[ ]:


#change data type of Installs and Reviews to int
app_df2['Installs']=app_df2['Installs'].astype(int)
app_df2['Reviews']=app_df2['Reviews'].astype(int)


# In[ ]:


app_df2.head()


# In[ ]:


#define a function to clean Size, first delete 'k' and 'M' and then transform to float
#change units of kb to Mb (1 Mb = 1024 kb) 
#'Various of device' change to NaN.
def k_transform(size):
    if 'k' in size:
        x=float(size.strip('k'))
        return round((x/1024),3)
    elif 'M' in size:
        x=float(size.strip('M'))
        return x
    else:
        size = np.nan
        return size


# In[ ]:


#apply k_transform to Size.
app_df2['Size']=app_df2['Size'].apply(k_transform)


# In[ ]:


#fill Nan value in Size with mean of different Category, by groupby()
#instead of drop them.
app_df2['Size'].fillna(app_df2.groupby('Category')['Size'].transform('mean'), inplace=True)


# In[ ]:


app_df2['Size'].isna().sum()


# In[ ]:


app_df2.head()


# In[ ]:


# delete $ in Price and transform to float.
app_df2.Price=app_df2.Price.astype('str')
app_df2.Price=app_df2.Price.apply(lambda x : x.strip('$'))
app_df2.Price=app_df2.Price.astype(float)


# In[ ]:


app_df2.Price.unique()


# In[ ]:


#Last updated can be transform to year only, set 2019 as 0, 2018 as 1, 2016 as 2 and so on
app_df2['Last Updated'] = app_df2['Last Updated'].str[-4:]


# In[ ]:


app_df2['Last Updated'].unique()


# In[ ]:


app_df2['Last Updated']= np.where(app_df2['Last Updated']=='2018',1,
                        np.where(app_df2['Last Updated']=='2017',2,
                        np.where(app_df2['Last Updated']=='2016',3,
                        np.where(app_df2['Last Updated']=='2015',4,
                        np.where(app_df2['Last Updated']=='2014',5,
                        np.where(app_df2['Last Updated']=='2013',6,
                        np.where(app_df2['Last Updated']=='2012',7,
                        np.where(app_df2['Last Updated']=='2011',8,9))))))))


# In[ ]:


app_df2['Last Updated'].unique()


# In[ ]:


#after clean and transform all columns, delete Category 
app_df2.drop('Category',inplace=True,axis=1)
app_df2.head()


# In[ ]:


app_df2.Rating.hist()


# #  Model Tuning
#     
# ### After cleaning and transforming data, we can move to model tuning, Rating is discrete data, binary model is suggested. here is going to use Decision tree, logistic Regression and K-NN as selected model.

# In[ ]:


from sklearn.model_selection import train_test_split
#split train data and test data
X_train, X_test, Y_train, Y_test = train_test_split(app_df2[['Reviews','Size','Installs','Price','Last Updated']], app_df2['Rating'], test_size = 0.2, random_state=5)

# print the shapes to check everything is OK
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'penalty': ['l1', 'l2'],
                     'C': [0.5, 1, 5, 10]}]

scores = ['precision', 'recall', 'f1']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, Y_train)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[ ]:


#use the best parameters for f1 socre
lr = LogisticRegression(C= 0.5, penalty = 'l2')

# fit the model using some training data
lr_fit = lr.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = lr.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))


# In[ ]:


predicted = lr.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = lr.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

class_names=np.array(['negative','positive'])

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# #  Imbalance
# 
# ### We can see from the matrix result, it is very good at predict positive rating, but bad at negative rating. That is beacuse the sample is imbalanced. Marjor class is usual better predicted. Here, we are going to use SMOTE method to handle this imbalanced data. Just oversmaple training data.

# In[ ]:


#install imblearn package firstly if it is not installed.
try:
    import imblearn
except ImportError:
    get_ipython().system('pip install -U imbalanced-learn')


# In[ ]:


# import SMOTE model and fit model with X_train and Y_train
#we just want to oversmaple training data and keep testing data unchanged.
from imblearn.over_sampling import SMOTE
import collections
X=X_train
Y=Y_train
X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
#print the count of each class (0 and 1 ).
print(sorted(collections.Counter(Y_resampled).items()))
#Now we have balanced class number.


# ##  Logistic Regression

# In[ ]:


tuned_parameters = [{'penalty': ['l1', 'l2'],
                     'C': [0.5, 1, 5, 10]}]

scores = ['precision', 'recall', 'f1']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    #use resampled data
    clf.fit(X_resampled, Y_resampled)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[ ]:


#use the best parameters for f1 socre
lr = LogisticRegression(C= 1, penalty = 'l1')

# fit the model using some resample data
lr_fit = lr.fit(X_resampled, Y_resampled)

# generate a mean accuracy score for the predicted data
train_score = lr.score(X_resampled, Y_resampled)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))


# In[ ]:


predicted = lr.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = lr.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[ ]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
#after oversampling, the prediction of negative improved highly.


# ##  Decision tree

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
tuned_parameters = [{'criterion': ['gini','entropy'],
                     'splitter':['random','best'],
                     'max_depth': [1,2,3], 
                     'max_features': ['sqrt','log2',None]}]
scores = ['precision', 'recall', 'f1']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5,
                       scoring= '%s_macro' % score)
    clf.fit(X_resampled, Y_resampled)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[ ]:


#use the best parameters for f1 socre
dec = DecisionTreeClassifier(criterion='gini',max_depth=2, max_features='sqrt', splitter='best')

# fit the model using some resample data
dec_fit = dec.fit(X_resampled, Y_resampled)

# generate a mean accuracy score for the predicted data
train_score = dec.score(X_resampled, Y_resampled)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))


# In[ ]:


predicted = dec.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = dec.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[ ]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# ##  K-NN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
tuned_parameters = [{'weights': ['uniform','distance'], 
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}]

scores = ['precision', 'recall', 'f1']

for score in scores:
    print("# Tuning hyperparameters for %s" % score)
    print("\n")
    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5,
                       scoring= '%s_macro' % score)
    clf.fit(X_resampled, Y_resampled)
    print("Best parameters set found on the training set:")
    print(clf.best_params_)
    print("\n")


# In[ ]:


#use the best parameters for f1 socre
knn = KNeighborsClassifier(algorithm= 'brute', weights='distance')

# fit the model using some training data
knn_fit = knn.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = knn.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))


# In[ ]:


predicted = knn.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = knn.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[ ]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# # Algorithm Voting
#     
# ### from matrix results of three algoritms, it can be seen that Logistic Regression is better at prediction of negative, while Decision Tree and K-NN are better at prediction of positive. Thence, we need build a voting model to imporve prediction on both.  

# In[ ]:


from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(C= 1, penalty = 'l1')
clf2 = DecisionTreeClassifier(criterion='gini',max_depth=2, max_features='sqrt', splitter='best')
clf3 = KNeighborsClassifier(algorithm= 'brute', weights='distance')
# 'soft' voting is choosed, because the result is better comparing with 'hard'.
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('dec',clf2),('knn', clf3)], voting='soft')
#fit with resampled data as usual.
eclf1 = eclf1.fit(X_resampled, Y_resampled)
train_score = eclf1.score(X_resampled, Y_resampled)
print("Accuracy score = " + str(round(train_score, 4)))


# In[ ]:


predicted = eclf1.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = eclf1.score(X_test, Y_test)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(test_score, 4)))


# In[ ]:


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# # Conclusion
# 
# ### After voting, we can see the result is imporved a lot than single algorithm, it predicts negative and positive both good.

# In[ ]:




