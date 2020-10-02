#!/usr/bin/env python
# coding: utf-8

# ### Hello everyone ,this script contains implementation all the popular classifiers in scikit-learn and it is especially for beginners.The classifiers have been compared on the basis of Accuracy, Confusion  Matric and ROC score.
# **I have also done some outliers analysis on the dataset and it can really be very helpful for beginners.Please upvote it if you like it.**

# In[ ]:


import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
import math
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics     
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ## Reading Data

# In[ ]:


df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
le = preprocessing.LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition']) #Attrition column contains only object values.
df.head()


# **Selecting only non-object data type columns.We can one hot encode the categorical labels but for the comparison , I am just going with only numeric values.**

# In[ ]:


df = df.select_dtypes(exclude=['object'])


# In[ ]:


# Dividing the data into quantiles and doing the outlier analysis.

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


# In[ ]:


# Heatmap showing the correlation of various columns with each other.

ax = plt.figure(figsize = (20,10))
ax = sns.heatmap(df.corr(),cmap = 'gist_heat')


# *Here we can see that most of the features are having less correlation with each other, so we can take all these features together for training our model*

# In[ ]:


# The features with skewed or non-normal distribution.
skew_df = df[['MonthlyIncome','NumCompaniesWorked','PerformanceRating','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion']]
fig , ax = plt.subplots(3,3,figsize = (20,10))
col = skew_df.columns
for i in range(3):
    for j in range(3):
        ax[i][j].hist(skew_df[col[3 * i + j]] , color = 'k')
        ax[i][j].set_title(str(col[3 * i + j]))
        ax[i][j].set_axis_bgcolor((1, 0, 0))


# ### Lets look at the outliers on any one of the above features.

# In[ ]:


target = df['Attrition']
train = df.drop('Attrition',axis = 1)
train.shape


# In[ ]:


pd.value_counts(target).plot(kind = 'bar',cmap = 'BrBG')
plt.rcParams['axes.facecolor'] = 'blue'
plt.title("Count of classes")


# **Here we can see that one class dominates the other with a large ratio so the data is highly imbalanced.**

# *Even if we predict some of the classes wrong , we can still get a good accuracy so even a higher accuracy doesn't guarantee our classifier is well generalised for any sort of data.*

# **For the comparison, I have taken these classifiers**
# 1. *Logistic Regression*
# 1. *SGD*
# 1. *Perceptron*
# 1. *SVM*
# 1. *KNN*
# 1. *GaussianNB*
# 1. *Decision Tree*
# 1. *XgBoost*
# 1. *MLP* 
# 1. *K Means Clustering*

# **The performance of these classifiers depend on the properties of data.Hence if a classifier is not giving good results for this dataset, it doesn't mean that its not good because the distribution of the data determines the model to be selected.**

# In[ ]:


train_accuracy = []
test_accuracy = []
models = ['Logistic Regression' , 'SGD' , 'Perceptron' , 'SVM' , 'KNN' , 'GaussianNB' , 'Decision Tree' , 'XgBoost' , 'MLP' , 'K Means Clustering']


# In[ ]:


#Defining a function which will give us train and test accuracy for each classifier.
def train_test_error(y_train,y_test):
    train_error = ((y_train==Y_train).sum())/len(y_train)*100
    test_error = ((y_test==Y_test).sum())/len(Y_test)*100
    train_accuracy.append(train_error)
    test_accuracy.append(test_error)
    print('{}'.format(train_error) + " is the train accuracy")
    print('{}'.format(test_error) + " is the test accuracy")


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(
    train, target, test_size=0.33, random_state=42)


# In[ ]:


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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)
train_predict = log_reg.predict(X_train)
test_predict = log_reg.predict(X_test)
y_prob = log_reg.predict(train)
y_pred = np.where(y_prob > 0.5, 1, 0)
train_test_error(train_predict , test_predict)


# ### Lets have a look at the confusion matrix

# In[ ]:


class_names = ['0', '1']
confusion_matrix=metrics.confusion_matrix(target,y_pred)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# ### Lets look at the Roc curve

# In[ ]:


probs = log_reg.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc)) 


# ## SGD
# 

# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss="hinge", penalty="l2")
sgd.fit(X_train,Y_train)
train_predict = sgd.predict(X_train)
test_predict = sgd.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = sgd.predict(X_test)
#preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron
per = Perceptron(fit_intercept=False, n_iter=10, shuffle=False).fit(X_train,Y_train)
train_predict = per.predict(X_train)
test_predict = per.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = per.predict(X_test)
#preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## SVM

# In[ ]:


from sklearn import svm
SVM = svm.SVC(probability=True)
SVM.fit(X_train,Y_train)
train_predict = SVM.predict(X_train)
test_predict = SVM.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = SVM.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## KNN

# In[ ]:


from sklearn import neighbors
n_neighbors = 15
knn = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knn.fit(X_train,Y_train)
train_predict = knn.predict(X_train)
test_predict = knn.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = knn.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## GaussianNB

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,Y_train)
train_predict = gnb.predict(X_train)
test_predict = gnb.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = gnb.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## Decision Tree

# In[ ]:


from sklearn import tree
dec = tree.DecisionTreeClassifier()
dec.fit(X_train,Y_train)
train_predict = dec.predict(X_train)
test_predict = dec.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = dec.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## XGBOOST

# In[ ]:


import xgboost
from xgboost import XGBClassifier
XgB = XGBClassifier(max_depth=1,min_child_weight=1,gamma=0.0,subsample=0.8,colsample_bytree=0.75,reg_alpha=1e-05)
XgB.fit(X_train,Y_train)
train_predict = XgB.predict(X_train)
test_predict = XgB.predict(X_test)
train_test_error(train_predict,test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = XgB.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# In[ ]:


probs = XgB.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## Multi Layer Perceptron

# In[ ]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train,Y_train)
train_predict = mlp.predict(X_train)
test_predict = mlp.predict(X_test)
train_test_error(train_predict , test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = mlp.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# ## K-MEANS CLUSTERING

# In[ ]:


from sklearn.cluster import KMeans
kms = KMeans(n_clusters=2, random_state=1)
kms.fit(X_train,Y_train)
train_predict = kms.predict(X_train)
test_predict = kms.predict(X_test)
train_test_error(train_predict,test_predict)


# In[ ]:


confusion_matrix=metrics.confusion_matrix(Y_test,test_predict)
plot_confusion_matrix(confusion_matrix, classes=class_names,
                      title='Confusion matrix')


# In[ ]:


probs = kms.predict(X_test)
#preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, probs)
roc_auc = metrics.auc(fpr, tpr)
df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
ggplot(df, aes(x = 'fpr', y = 'tpr'))+ geom_line(aes(y = 'tpr')) + geom_abline(linetype = 'dashed') + geom_area(alpha = 0.1) + ggtitle("ROC Curve w/ AUC = %s" % str(roc_auc))


# In[ ]:


roc_score = np.asarray([0.7429,0.5,0.5,0.5978,0.6874,0.6005,0.7348,0.7348,0.5,0.4988])


# In[ ]:


results = DataFrame({"Roc score" : roc_score, "Test Accuracy" : test_accuracy , "Train Accuracy" : train_accuracy} , index = models)


# In[ ]:


results


# **Since Logistic Regression has the highest test accuracy and roc score, Logistic regression is the winner**
# *Runners up are  XgBoost and SVM*

# *Accuracy is not the correct measure in this cases so its better to compare roc score of the classifiers.Next time coming up with SMOTE analysis of dataset along with some data tuning and deep data exploration and new accuracy metric for imbalanced data*

# In[ ]:




