#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np 
import pandas as pd 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# Read database. Show data information: we can see that we do not have any missing values. 
# We have 28 principal components from the PCA (already applied in dataset)

# In[ ]:


df = pd.read_csv('../input/creditcard.csv')
print(df.info())


# **Exploratory data analysis:
# **First I plotted separately the amount of the transactions vs. time for fraudulent and non-fraudulent as well as the distributions of amounts for each case. The goal was to see if fraudulent transactions have higher or lower amounts (on average) than non-fraudulents and if they follow any temporal pattern.
# 

# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3)
ax[0].scatter(df['Time'][df['Class'] == 0], df['Amount'][df['Class']==0], color='b' )
ax[0].scatter(df['Time'][df['Class'] == 1], df['Amount'][df['Class']==1], color='r' , marker='.')
ax[0].legend(['non fraudulent', 'fraudulent'], loc='best')
ax[1].hist( df['Amount'][df['Class']==0], 100, facecolor='b', alpha=0.5, label="Distribution of amounts for non-fraudulent ", range = [0,2000])
ax[2].hist( df['Amount'][df['Class']==1], 100, facecolor='r', ec="black", lw=0.5, alpha=0.5, label="Distribution of amounts for fraudulent", range = [0,2000])
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Amount")
ax[1].set_xlabel("Amount")
ax[1].set_ylabel("Frequency")
ax[2].set_xlabel("Amount")
ax[2].set_ylabel("Frequency")
ax[1].set_title('non-fraudulent')
ax[2].set_title('fraudulent')
fig.subplots_adjust(left=0, right=2, bottom=0, top=1, hspace=0.05, wspace=0.5)


# We can see that all fraudulent amounts are below 2000 euros. The majority of non-fraudulent namounts are also below 2000 euros, although there are several large amounts. This is a sign that the amount won't be a very important variable to detect a non-fraudulent transaction.
# Let's analyze the correlation matrix. We know that all the PCs are going to have correlation = 0 because the PCA procedure obtains uncorrelated components. What we want to analyze here is which of the components are more correlated with the class, the time and amount.

# In[ ]:


import seaborn as sns
corr=df.corr()
mask=np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, mask=mask, vmax=.6, square=True)
plt.show()


# We can see that the amount of the transaction or the time is not correlated with the class. The most positively correlated PC with the class is V11 and V4, while the most negative correlated are V17 and V14. We will try to use this four PCs for the first simple model and then try more complicated models to detect the non-fraudulent transactions
# Finally, we evaluate the distributions of the four most correlated features and the two least correlated for both fraudulent and non-fraudulent cases.

# In[ ]:


import matplotlib.gridspec as gridspec
features=['V17','V14', 'V11', 'V4', 'V15', 'V13']
nplots=np.size(features)
plt.figure(figsize=(15,4*nplots))
gs = gridspec.GridSpec(nplots,1)
for i, feat in enumerate(features):
    ax = plt.subplot(gs[i])
    sns.distplot(df[feat][df.Class==1], bins=30)
    sns.distplot(df[feat][df.Class==0],bins=30)
    ax.legend(['fraudulent', 'non-fraudulent'],loc='best')
    ax.set_xlabel('')
    ax.set_title('Distribution of feature: ' + feat)


# We can see that the division between the two classes is almost impossible if we use V15 and V13
# 
# As the description says, only 0.172% of the total instances are frauds (492 out of 284.807). 
# Thus the dataset is highly unbalanced. To evaluate the accuracy of our classifier we will use the AUPRC.
# Now that we have done some exploratory analysis, we will start by building simple models with the dataset as it is. After that, we will try to oversample the 'fraud' class in order to balance the dataset and finally we will build more complicated models. I will perform a comparison of the model performances.
# 
# Now let's evaluate the range of values of the features and their average + std
# 

# In[ ]:


df.boxplot()
plt.ylim((-5,350))
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)


# We can see that 'amount' and 'time' need to be scaled... We will include a MinMax() scaler
# to scale all features from 0-1. Note this is not needed if we use DecisionTrees or RandomForests...
# 
# Finally, we will train a simple LogisticRegression model to see what the AUCPR is. Note this is not a very smart model for several reasons:
# First, the data is highly unbalanced. Second, we are assuming a linear classification problem. Finally, we are not performing any kind of feature selection.
# A smarter model will be applied in another kernel, where I will apply GMM to the non-fraudulent cases and detect anomalies by comparing the fraudulent test cases to the trained GMM. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Class'],axis=1), df['Class'], test_size=0.2, random_state=0, stratify = df['Class'])
print('Percentage of fraudulent in train ='+str(sum(y_train)/np.size(X_train,0)*100)+'%')
print('Percentage of fraudulent in test ='+str(sum(y_test)/np.size(X_test,0)*100)+'%')

#Now perform cross validation with grid search to find optimal parameter for the model.
#Note we apply this on training data. 
kfold = KFold(n_splits=3, random_state=1)     #Create 3-CV split object    
model=LogisticRegression()
pipe_lm = Pipeline([('minmax',MinMaxScaler()), ('lmodel',model )])
param_grid = [{'lmodel__C': [0.01, 0.1, 1.0]}]
clf = GridSearchCV(pipe_lm, param_grid, cv=kfold, scoring='average_precision')                #Nested-3fold-CV
outer_average_precission = cross_val_score(clf, X_train,y_train, scoring = 'average_precision', cv=kfold)              #Outer-3fold-CV
print('3-fold CV average AUCPR: %.3f +/- %.3f' % ( outer_average_precission.mean(), outer_average_precission.std()))    

#Fit the model with optimal parameter using all the training data to evaluate feature importance.                 
clf.fit(X_train, y_train)
lm_best_alpha=LogisticRegression(C=clf.best_params_['lmodel__C'], random_state=1)
pipe_lm_best = Pipeline([('minmax',MinMaxScaler()),  ('lmodel',lm_best_alpha )])
pipe_lm_best.fit(X_train,y_train)
feat_labels = df.columns[0:30]
importances=lm_best_alpha.coef_
indices = np.argsort(importances)[::-1]
importances=importances[0]
plt.figure()
plt.title('Logistic Regression coefficients')
plt.bar(range(np.size(feat_labels)),importances[indices[0]],color='lightblue',align='center')
plt.xticks(range(np.size(feat_labels)),feat_labels[indices[0]], rotation=90)
plt.tight_layout()
plt.show()


# We can see on the results above that the logistic regression coefficients for V4, V11, V13 and V12 are the highest (in absolute value). This means that they are more important for this model. Note this is not surprising since we saw in the EDA that these were the features that gave most separable distributions between fraudulent and non-fraudulent. 
# Finally, let's evaluate with more detail the classifier performance (AUPRC + PR curve + Precision + Recall) to see what should we do about the unbalance issue

# In[ ]:


from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print(clf.best_params_)
clf.best_estimator_.fit(X_train,y_train)
y_pred=clf.best_estimator_.predict(X_test)
print('Classification report')
print(classification_report(y_test,y_pred))
print('Test AUCPR = ' + str(average_precision_score(y_test, y_pred)))

precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format( average_precision_score(y_test, y_pred)))


# We can see that the performance of logistic regression is pretty low. In the next kernel, I'll apply a more advanced method to predict!

# In[ ]:




