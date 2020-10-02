#!/usr/bin/env python
# coding: utf-8

# This kernel shows the implementation of 
# * Logistic regression for binary classification
# * Logistic regression CV with Lasso regularization
# * Logistic regression CV with Ridge Regularization
# * Using multiple binary classification for multi class classification using Logistic regression
# * Logistic regression multiclass classification
# * XG boost
# * XG boost with random search.
# 
# The dataset is data of employee transport. The aim for the kernel is to find car as the preference for transport and then extend it to predict the choice of transport for employees. 

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import unidecode
fig, ax=plt.subplots()
pd.set_option('display.max_columns', None)
sns.set(style='darkgrid',palette="muted")


# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]}', size=20)
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = 'df'
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


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/employeetransport/Cars.csv')
prediction_set=pd.read_excel('/kaggle/input/cars-prediction/predict.xlsx')


# In[ ]:


df.head()


# In[ ]:


df.describe().transpose()


# Checking for Null values

# In[ ]:


df.isna().any().any()


# Checking for Datatypes

# In[ ]:


df.dtypes


# Types of transport used by employees

# In[ ]:


df.Transport.value_counts()


# Male/Female counts

# In[ ]:


df.Gender.value_counts()


# Groupings by transport types

# In[ ]:


df.groupby('Transport').mean()


# In[ ]:


df.groupby('Transport').mean().plot(kind='bar')


# Distribution plots

# In[ ]:


plotPerColumnDistribution(df, 8, 4)


# Correlation plots

# In[ ]:


plotCorrelationMatrix(df, 7)


# Density plots

# In[ ]:


plotScatterMatrix(df, 10, 4)


# Pair plots

# In[ ]:


#pair plots of entire dataset
pp = sns.pairplot(df, hue = 'Transport', palette = 'deep', size=2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])


# In[ ]:


fig, axs=plt.subplots(nrows=1,ncols=3,figsize=(20,5))
sns.kdeplot(df.Salary[(df['Transport']=='Public Transport')],color="g",shade=True,ax=axs[0])
sns.kdeplot(df.Salary[(df['Transport']=='2Wheeler')],color="b",shade=True,ax=axs[0])
sns.kdeplot(df.Salary[(df['Transport']=='Car')],color="r",shade=True,ax=axs[0])
axs[0].legend(["Public trans","2 Wheeler","Cars"],loc='upper right')
axs[0].set_xlabel('Salary')
axs[0].set_title('Density Distribution of Salary',size=15)
axs[0].set_yticks([])
#########
sns.kdeplot(df['Work Exp'][(df['Transport']=='Public Transport')],color="g",shade=True,ax=axs[1])
sns.kdeplot(df['Work Exp'][(df['Transport']=='2Wheeler')],color="b",shade=True,ax=axs[1])
sns.kdeplot(df['Work Exp'][(df['Transport']=='Car')],color="r",shade=True,ax=axs[1])
axs[1].legend(["Public trans","2 Wheeler","Cars"],loc='upper right')
axs[1].set_xlabel('Work Exp')
axs[1].set_title('Density Distribution of Work Exp',size=15)
axs[1].set_yticks([])
#########
sns.kdeplot(df['Age'][(df['Transport']=='Public Transport')],color="g",shade=True,ax=axs[2])
sns.kdeplot(df['Age'][(df['Transport']=='2Wheeler')],color="b",shade=True,ax=axs[2])
sns.kdeplot(df['Age'][(df['Transport']=='Car')],color="r",shade=True,ax=axs[2])
axs[2].legend(["Public trans","2 Wheeler","Cars"],loc='upper right')
axs[2].set_xlabel('Age')
axs[2].set_title('Density Distribution of Age',size=15)
axs[2].set_yticks([])


# **Transport preference by profession**

# In[ ]:


fig, axs=plt.subplots(nrows=1,ncols=2,figsize=(20,5))
sns.countplot(x='Engineer',hue='Transport' ,data=df, ax=axs[0])
sns.countplot(x='MBA',hue='Transport' ,data=df, ax=axs[1])
fig.suptitle('Transport by Profession', fontsize=16)


# Converting to Numeric Types for Logistic regression

# In[ ]:


df['Male']=[1 if x=='Male' else 0 for x in df['Gender']]
df['Car']=[1 if x=='Car' else 0 for x in df['Transport']]
df['PublicTransport']=[1 if x=='Public Transport' else 0 for x in df['Transport']]
df['2Wheeler']=[1 if x=='2Wheeler' else 0 for x in df['Transport']]


# In[ ]:


df.head()


# In[ ]:


df.describe().transpose()


# **Correlation Matrix**

# In[ ]:


df.corr()


# In[ ]:


df.columns.values


# **Correlation w.r.t Use of Car**

# In[ ]:


dims = (8, 5)
sns.set(style='darkgrid',palette="muted")
fig, ax=plt.subplots(figsize=dims)
df.corr()['Car'][:-3].sort_values(ascending = False).plot(kind='bar')


# **Trying to analyze and predict usage of car.**

# In[ ]:


dfCar=df.loc[:,['Age', 'Engineer', 'MBA', 'Work Exp', 'Salary','Distance', 'license','Male','Car']]


# In[ ]:


X=dfCar.iloc[:,:-1]
y=dfCar.loc[:,'Car']


# **Declaring Functions**

# In[ ]:


#plot precicion, recall and thresholds
#predicted_proba[:,1]
def plotPrecisionRecallThreshold(y_test, pred_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_prob) 
   #retrieve probability of being 1(in second column of probs_y)
    pr_auc = metrics.auc(recall, precision)
    plt.title("Precision-Recall vs Threshold Chart")
    plt.plot(thresholds, precision[: -1], "b--", label="Precision")
    plt.plot(thresholds, recall[: -1], "r--", label="Recall")
    plt.ylabel("Precision, Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.ylim([0,1])
    
def plotROC(y_test,pred_prob):
    fpr, tpr, threshold=metrics.roc_curve(y_test,pred_prob)
    plt.title("ROC Curve")
    sns.lineplot(x=fpr,y=tpr,palette="muted")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    
def areaUnderROC(y_test, pred_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_prob) 
    return metrics.auc(recall, precision)


# **Logistic regression to predict preference for Cars**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=232)


# **First trying without parameterization, default lbfs solver is not supported without l2 parameterization, so using saga**

# In[ ]:


model = LogisticRegression(penalty='none', solver='saga')
result = model.fit(X_train, y_train)


# In[ ]:


from math import exp
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in model.coef_.flatten()])
print('coefficients')
print(model.coef_.flatten())
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Model gives an accuracy of 96%**

# Work Exp and Licence are the most importance positive for employees prefering cars, where as Age and MBA negatively show on preference of cars according to our logistic regression.

# **Plotting the confusion matrix**

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(2),range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .5")


# **Precision Recall curve**

# In[ ]:


predicted_proba=model.predict_proba(X_test)
plotPrecisionRecallThreshold(y_test, predicted_proba[:,1])


# **ROC Curve**

# In[ ]:


plotROC(y_test, predicted_proba[:,1])
print('area under the curve: %.2f' %areaUnderROC(y_test, predicted_proba[:,1]))


# **Logistic regression with LassoCV**

# In[ ]:


#Car: 65,95,50
#found these values with some hit and try
#Logit CV with ridge
#logistic regression CV. L1 Lasso
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=[21,23,24],cv=5,penalty='l1',solver='saga', random_state=232)
result = model.fit(X_train, y_train)


# In[ ]:


print('best regularization strength: %d'  %model.C_)


# In[ ]:


prediction_test = model.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in model.coef_.flatten()])
print('coefficients')
print(model.coef_.flatten())
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Accuracy: 95%**

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(2),range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .5")


# In[ ]:


plotROC(y_test, predicted_proba[:,1])
print('area under the curve: %.2f' %areaUnderROC(y_test, predicted_proba[:,1]))


# **Logistic regression with Ridge CV**

# In[ ]:


#l2
#car:1,5,50
#logistic regression CV. L2 Rigde
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=np.linspace(1,10,50),cv=5,penalty='l2', random_state=232)
result = model.fit(X_train, y_train)


# In[ ]:


print('best regularization strength: %d'  %model.C_)


# In[ ]:


prediction_test = model.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in model.coef_.flatten()])
print('coefficients')
print(model.coef_.flatten())
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Accuracy 96%**

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(2),range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .5")


# **Logistic regression with Ridge CV**

# In[ ]:


#l2
#car:1,5,50
#logistic regression CV. L2 Rigde
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(Cs=np.linspace(1,10,50),cv=5,penalty='l2', random_state=232)
result = model.fit(X_train, y_train)


# In[ ]:


print('best regularization strength: %d'  %model.C_)


# In[ ]:


prediction_test = model.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in model.coef_.flatten()])
print('coefficients')
print(model.coef_.flatten())
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Accuracy 96%**

# **Confusion Matrix**

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(2),range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .5")


# **ROC Curve**

# In[ ]:


plotROC(y_test, predicted_proba[:,1])
print('area under the curve: %.2f' %areaUnderROC(y_test, predicted_proba[:,1]))


# **Logistic regression with Elastic net**

# In[ ]:


modelCar = LogisticRegressionCV(Cs=np.linspace(1,50,50),cv=5,penalty='elasticnet',solver='saga',l1_ratios=np.linspace(0,1,10), random_state=232)
result = modelCar.fit(X_train, y_train)
print('best regularization strength: %d'  %model.C_)
print('l1_ration %.2f' %modelCar.l1_ratio_)


# In[ ]:


prediction_test = modelCar.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in modelCar.coef_.flatten()])
print('coefficients')
print(modelCar.coef_.flatten())
weights = pd.Series(modelCar.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Confusion Matrix**

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(2),range(2))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for threshold: .5")


# Highest accuracy is with elastic net. Will try elastic net for predicting public transport and 2wheeler as well

# In[ ]:


df2Wheeler=df.loc[:,['Age', 'Engineer', 'MBA', 'Work Exp', 'Salary','Distance', 'license','Male','2Wheeler']]
dfPubTrans=df.loc[:,['Age', 'Engineer', 'MBA', 'Work Exp', 'Salary','Distance', 'license','Male','PublicTransport']]


# **Modelling for 2 wheeler**

# In[ ]:


X=df2Wheeler.iloc[:,:-1]
y=df2Wheeler.loc[:,'2Wheeler']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=232)
model2wheel = LogisticRegressionCV(Cs=np.linspace(1e-2,10,50),cv=5,penalty='elasticnet',solver='saga',l1_ratios=np.linspace(0,1,10), random_state=232)
result = model2wheel.fit(X_train, y_train)
print('best regularization strength: %d'  %model2wheel.C_)
print('l1_ration %.2f' %model2wheel.l1_ratio_)


# In[ ]:


prediction_test = model2wheel.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in model2wheel.coef_.flatten()])
print('coefficients')
print(model2wheel.coef_.flatten())
weights = pd.Series(model2wheel.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Modeliing for public transport**

# In[ ]:


X=dfPubTrans.iloc[:,:-1]
y=dfPubTrans.loc[:,'PublicTransport']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=232)
modelPT = LogisticRegressionCV(Cs=np.linspace(1,20,50),cv=5,penalty='elasticnet',solver='saga',l1_ratios=np.linspace(0,1,10), random_state=232)
result = modelPT.fit(X_train, y_train)
print('best regularization strength: %d'  %modelPT.C_)
print('l1_ration %.2f' %modelPT.l1_ratio_)


# In[ ]:


prediction_test = modelPT.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in modelPT.coef_.flatten()])
print('coefficients')
print(modelPT.coef_.flatten())
weights = pd.Series(modelPT.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# Predicting on the prediction set using three logistic regression models above

# In[ ]:


prediction_set['Male']=[1 if x=='Male' else 0 for x in prediction_set['Gender']]
prediction_set


# **choosing columns to run the model on**

# In[ ]:


X_predict=dfCar=prediction_set.loc[:,['Age', 'Engineer', 'MBA', 'Work Exp', 'Salary','Distance', 'license','Male']]


# In[ ]:


modelPT.predict_proba(X_predict)


# In[ ]:


pCar=modelCar.predict_proba(X_predict)
p2wheel=model2wheel.predict_proba(X_predict)
pPubTrans=modelPT.predict_proba(X_predict)
pd.concat([pd.DataFrame(data=pCar, index=[1,2], columns=['NoCar','Car']),pd.DataFrame(data=p2wheel, index=[1,2], columns=['No2Wheel','2Wheel']),
          pd.DataFrame(data=pPubTrans, index=[1,2], columns=['NoPT','PT'])],axis=1)


# **Public transport has the Highest probability.**

# **Logistic regression with Multi classifier**

# In[ ]:


X=df.loc[:,['Age', 'Engineer', 'MBA', 'Work Exp', 'Salary','Distance', 'license','Male']]
y=df.Transport


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=232)
model = LogisticRegressionCV(Cs=np.linspace(1e-4,10,50),cv=5,penalty='elasticnet',solver='saga',l1_ratios=np.linspace(0,1,10), multi_class='multinomial', random_state=232)
result = model.fit(X_train, y_train)
print('best regularization strength:' ,model.C_)
print('l1_ratios',model.l1_ratio_)


# In[ ]:


prediction_test = model.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))
print('probalbilities')
print([exp(x)/(1+exp(x)) for x in modelPT.coef_.flatten()])
print('coefficients')
print(model.coef_.flatten())
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# **Accuracy for mulyiclass classification is 77%**
# license and gender remain as the most important features.

# Predicting out prediction set

# In[ ]:


model.predict(X_predict)


# Checking the probabilities

# In[ ]:


model.predict_proba(X_predict)
pd.DataFrame(data=model.predict_proba(X_predict), index=['1',2], columns=['Car','2Wheeler','PublicTransport'])


# The probabilities are similar to binary classifier results.

# **Trying xgboost to improve accuracy**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate   #Additional scklearn functions


# In[ ]:


#defining X as transport for XGBoost.
X=df.loc[:,['Age', 'Male', 'Engineer', 'MBA', 'Work Exp', 'Salary','Distance', 'license']]
y=df.loc[:,['Transport']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=232)
model=GradientBoostingClassifier(random_state=232)
model.fit(X_train, y_train)


# In[ ]:


prediction_test = model.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))


# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(3),range(3))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with XGBoost")


# **Hyper parameter tuning using Randomized Search**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


random_grid = {
    'learning_rate':[1e-4,1e-2,0.1,1],
    'max_depth':[2,5,15],
    'max_features':[2,4,7],
    'max_leaf_nodes':[2,8,15],
    'min_samples_leaf':[2,6,10],
    'min_samples_split':[2,6,10],
    'n_estimators':[100,200,300]}


# In[ ]:


xgb=GradientBoostingClassifier()
rf_random = RandomizedSearchCV(estimator = xgb, param_distributions =
random_grid, n_iter = 100, cv = 5, verbose=2, random_state=232,
n_jobs = -1)
rf_random.fit(X_train, y_train)


# In[ ]:


best_random=rf_random.best_estimator_


# In[ ]:


prediction_test = best_random.predict(X_test)
#df['logisticCVL1']=model.predict(X)
# Print the prediction accuracy
print('accuracy %.2f' %(metrics.accuracy_score(y_test, prediction_test)))


# accuracy improved from 81% to 84%

# In[ ]:


arr=metrics.confusion_matrix(y_test,prediction_test)
df_cm = pd.DataFrame(arr, range(3),range(3))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1)#for label size
sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix with XGBoost with Randomized search")


# Trying this to predict on our predictiokn set

# In[ ]:


best_random.predict(X_predict)


# Results are inline with logistic regression.

# 
