#!/usr/bin/env python
# coding: utf-8

# ## **Introduction**
# 
# The aim of this project is detecting fraudulent or non-fraudulent transactions while dealing with imbalanced data. To achieve this, various supervised learning algorithms will be used and the results will be compared. 
# 
# Imbalanced data refers to classification problems based on the binary class inequality. There are several methods for dealing with this problem like Re-Sampling, Generate Synthetic Samples, Anomaly Detection Methods or performance metrics instead of accuracy results. 
# 
# In this project, the undersampling method will be implemented to the majority class and performance metrics such as Precision, Recall, F1 Score and AUC and some anomaly detection methods like one-class SVM and Neural Network will be used to find the best algorithm which highly predicted fraudulent or non-fraudulent transactions.
# 
# The project has 4 main topics:
# 
# 1. Data Exploration
# 2. Hyperparameter Optimisation
# 3. Model Building
# 4. Comparing Performance Metrics

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')


# ## **1. Data Exploratory**
# 
# In this part; the structure of the data, missing values, features distribution and the relationship between them and target value characteristics will be examined in detail.
# 
# First data structure will be checked. 

# In[ ]:


data.head()


# In[ ]:


data.describe().transpose()


# In[ ]:


data.info()


# As mention on the data information section, except ``Time``, ``Amount`` and ``Class`` features others can not interpret alone. And they don't give information about context. But we all know that features which are from ``V1`` to ``V28`` have been dimensionally reduction by PCA and no need to be standardized again. But the other features which we have meaning in that data, it can be expanded on. 

# In[ ]:


# Next, Class feature will be examined.
plt.figure(figsize=(10,10))
sns.countplot(
    y="Class", 
    data=data,
    facecolor=(0, 0, 0, 0),
    linewidth=5, 
    edgecolor=sns.color_palette("dark", 2))

plt.title('Fraudulent Transaction Summary')
plt.xlabel('Count')
plt.ylabel('Fraudulent Transaction   Non-Fraudulent Transaction', fontsize=12)


# In[ ]:


data_value= data["Class"].value_counts()


# In[ ]:


print(data_value)
print(data_value/284807)


# The graph and tables show that there is a huge difference between non-fraudulent and fraudulent data. This situation can interpretable as imbalanced data. 
# Imbalanced data can cause classification problems like incorrect high accuracy. There are some approaches to avoid imbalanced data like oversampling, undersampling or Synthetic Data Generation. But in this project, I will use the undersampling method and values of the majority class will be reduced. Then I will compare models by performance metrics. 

# Another part is, ``Class`` structure will be converted to category and distribution of Time and Amount features will be examined. 

# In[ ]:


data['Class']= data['Class'].astype('category')


# In[ ]:


#Distribution of Time
plt.figure(figsize=(15,10))
sns.distplot(data['Time'])


# In[ ]:


#Distribution of Amount
plt.figure(figsize=(10,10))
sns.distplot(data['Amount'])


# Above graphs show that ``Time`` and ``Amount`` features needed to standardize.  Standardization  will be used to ``Time`` and ``Amount`` features for 0 mean and 1 std. This method preserves the shape of data and help to build features that have similar ranges to each other. 
# 
# Before standardization, I want to create a feature namely ``Hour`` which will help to examine ``Time`` feature and its relationship with ``Class`` and ``Amount`` data in a better way. 

# In[ ]:


data['Hour'] = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)


# In[ ]:


#Class vs Amount vs Hour
pd.pivot_table(
    columns="Class", 
    index="Hour", 
    values= 'Amount', 
    aggfunc='count', 
    data=data)


# In[ ]:


#Hour vs Class
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

sns.countplot(
    x="Hour",
    data=data[data['Class'] == 0], 
    color="#98D8D8",  
    ax=axes[0])
axes[0].set_title("Non-Fraudulent Transaction")


sns.countplot(
    x="Hour",
    data=data[data['Class'] == 1],
    color="#F08030", 
    ax=axes[1])
axes[1].set_title("Fraudulent Transaction")


# Above graphs show that non-fraudulent and fraudulent transactions have been made in every hour. For the fraudulent transaction in third and twelfth hours have the highest record. On the other hand, after the eighth hour, non-fraudulent transaction counts are nearly close to each other.

# In[ ]:


#Amount vs Hour vs Class
fig, axees = plt.subplots(2, 1, figsize=(15, 10))

plt.title("Non-Fraudulent Transactions")
sns.barplot(
    x='Hour',
    y='Amount', 
    data=data[data['Class'] == 0], 
    palette="ocean", 
    ax=axees[0])

plt.title("Fraudulent Transactions")
sns.barplot(
    x='Hour', 
    y='Amount', 
    data=data[data['Class'] == 1], 
    palette="Reds", 
    ax=axees[1])


# Above graphs show fraudulent and non-fraudulent transactions' amounts at 1 hour granularity. Based on the error bar, the amount of variation of the non-fraudulent transaction in each hour is not widely. However, in the fraudulent transaction, some data points especially first, sixth, and eleventh hours, the range of amounts is visible large. This means that there is a high difference in the amount varies between upper and lower limits.

# In[ ]:


#Drop hour feature before continues next analysis.
data=data.drop(['Hour'], axis=1)


# Data exploration results and graphs show that feature size is big and class sizes imbalanced, so, dimensionality reduction helps to an interpretation of results easier. To achieve this, t-distributed stochastic neighbor embedding(t-SNE) method will be used. This method is one of the dimensionality reduction technique to make visualization in a low-dimensional space. Thus we can look details more smooth. This technique works well on high dimensional data and converts it to two- or three- dimensional spot.

# In[ ]:


data_nonfraud = data[data['Class'] == 0].sample(2000)
data_fraud  = data[data['Class'] == 1]

data_new = data_nonfraud.append(data_fraud).sample(frac=1)
X = data_new.drop(['Class'], axis = 1).values
y = data_new['Class'].values


# In[ ]:


tsne = TSNE(n_components=2, random_state=42)
X_transformation = tsne.fit_transform(X)


# In[ ]:


plt.figure(figsize=(10, 10))
plt.title("t-SNE Dimensionality Reduction")

def plot_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Non_Fraudulent", alpha=0.5, linewidth=0.15, c='#17becf')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Fraudulent", alpha=0.5, linewidth=0.15, c='#d62728')
    plt.legend()
    return plt.show()

plot_data(X_transformation, y)


# Above graph shows that fraudulent and non-fraudulent transactions aren't well separated into two different clusters in the two-dimensional space. This led to two types of transactions are very similar. Also, this graph demonstrates that accuracy results won't be enough for choosing the best algorithm. 

# ### **Standardization**
# Standardization of Time and Amount features will be made. 

# In[ ]:


data[['Time', 'Amount']] = StandardScaler().fit_transform(data[['Time', 'Amount']])


# ### **Pearson Correlation Matrix**
# The final part is computing a correlation matrix by the Pearson method and analyze relationships between features.

# In[ ]:


corr=data.corr(method='pearson')


# In[ ]:


plt.figure(figsize=(18, 18))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(
    corr, 
    xticklabels=corr.columns,
    yticklabels=corr.columns, 
    cmap="coolwarm", 
    annot=True, 
    fmt=".2f",
    mask=mask, 
    vmax=.2, 
    center=0,
    square=True, 
    linewidths=1, 
    cbar_kws={"shrink": .5})


# The correlation matrix shows that almost all parameters have no strong relationship with each other. The highest correlation is negative and 53%. These results shows that there is no need to take out any feature from model building on the ground of high correlation. 

# ## **2. Hyperparameter Optimization **
# 
# This method helps to find the most optimal parameters for machine learning algorithms. It has crucial importance before proceeding to model training. The Grid Search algorithm will be used for the tuning hyperparameters. Then, XGBoost model will be built to achieve the feature importance graph. This graph helps to choose parameters which will be used on the training model. 

# In[ ]:


# First train and label data created. 
train_data, label_data = data.iloc[:,:-1],data.iloc[:,-1]

#Convert to matrix
data_dmatrix = xgb.DMatrix(data=train_data, label= label_data)


# In[ ]:


#Split data randomly to train and test subsets.
X_train, X_test, y_train, y_test = train_test_split(
                                    train_data, label_data, test_size=0.3,random_state=42)


# In[ ]:


## Defining parameters

#grid_param = {'n_estimators': [50, 100, 500],'max_depth': [4, 8], 
            #'max_features': ['auto', 'log2'], 
            #'criterion': ['gini', 'entropy'],
            #'bootstrap': [True, False]}

## Building Grid Search algorithm with cross-validation and F1 score.

#grid_search = GridSearchCV(estimator=xg_class,  
                     #param_grid=grid_param,
                     #scoring='f1',
                     #cv=5,
                     #n_jobs=-1)

## Lastly, finding the best parameters.

#grid_search.fit(X_train, y_train)
#best_parameters = grid_search.best_params_  
#print(best_parameters)


# Based on the result of GridSearch parameters, parameters will be defined and XGBoost algorithm can build. 

# In[ ]:


params = {
    'objective':'reg:logistic',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'bootstrap': True, 
    'criterion': 'gini', 
    'max_depth': 4, 
    'max_features': 'auto', 
    'n_estimators': 50
}
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

#Feature importance graph
plt.rcParams['figure.figsize'] = [20, 10]
xgb.plot_importance(xg_reg)


# Above graph shows that the highest important feature is ``V16`` and this feature has a great difference with the second important one based on F score. Lowest importance parameters are ``V13``,``V25``,``Time``,``V20``,``V22``,``V8``,``V15``,``V19``, and ``V2``. These variables will be eliminated from data before model building. 

# In[ ]:


data_model = data.drop(['V13', 'V25', 'Time', 'V20', 'V22', 'V8', 'V15', 'V19', 'V2'], axis=1)


# ## **3.Model Building**
# 
# In this part, Random Forest, Support Vector Machine, Logistic Regression, and Multilayer Perceptron - Neural Network algorithms will be built.
# 
# * Binary Support Vector Machine and Neural Network algorithms are one of the Anomaly Detection methods, therefore, they are chosen.
# * Since imbalance data have a predisposition to overfitting, Random Forest is one of the methods for preventing overfitting. For this reason, this method has been chosen. 
# * Logistic Regression is one of the important models when the target variable is binary. 
# 
# In this part, the important parameter is ``"class_weight"`` with balanced mode. It helps to adjust the model and this mode uses the values of y to automatically adjust weights. This adjustment method will help to get the best recall-precision trade-off.
# 
# Before the model building, the undersampling method will be applied. The output of this process will fed into model building phase.

# ### ** 3.1. Undersampling Method**
# One of the most common ways of dealing with imbalanced data is undersampling method. This method helps to decrease the number of majority class. In this project, %5 out of non-fraudulent data have been chosen. 

# In[ ]:


data_under_nonfraud = data_model[data_model['Class'] == 0].sample(15000)
data_under_fraud  = data_model[data_model['Class'] == 1]

data_undersampling = data_under_nonfraud.append(data_under_fraud, 
                                                ignore_index=True, sort=False)


# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(y="Class", data=data_undersampling,palette='Dark2')
plt.title('Fraudulent Transaction Summary')
plt.xlabel('Count')
plt.ylabel('Fraudulent Transaction,        Non-Fraudulent Transaction')


# ### ** 3.2. Data Splitting **

# In[ ]:


# New data will be split randomly to train and test subsets. Train data proportion is 70% and the test data proportion is 30%.

model_train, model_label = data_undersampling.iloc[:,:-1],data_undersampling.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(
                                        model_train, model_label, test_size=0.3, random_state=42)


# ### ** 3.3. K-Fold Cross Validation Method**

# In[ ]:


#5-fold Cross Validation method will be used.

kfold_cv=KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kfold_cv.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# ### **3.4. Random Forest**

# In[ ]:


# Define the model as the Random Forest
modelRF = RandomForestClassifier(
    n_estimators=500, 
    criterion = 'gini', 
    max_depth = 4, 
    class_weight='balanced', 
    random_state=42
).fit(X_train, y_train)

# Obtain predictions from the test data 
predict_RF = modelRF.predict(X_test)


# ### ** 3.5. Support Vector Machine**

# In[ ]:


# Define the model as the Support Vector Machine
modelSVM = svm.SVC(
    kernel='rbf', 
    class_weight='balanced', 
    gamma='scale', 
    probability=True, 
    random_state=42
).fit(X_train, y_train)

# Obtain predictions from the test data 
predict_SVM = modelSVM.predict(X_test)


# ### **3.6. Logistic Regression**

# In[ ]:


# Define the model as the Logistic Regression
modelLR = LogisticRegression(
    solver='lbfgs', 
    multi_class='multinomial',
    class_weight='balanced', 
    max_iter=500, 
    random_state=42
).fit(X_train, y_train)

# Obtain predictions from the test data 
predict_LR = modelLR.predict(X_test)


# ### **3.7. Neural Network - Multilayer Perceptron**

# In[ ]:


# Define the model as the Multilayer Perceptron
modelMLP = MLPClassifier(
    solver='lbfgs', 
    activation='logistic', 
    hidden_layer_sizes=(100,),
    learning_rate='constant', 
    max_iter=1500, 
    random_state=42
).fit(X_train, y_train)

# Obtain predictions from the test data 
predict_MLP = modelMLP.predict(X_test)


# ## **4.Comparing Performance Metrics**
# 
# In this part, instead of accuracy results, other performance metrics will be compared. Because, the highest accuracy results in imbalanced data may be achieved from non-fraudulent transaction predictions, thus, the results can be misleading for predictive modeling.
# 
# The following metrics are in interest;
# 
# * Confusion Matrix Fraud predictive part,
# * Precision,
# * Recall,
# * F1 Score and
# * AUC values 
# 
# Before starting to compare, I would like to explain some performance metrics:
# 
# * **Precision:** It explains, when the predicted value is 1, how often is it correct. 
# * **Recall:** It explains, when the actual value is 1, how often does it predict 1.
# * **F1 Score:** It explains, the weighted average of the recall and precision.
# * **AUC:** It explains, which model predicts the best classification. . 
# 
# Lastly, we can say that precision and recall are good metrics when the positive class is smaller. These metrics are good to detect positive samples accuratley.

# In[ ]:


RF_matrix = confusion_matrix(y_test, predict_RF)
SVM_matrix = confusion_matrix(y_test, predict_SVM)
LR_matrix = confusion_matrix(y_test, predict_LR)
MLP_matrix = confusion_matrix(y_test, predict_MLP) 


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 8))

sns.heatmap(RF_matrix, annot=True, fmt="d",cbar=False, cmap="Paired", ax = ax[0])
ax[0].set_title("Random Forest", weight='bold')
ax[0].set_xlabel('Predicted Labels')
ax[0].set_ylabel('Actual Labels')
ax[0].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])
ax[0].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

sns.heatmap(SVM_matrix, annot=True, fmt="d",cbar=False, cmap="Dark2", ax = ax[1])
ax[1].set_title("Support Vector Machine", weight='bold')
ax[1].set_xlabel('Predicted Labels')
ax[1].set_ylabel('Actual Labels')
ax[1].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])
ax[1].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])


fig, axe = plt.subplots(1, 2, figsize=(15, 8))

sns.heatmap(LR_matrix, annot=True, fmt="d",cbar=False, cmap="Pastel1", ax = axe[0])
axe[0].set_title("Logistic Regression", weight='bold')
axe[0].set_xlabel('Predicted Labels')
axe[0].set_ylabel('Actual Labels')
axe[0].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])
axe[0].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])

sns.heatmap(MLP_matrix, annot=True, fmt="d",cbar=False, cmap="Pastel1", ax = axe[1])
axe[1].set_title("Multilayer Perceptron", weight='bold')
axe[1].set_xlabel('Predicted Labels')
axe[1].set_ylabel('Actual Labels')
axe[1].yaxis.set_ticklabels(['Non-Fraud', 'Fraud'])
axe[1].xaxis.set_ticklabels(['Non-Fraud', 'Fraud'])


# Below graphs show that the confusion matrix result of each ML algorithm. For imbalanced data, confusion matrix results can be incorrect. However, it is useful to say how many fraudulent transactions predicted correctly.
# Based on the graphs, Multilayer Perceptron, Random Forest and Logistic Regression models predict much the same Fraudulent transaction.  

# In[ ]:


print("Classification_RF:")
print(classification_report(y_test, predict_RF))
print("Classification_SVM:")
print(classification_report(y_test, predict_SVM))
print("Classification_LR:")
print(classification_report(y_test, predict_LR))
print("Classification_MLP:")
print(classification_report(y_test, predict_MLP))


# Above table shows, precision, recall, and F1-score results. 
# 
# * Logistic Regression model has the highest recall. This means that the Logistic Regression model has a better prediction of an actual fraudulent transaction as a fraudulent transaction. 
# * However, when we look at the precision result, Logistic Regression is one of the lowest results. The highest one achieved with Random Forest. High precision relates to the low false positive rate, so we can say that Random Forest model predict the least false fraudulent transaction. 
# * F1-Score gives a better explanation on the grounds that it is calculated from the harmonic mean of Precision and Recall. Especially, the highest recall and lower precision situations. F1 Score is mostly better metrics to choose the best-predicted model. In light of this information, we can say that Random Forest is the best-predicted algorithms in all models. 

# Final comparing will be made with ROC Curve and AUC Score. ROC curve gives a good metric when the detection of both classes is equally important. With an AUC area, we can define the better classifier algorithm.

# In[ ]:


#RF AUC
rf_predict_probabilities = modelRF.predict_proba(X_test)[:,1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_predict_probabilities)
rf_roc_auc = auc(rf_fpr, rf_tpr)

#SVM AUC
svm_predict_probabilities = modelSVM.predict_proba(X_test)[:,1]
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_predict_probabilities)
svm_roc_auc = auc(svm_fpr, svm_tpr)

#LR AUC
lr_predict_probabilities = modelLR.predict_proba(X_test)[:,1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_predict_probabilities)
lr_roc_auc = auc(lr_fpr, lr_tpr)

#MLP AUC
mlp_predict_probabilities = modelMLP.predict_proba(X_test)[:,1]
mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_predict_probabilities)
mlp_roc_auc = auc(mlp_fpr, mlp_tpr)


# In[ ]:


plt.figure()
plt.plot(rf_fpr, rf_tpr, color='red',lw=2,
         label='Random Forest (area = %0.2f)' % rf_roc_auc)

plt.plot(svm_fpr, svm_tpr, color='blue',lw=2, 
         label='Support Vector Machine (area = %0.2f)' % svm_roc_auc)

plt.plot(lr_fpr, lr_tpr, color='green',lw=2, 
         label='Logistic Regression (area = %0.2f)' % lr_roc_auc)

plt.plot(mlp_fpr, mlp_tpr, color='orange',lw=2, 
         label='Multilayer Perceptron (area = %0.2f)' % mlp_roc_auc)

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# Based on the above ROC curve, we can say that, Logistic Regression, Random Forest  and Neural Network-Multilayer Perceptron algorithms have nearly similar AUC results. A great model has AUC near to the 1 which means it has a good measure of separability.
# 
# This conclusion can be demonstrated by ROC curve results as well. These algorithms leans towards True Positive Rate rather than False Positive Rate. As a result, we can say that these algorithms have better performance of classification. 

# Lastly, we can calculate, the average precision score for these 3 models. The results show all models have almost the same score.

# In[ ]:


print("Average precision score of Logistic Regression", average_precision_score(y_test, modelLR.predict_proba(X_test)[:,1]))
print("Average precision score of Random Forest", average_precision_score(y_test, modelRF.predict_proba(X_test)[:,1]))
print("Average precision score of Multilayer Perceptron", average_precision_score(y_test, modelMLP.predict_proba(X_test)[:,1]))


# ## **CONCLUSION**
# 
# In this project, I wanted to show how you can deal with imbalanced data and which performance metrics' particularly important compared to usual practices fairly balanced data. Also, I got a chance to use dimensionality reduction and expand on hyperparameter optimization techniques. I hope that this project can help other people who want to learn and work more about imbalanced data. 
# 
# Any comments and feedback are welcomed.
