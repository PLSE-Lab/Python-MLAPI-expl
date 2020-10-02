#!/usr/bin/env python
# coding: utf-8

# ## Context
# 
# This case is about a bank (Thera Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns to better target marketing to increase the success ratio with a minimal budget.
# 
# The department wants to build a model that will help them identify the potential customers who have a higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign.
# 
# ## Content
# 
# * Column descriptions
# * ID Customer ID
# * Age Customer's age in completed years
# * Experience #years of professional experience
# * Income Annual income of the customer (USD 1000)
# * ZIPCode Home Address ZIP code.
# * Family Family size of the customer
# * CCAvg Avg. spending on credit cards per month (USD 1000)
# * Education Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional
# * Mortgage Value of house mortgage if any. (USD 1000)
# * Personal Loan Did this customer accept the personal loan offered in the last campaign?
# * Securities Account Does the customer have a securities account with the bank?
# * CD Account Does the customer have a certificate of deposit (CD) account with the bank?
# * Online Does the customer use internet banking facilities?
# * CreditCard Does the customer uses a credit card issued by UniversalBank?
# 
# ## Objective
# 
# Build a model that will help them identify the potential customers who have a higher probability of purchasing the loan.
# 
# 

#  

# ### Import Library

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import Dataset
# 
# Import the dataset in pandas dataframe form and display the 5 top and bottom of the dataset

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


original = pd.read_excel('../input/Bank_Personal_Loan_Modelling.xlsx',"Data")


# In[ ]:


feature=original.drop("Personal Loan",axis=1)
target=original["Personal Loan"]

loans = feature.join(target)


# In[ ]:


loans.head(5)


# In[ ]:


loans.tail(5)


# ## Exploratory Data Analysis

# ### Detailed info

# In[ ]:


listItem = []
for col in loans.columns :
    listItem.append([col,loans[col].dtype,
                     loans[col].isna().sum(),
                     round((loans[col].isna().sum()/len(loans[col])) * 100,2),
                    loans[col].nunique(),
                     list(loans[col].sample(5).drop_duplicates().values)]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
dfDesc


# ### Data Cleaning

# ### Missing value visualization

# In[ ]:


sns.heatmap(loans.isna(),yticklabels=False,cbar=False,cmap='viridis')


# ### Irregular value analysis

# In[ ]:


loans.describe().transpose()


# ### Irregular value visualization

# In[ ]:


outvis = loans.copy()
def fungsi(x):
    if x<0:
        return np.NaN
    else:
        return x
    
outvis["Experience"] = outvis["Experience"].apply(fungsi)


# In[ ]:


sns.heatmap(outvis.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### 2nd Irregular value analysis

# In[ ]:


pd.DataFrame(loans.groupby("Education").mean()["Experience"])


# In[ ]:


pd.DataFrame(loans.groupby("Age").mean()["Experience"]).tail(8)


# In[ ]:


pltdf = pd.DataFrame(loans.groupby("Age").mean()["Experience"]).reset_index()
sns.lmplot(x='Age',y='Experience',data=pltdf)
plt.ylabel("Experience (Average)")
plt.title("Average of Experience by Age")
plt.show()


# In[ ]:


pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age")).head()


# In[ ]:


pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age"))["Age"].unique()


# In[ ]:


pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age"))["Experience"].unique()


# ### Irregular value handling feature 1

# In[ ]:


loans["Experience"] = loans["Experience"].apply(abs)


# In[ ]:


# def fungsi(x):
#     if x<0:
#         return np.NaN
#     else:
#         return x
    
# loans["Experience"] = loans["Experience"].apply(fungsi)

# loans.dropna(inplace=True)


# In[ ]:


# def fungsi(x):
#     if x== -1:
#         return 2
#     elif x== -2:
#         return 1
#     elif x== -3:
#         return 0
#     else:
#         return x
    
# loans["Experience"] = loans["Experience"].apply(fungsi)


# In[ ]:


loans.describe().transpose()


# ### Data type analysis
# 
# It's important to make sure that each feature already use correct data type. Please make sure all features already use correct data type based on whether the feature is categorical or numerical. Please change:
# 1. categorical feature into 'int64', and
# 2. numerical feature into 'float64'

# In[ ]:


# loans.info()


# Categorical feature:
#     
#     ordinal:
#     -Family
#     -Education
#     
#     nominal:
#     -ID
#     -Zip Code
#     -Securities Account
#     -CD Account
#     -Online
#     -Credit Card
# 
# Numerical feature:
#     
#     Interval or Ratio:    
#     -Age
#     -Experience
#     -Income
#     -CCAvg
#     -Mortage

# In[ ]:


loans[["Age","Experience","Income","CCAvg","Mortgage"]] = loans[["Age","Experience","Income","CCAvg","Mortgage"]].astype(float)


# In[ ]:


loans.info()


# ### Feature correlation analysis
# 
# The purpose of this section to find if there are possibility of multi-correlation between features and in the same time to get insight about which features (X) that have good correlation with our target (y).
# 

# In[ ]:


feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]


# ### 1. Heatmap correlation
# 
# Heatmap is one of simplest method to analyze feature correlation.
# 1. Heatmap correlation with only features (X) - we need to know correlation between features and avoid multi-correlation features,
# 2. Heatmap correlation with features (X) and target (y) - we need to know which features that have good correlation with our target,

# In[ ]:


# plt.figure(figsize=(10, 10))
# sns.heatmap(feature.corr(),annot=True,square=True)


# In[ ]:


#https://seaborn.pydata.org/generated/seaborn.heatmap.html

corr = feature.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))
sns.heatmap(corr, mask=mask,annot=True,square=True)


# In[ ]:


# plt.figure(figsize=(10, 10))
# sns.heatmap(feature.join(target).corr(),annot=True,square=True)


# In[ ]:


# plt.figure(figsize=(20, 20))
# sns.pairplot(feature.join(target).drop(["ZIP Code"],axis=1),hue="Personal Loan")


# In[ ]:


loans_corr = feature.join(target).corr()

mask = np.zeros((13,13))
mask[:12,:]=1

plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    sns.heatmap(loans_corr, annot=True,square=True,mask=mask)


# ### Distribution analysis

# In[ ]:


sns.distplot(feature["Mortgage"])
plt.title("Mortgage Distribution with KDE")


# ### Irregular value handling feature 2 (extreme positive skewed data)

# In[ ]:


SingleLog_y = np.log1p(feature["Mortgage"])              # Log transformation of the target variable
sns.distplot(SingleLog_y, color ="r")
plt.title("Mortgage Distribution with KDE First Transformation")


# In[ ]:


DoubleLog_y = np.log1p(SingleLog_y)
sns.distplot(DoubleLog_y, color ="g")
plt.title("Mortgage Distribution with KDE Second Transformation")


# In[ ]:


loans["Mortgage"] = DoubleLog_y


# ### Distribution analysis 2

# In[ ]:


source_counts =pd.DataFrame(loans["Personal Loan"].value_counts()).reset_index()
source_counts.columns =["Labels","Personal Loan"]
source_counts


# In[ ]:


#https://matplotlib.org/gallery/pie_and_polar_charts/pie_features.html

fig1, ax1 = plt.subplots()
explode = (0, 0.15)
ax1.pie(source_counts["Personal Loan"], explode=explode, labels=source_counts["Labels"], autopct='%1.1f%%',
        shadow=True, startangle=70)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Personal Loan Percentage")
plt.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['Income'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['Income'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("Income Distribution")


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['CCAvg'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['CCAvg'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("CCAvg Distribution")


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['Experience'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['Experience'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("Experience Distribution")


# In[ ]:


sns.countplot(x='Securities Account',data=loans,hue='Personal Loan')
plt.title("Securities Account Countplot")


# In[ ]:


sns.countplot(x='Family',data=loans,hue='Personal Loan')
plt.title("Family Countplot")


# In[ ]:


sns.boxplot(x='Education',data=loans,hue='Personal Loan',y='Income')
plt.legend(loc='lower right')
plt.title("Education and Income Boxplot")


# In[ ]:


sns.boxplot(x='Family',data=loans,hue='Personal Loan',y='Income')
plt.legend(loc='upper center')
plt.title("Family and Income Boxplot")


# ### Feature Selection
# 
# Based on "Feature correlation analysis" & "Distribution analysis" you can throw away some unnecessary features or even you want to add new feature. Please do some handling about feature selection (selecting necessary features) and state your reason for such handling.

# In[ ]:


feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]


# ### Features Removing

# In[ ]:


feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)


# ### Adding New Feature

# In[ ]:


# feature["Combination"] = (feature["Income"]/12)-feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)/feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)*feature["CCAvg"]
feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.30,
                                                    random_state=101)


# In[ ]:


y_train.value_counts()


# In[ ]:


from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=101)
xgb.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, confusion_matrix, accuracy_score,recall_score


# ### XGBClassifier model  with imbalance dataset evaluation

# In[ ]:


predict = xgb.predict(X_test)
predictProb = xgb.predict_proba(X_test)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=101)

X_ros, y_ros = ros.fit_sample(X_train, y_train)


# In[ ]:


pd.Series(y_ros).value_counts()


# In[ ]:


xgb = XGBClassifier(n_estimators=97,random_state=101)
xgb.fit(X_ros, y_ros)


# ### XGBClassifier model with balance dataset (train evaluation)

# In[ ]:


predict = xgb.predict(X_train.values)
predictProb = xgb.predict_proba(X_train.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_train, predict))
print("\nclassification_report :\n",classification_report(y_train, predict))
print('Recall Score',recall_score(y_train, predict))
print('ROC AUC :', roc_auc_score(y_train, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_train, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_train, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# ### XGBClassifier model with balance dataset (test evaluation)

# In[ ]:


predict = xgb.predict(X_test.values)
predictProb = xgb.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# In[ ]:





# In[ ]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,
                                                       X=feature,
                                                       y=target,
                                                       train_sizes=np.linspace(0.01, 1.0, 10),
                                                       cv=10)

print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print(train_mean)
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[ ]:


# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Train and Test Accuracy Comparison")
plt.show()


# In[ ]:


coef1 = pd.Series(xgb.feature_importances_,feature.columns).sort_values(ascending=False)

pd.DataFrame(coef1,columns=["Features"]).transpose().plot(kind="bar",title="Feature Importances") #for the legends

coef1.plot(kind="bar",title="Feature Importances")


# ### Other Model

# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier(min_samples_leaf=10)
dtree.fit(X_ros,y_ros)


# In[ ]:


predict = dtree.predict(X_test.values)
predictProb = dtree.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier(n_estimators=101,max_depth=250,max_leaf_nodes=50,random_state=101)
rfc.fit(X_ros,y_ros)


# In[ ]:


predict = rfc.predict(X_train.values)
predictProb = rfc.predict_proba(X_train.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_train, predict))
print("\nclassification_report :\n",classification_report(y_train, predict))
print('Recall Score',recall_score(y_train, predict))
print('ROC AUC :', roc_auc_score(y_train, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_train, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_train, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# In[ ]:


predict = rfc.predict(X_test.values)
predictProb = rfc.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# In[ ]:


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=rfc,
                                                       X=feature,
                                                       y=target,
                                                       train_sizes=np.linspace(0.01, 1.0, 10),
                                                       cv=10)

print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print(train_mean)
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[ ]:


# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Train and Test Accuracy Comparison")
plt.show()


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_ros,y_ros)


# In[ ]:


predict = knn.predict(X_test.values)
predictProb = knn.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,100):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_ros,y_ros)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_ros,y_ros)


# In[ ]:


predict = logmodel.predict(X_test.values)
predictProb = logmodel.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())


# In[ ]:


import keras


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# In[ ]:


features=original.drop("Personal Loan",axis=1)
targets=original["Personal Loan"]

loans = features.join(targets)


# In[ ]:


loans.shape


# In[ ]:


import random


for i in range(1,7):
    copy = loans
    copy['Income']=copy['Income']+random.gauss(1,10) # add noice to income
    loans=loans.append(copy,ignore_index=True) # make voice df 2x as big
    print("shape of df after {0}th intertion of this loop is {1}".format(i,loans.shape))


# In[ ]:


loans["Experience"] = loans["Experience"].apply(abs)


# In[ ]:


feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]

feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)

# feature["Combination"] = (feature["Income"]/12)-feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)/feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)*feature["CCAvg"]
feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]

from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)

from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.30,
                                                    random_state=101)

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=101)

X_ros, y_ros = ros.fit_sample(X_train, y_train)


# In[ ]:


clf = Sequential([
    Dense(units=72, kernel_initializer='uniform', input_dim=9, activation='relu'),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dropout(0.1),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dropout(0.1),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])

clf.summary()


# In[ ]:


# from tensorflow.python.keras.callbacks import EarlyStopping


# In[ ]:


import keras.backend as K


# In[ ]:


def recall(y_true, y_pred): 

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall


# In[ ]:


clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=[recall,'accuracy'])


# In[ ]:


history= clf.fit(X_ros, y_ros, batch_size=50, epochs=25)


# In[ ]:


original = pd.read_excel('../input/Bank_Personal_Loan_Modelling.xlsx',"Data")

feature=original.drop("Personal Loan",axis=1)
target=original["Personal Loan"]

loans = feature.join(target)

loans["Experience"] = loans["Experience"].apply(abs)

feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]

feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)

# feature["Combination"] = (feature["Income"]/12)-feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)/feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)*feature["CCAvg"]
feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]

from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)

from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.30,
                                                    random_state=101)


# In[ ]:


score = clf.evaluate(X_test, y_test, batch_size=128)
print('\nAnd the Test Score is ',"\nRecall :", score[1],"\nAccuracy :",score[2])


# In[ ]:


predictProb = pd.DataFrame(clf.predict(X_test.values))


def fungsi(x):
    if x<0.5:
        return 0
    else:
        return 1

predict = predictProb[0].apply(fungsi)


# In[ ]:


# score1 =cross_val_score(X=X_train,y=y_train,estimator=clf,scoring="recall")
# score2 =cross_val_score(X=X_train,y=y_train,estimator=clf,scoring="roc_auc")

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[0]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))


# In[ ]:


history_dict = history.history
history_dict.keys()


# In[ ]:


#plot the metrics during training. 
epochs = range(1, len(history_dict['loss']) + 1)

plt.plot(epochs, history_dict['acc'], 'r',label='acc')
plt.plot(epochs, history_dict['recall'], 'b',label='recall')

plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()

