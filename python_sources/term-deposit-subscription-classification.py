#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#Imporing necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import imblearn


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 
# ## Reading data file and cheking data

# In[ ]:


#reading source file
data=pd.read_csv("/kaggle/input/bank-full.csv")


# In[ ]:


#to check the head of the data-frame
data.head(10)


# 
# ## Attribute information
# ### Input variables:
# 
# #### 1 - age (numeric)
# #### 2 - job : type of job (categorical)
# #### 3 - marital : marital status (categorical)
# #### 4 - education (categorical)
# #### 5 - default: has credit in default? (categorical)
# #### 6 - balance(Numeric)
# #### 7 - housing: has housing loan? (categorical)
# #### 8 - loan: has personal loan? (categorical)
# #### 9 - contact: contact communication type (categorical) 
# #### 10 - day: last contact day of the month (numberical)
# #### 11 - month: last contact month of year (categorical)
# #### 12 - duration: last contact duration, in seconds (numeric). 
# 
# ##### Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 
# #### 13 - campaign: number of contacts performed during this campaign and for this client (numeric)
# #### 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted)
# #### 15 - previous: number of contacts performed before this campaign and for this client (numeric)
# #### 16 - poutcome: outcome of the previous marketing campaign (categorical)
# 
# ### Output variable (desired target):
# #### 21 - Target - has the client subscribed a term deposit? (binary: 'yes','no')
# 

# In[ ]:


#checking the dtypes of the data
data.dtypes


# 
# ##### There are object data-types in the data-set which would need conversion at the latter stage of our analysis

# In[ ]:


#Checking the information of the data set
data.info()


# In[ ]:


#Checking the shape of the data-set and the target column
print(data.shape)
data['Target'].value_counts()


# ##### We can infer that the data-set has 45211 records with 16 independent variables and 1 target variable where as 39922 people have not subscribed to term deposit whereas 5289 people have subscribed to term deposit

# In[ ]:


#To check if there are any null values present
nulllvalues=data.isnull().sum()
print(nulllvalues)


# ##### There are no null values present in the data-set

# In[ ]:


#To check if there are any NaN values present
NaNvalues=data.isna().sum()
print(NaNvalues)


# ##### There are no NaN values present in the data-set

# In[ ]:


#Changing Target to numerical representation to use in EDA
Target_dict={'yes':1,'no':0}

data['Target']=data.Target.map(Target_dict)

data.head()


# In[ ]:


#To describe the data- Five point summary
data.describe().T


# ##### Mean of Campaign, Previous, Balance,  Duration,  pdays is much more than the median which infers that they have outliers

# ## Exploratory Data Analytics

# ### Univariate Analysis

# In[ ]:


#Distribution of continous data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('Age')
sns.distplot(data['age'],color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('Balance')
sns.distplot(data['balance'],color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('Duration')
sns.distplot(data['duration'],color='green')



plt.figure(figsize=(30,6))

#Subplot 1- Boxplot
plt.subplot(1,3,1)
plt.title('Age')
sns.boxplot(data['age'],orient='horizondal',color='red')

#Subplot 2
plt.subplot(1,3,2)
plt.title('Balance')
sns.boxplot(data['balance'],orient='horizondal',color='blue')

#Subplot 3
plt.subplot(1,3,3)
plt.title('Duration')
sns.boxplot(data['duration'],orient='horizondal',color='green')


# ##### Average age is between 30 and 50 Years and there are some outliers
# ##### Average duration is between 0 and 800. Huge number of outliers and are right skewed
# #####  Balance  is right skewed and have huge number of outliers 

# #Distribution of continous data
# 
# plt.figure(figsize=(30,6))
# 
# #Subplot 1
# plt.subplot(1,3,1)
# plt.title('Campaign')
# sns.distplot(data['campaign'],color='red')
# 
# #Subplot 2
# plt.subplot(1,3,2)
# plt.title('P-days')
# #sns.distplot(data['pdays'],color='blue')
# 
# #Subplot 3
# plt.subplot(1,3,3)
# plt.title('Previous')
# sns.distplot(data['previous'],color='green')
# 
# 
# 
# plt.figure(figsize=(30,6))
# 
# #Subplot 1- Boxplot
# plt.subplot(1,3,1)
# plt.title('Campaign')
# sns.boxplot(data['campaign'],orient='horizondal',color='red')
# 
# #Subplot 2
# plt.subplot(1,3,2)
# plt.title('P-days')
# #sns.boxplot(data['pdays'],orient='horizondal',color='blue')
# 
# #Subplot 3
# plt.subplot(1,3,3)
# plt.title('Previous')
# sns.boxplot(data['previous'],orient='horizondal',color='green')
# 

# ##### Average campaigns are around 2 with good number of outliers
# #### pdays and previous data have huge number of outliers

# In[ ]:


# Distribution of Categorical data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('Contact')
sns.countplot(data['contact'],color='cyan')

#Subplot 2
plt.subplot(1,3,2)
plt.title('Education')
sns.countplot(data['education'],color='violet')

#Subplot 3
plt.subplot(1,3,3)
plt.title('Marital')
sns.countplot(data['marital'],color='green')

plt.figure(figsize=(30,6))

#Subplot 4
plt.subplot(1,3,1)
plt.title('Default')
sns.countplot(data['default'],color='red')

#Subplot 5
plt.subplot(1,3,2)
plt.title('Housing')
sns.countplot(data['housing'],color='blue')

#Subplot 6
plt.subplot(1,3,3)
plt.title('Loan')
sns.countplot(data['loan'],color='orange')


# #### Cellular way of contact is higher than other methods
# #### Most of the customers have Secondary education followed by Tertiary education
# #### Most of the customers in this data set are married
# #### Most of the customers have not defaulted on their credit
# #### People have housing loan is more
# #### People not having personal loan is higher than people having personal loan

# In[ ]:


# Distribution of Categorical data

plt.figure(figsize=(30,6))

#Subplot 1
plt.subplot(1,3,1)
plt.title('Day')
sns.countplot(data['day'],color='cyan')

#Subplot 2
plt.subplot(1,3,2)
plt.title('Month')
sns.countplot(data['month'],color='violet')

#Subplot 3
plt.subplot(1,3,3)
plt.title('Poutcome')
sns.countplot(data['poutcome'],color='green')


# #### May was the most last contacted month
# #### Most of the outcome of the previous capaign was others

# In[ ]:


data['job'].value_counts().head(30).plot(kind='bar')


# ##### Blue collar and Management jobs are the highest followed by Technician

# In[ ]:


# Distribution of Target column
sns.countplot(data['Target'])


# #### Target column is highly imbalanced with only close the 5000 people have taken term deposits

# #### Around 5000 people have taken term deposits and 40000 have not taken term deposits. That is around 12.5% success rate.

# ## Bi-variate Aanalysis

# In[ ]:


sns.catplot(x='Target',y='age', data=data)


# #### Age doesnot seem to influence the target

# In[ ]:


sns.catplot(x='Target',y='balance', data=data)


# #### Balance has a slight influence on the target

# In[ ]:


sns.catplot(x='Target',y='duration', data=data)


# #### Duration does have good corelation with the Target

# In[ ]:


sns.catplot(x='Target',y='campaign', data=data)


# #### More the number of campaigns, lesser the customers who have subscribed to Term deposits

# In[ ]:


sns.catplot(x='Target',y='pdays', data=data)


# #### pdays doesnot seem to influence the target

# In[ ]:


sns.catplot(x='Target',y='previous', data=data)


# #### previous doesnot seem to influence the target

# In[ ]:


sns.countplot(x='education',hue='Target', data=data)


# ##### we can infer that people with secondary  and tertiary education  opt for term deposit comparitively

# In[ ]:


sns.violinplot(x="Target", y="duration", data=data,palette='rainbow')


# ##### Increase in duration of last call shows the variation in target output column

# In[ ]:


sns.catplot(x='marital',hue='Target',data=data,kind='count',height=4)


# ##### People with matital status as "single" invest in term deposit by their total percentage
# 

# ## Multi-Variate Analysis

# In[ ]:


sns.pairplot(data, palette="Set2")


# In[ ]:


#To find the correlation between the continous variables
correlation=data.corr()
correlation.style.background_gradient(cmap='coolwarm')


# In[ ]:


sns.heatmap(correlation)


# ##### Correlation between pdays and previous column is better where as all other independent columns has very less correlation
# ##### There are no strong linear relationships between any two variables except Target and duration
# 

# # Preparing the data for analytics

# In[ ]:


data.head()
data['Target']=data['Target'].astype('object')
data.head()


# ## Strategy 1: Removing the outliers from numerical columns
# ### Identifying the z-score for numerical columns

# In[ ]:


integers = data.columns[data.dtypes == 'int64']

for col in integers:
    col_z = col + '-z'
    data[col_z] = (data[col] - data[col].mean())/data[col].std(ddof=0) 

data.drop(['age','balance','day','duration','campaign','pdays','previous'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


#Checking the dtypes after obtaining z-score
data.dtypes


# ### Changing the categorical variables to numerical representation
# 

# In[ ]:


cleanup_nums = {
               "education":     {"primary": 1, "secondary": 2,"tertiary":3,"unknown":-1},
               "housing":     {"yes": 1, "no": 0},
               "loan":        {"yes": 1, "no": 0},
               "default":        {"yes": 1, "no": 0},
               "marital":     {"single": 1, "married": 2,"divorced":3},
               "poutcome":     {"success": 3, "other": 2,"unknown":-1,"failure":0},
               "contact":{"cellular": 1, "telephone": 2, "unknown": -1},
               "Target":{"1":1,"0":0}
                
                }
                
data.replace(cleanup_nums, inplace=True)

for categories in data.columns[data.columns=='object']:
    data[categories]=data[categories].astype("int32")

data.dtypes


# In[ ]:


data.head()


# ### Removing all columns with z-score greater and lesser than 3 and -3 respectivley as the values are outliers
# 

# In[ ]:


floats = data.columns[data.dtypes == 'float64']

for x in floats:
    indexNames_larger = data[ data[x]>3].index
    indexNames_lesser = data[ data[x]<-3].index
    # Delete these row indexes from dataFrame
    data.drop(indexNames_larger , inplace=True)
    data.drop(indexNames_lesser , inplace=True)
data.shape
data.head()


# ### One Hot encoding is performed on Month and Job columns

# In[ ]:


categoricals=['month','job']

for cols in categoricals:
    data=pd.concat([data,pd.get_dummies(data[cols],prefix=cols)],axis=1)
    data.drop(cols,axis=1,inplace=True)


# In[ ]:


data['Target']=data['Target'].astype('category')

data.dtypes


# ## Preparing the independent and target variables as X and Y
# ### Dropping duration column to get a realistic model
# 

# In[ ]:


import imblearn
X=data.drop(['Target','duration-z'],axis=1)
Target_Variable=data['Target']
Y=Target_Variable
X.head()


# In[ ]:


Y=Y.astype("int32")


# In[ ]:


Y.head()
Y.value_counts()


# #### After removing outliers we have 36155 records who have not purchased term deposit and 4054 records which have purchased term deposits

# ### Split the data into training and test set in the ratio of 70:30 (Training:Test) based on dependent and independent variables.

# In[ ]:


#Importing necessary libraries
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=22)
print(Ytrain.value_counts())
print(Ytest.value_counts())


# ## Strategy 2: Oversampling the training data to balance the Target column.

# In[ ]:


from imblearn.over_sampling import SMOTE


# In[ ]:


print("Before OverSampling, counts of label '1': {}".format(sum(Ytrain==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Ytrain==0)))

sm = SMOTE(random_state=2)
Xtrain_res, Ytrain_res = sm.fit_sample(Xtrain, Ytrain)

print('After OverSampling, the shape of train_X: {}'.format(Xtrain_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(Ytrain_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(Ytrain_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(Ytrain_res==0)))


# ## Base Model- Logistic Regression

# In[ ]:


log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]
log = pd.DataFrame(columns=log_cols)


# In[ ]:


#importing necessary libraries
from sklearn.linear_model import LogisticRegression


# In[ ]:


model_log_regression=LogisticRegression(solver="liblinear")


# In[ ]:


model_log_regression.fit(Xtrain_res,Ytrain_res)
coef_df = pd.DataFrame(model_log_regression.coef_)
coef_df['intercept'] = model_log_regression.intercept_
print(coef_df)


# In[ ]:


#Checking the score for logistic regression
logistic_regression_Trainscore=model_log_regression.score(Xtrain_res,Ytrain_res)
print("The score for Logistic regression-Training Data is {0:.2f}%".format(logistic_regression_Trainscore*100))
logistic_regression_Testscore=model_log_regression.score(Xtest,Ytest)
print("The score for Logistic regression-Test Data is {0:.2f}%".format(logistic_regression_Testscore*100))


# In[ ]:


#Predicting the Y values
Ypred=model_log_regression.predict(Xtest)

#Misclassification error
LR_MSE=1-logistic_regression_Testscore
print("Misclassification error of Logistical Regression model is {0:.1f}%".format(LR_MSE*100))


# In[ ]:


#Confusion Matrix
from sklearn import metrics
cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)
print(metrics.classification_report(Ytest, Ypred, digits=3))


# # The confusion matrix
# 
# True Positives (TP): we correctly predicted that they have taken Term Deposit is 620
# 
# True Negatives (TN): we correctly predicted that they have not taken Term Deposit is 9500
# 
# False Positives (FP): we incorrectly predicted that have taken Term Deposit (a "Type I error") 1300 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they have not taken Term Deposit  (a "Type II error") 640 Falsely predict negative Type II error

# In[ ]:


accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred)
recall_score=metrics.recall_score(Ytest,Ypred)
f1_score=metrics.f1_score(Ytest,Ypred)
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The F1 score of this model is {0:.2f}%".format(f1_score*100))


# In[ ]:


#AUC ROC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(Ytest, model_log_regression.predict(Xtest))
fpr, tpr, thresholds = roc_curve(Ytest, model_log_regression.predict_proba(Xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:


auc_score = metrics.roc_auc_score(Ytest, model_log_regression.predict_proba(Xtest)[:,1])
print("The AUC score is {0:.2f}".format(auc_score))


# ### Logistic Regression Results:
# #### The Accuracy of this model is 83.87%
# #### The Percission of this model is 32.45%
# #### The Recall score of this model is 49.21%
# #### The F1 score of this model is 39.11%
# #### The AUC Score of this model is 75%

# In[ ]:


log_entry = pd.DataFrame([["Logistic Regression",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)
log = log.append(log_entry)
log


# ## Base Model- Decision Tree Classifier

# In[ ]:


#Importing necessary libraries
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#Going with Decision Tree classifier with gini criteria, max_depth is kept at 10 to avoid overfitting of data
dtc=DecisionTreeClassifier(criterion='gini',random_state = 22,max_depth=10, min_samples_leaf=3,max_leaf_nodes=None)


# In[ ]:


#Fitting the data
dtc.fit(Xtrain_res,Ytrain_res)


# In[ ]:


#Predicting the data
Ypred=dtc.predict(Xtest)


# In[ ]:


from sklearn import metrics


# In[ ]:


#Checking the score for Decision Tree Classifier
Decision_Tree_Trainscore=dtc.score(Xtrain_res,Ytrain_res)
print("The score for Decision Tree-Training Data is {0:.2f}%".format(Decision_Tree_Trainscore*100))
Decision_Tree_Testscore=dtc.score(Xtest,Ytest)
print("The score for Decision Tree-Test Data is {0:.2f}%".format(Decision_Tree_Testscore*100))


# In[ ]:


#Misclassification error
DTC_MSE=1-Decision_Tree_Testscore
print("Misclassification error of Decision Tree Classification model is {0:.1f}%".format(DTC_MSE*100))


# In[ ]:


accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred)
recall_score=metrics.recall_score(Ytest,Ypred)
f1_score=metrics.f1_score(Ytest,Ypred)
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The F1 score of this model is {0:.2f}%".format(f1_score*100))


# In[ ]:


#Confusion Matrix
cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, cmap="YlGnBu")
print(metrics.classification_report(Ytest, Ypred, digits=3))


# ##### The confusion matrix
# 
# True Positives (TP): we correctly predicted that they have taken Term Deposit is 570
# 
# True Negatives (TN): we correctly predicted that they have not taken Term Deposit is 9800
# 
# False Positives (FP): we incorrectly predicted that have taken Term Deposit (a "Type I error") 1000 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they have not taken Term Deposit  (a "Type II error") 700 Falsely predict negative Type II error

# In[ ]:


#AUC ROC curve

dtc_auc = roc_auc_score(Ytest, dtc.predict(Xtest))
fpr, tpr, thresholds = roc_curve(Ytest, dtc.predict_proba(Xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree Classifier (area = %0.2f)' % dtc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('dtc_ROC')
plt.show()


# In[ ]:


## Calculating feature importance
feat_importance = dtc.tree_.compute_feature_importances(normalize=False)

feat_imp_dict = dict(zip(X.columns, dtc.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.sort_values(by=0, ascending=False)


# #### From the feature importance dataframe we can infer that campaign,housing are the variables that impact term depositors

# In[ ]:


auc_score = metrics.roc_auc_score(Ytest, dtc.predict_proba(Xtest)[:,1])
print("The AUC score is {0:.2f}".format(auc_score))


# ### Decision Tree Results:
# #### The Accuracy of this model is 85.92%
# #### The Percission of this model is 36.35%
# #### The Recall score of this model is 44.88%
# #### The F1 score of this model is 40.17%
# #### The AUC score of this model is 72%

# In[ ]:


log_entry = pd.DataFrame([["Decision Tree Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)
log = log.append(log_entry)
log


# ## Ensemble Technique- Random Forest Classifier

# In[ ]:


# Importing libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

# Random Forest Classifier with gini critireon and max_depth of 150 to increase overfitting
kfold = model_selection.KFold(n_splits=10, random_state=22,shuffle=True)
rf = RandomForestClassifier(n_estimators = 100,criterion = 'gini', max_depth = 150, min_samples_leaf=1,class_weight='balanced')
rf = rf.fit(Xtrain_res, Ytrain_res)
results = model_selection.cross_val_score(rf, Xtrain_res, Ytrain_res, cv=kfold)
print(results)
Ypred = rf.predict(Xtest)


# In[ ]:


Random_Forest_Trainscore=rf.score(Xtrain_res,Ytrain_res)
print("The score for Random Forest-Training Data is {0:.2f}%".format(Random_Forest_Trainscore*100))
Random_Forest_Testscore=rf.score(Xtest,Ytest)
print("The score for Random Forest-Test Data is {0:.2f}%".format(Random_Forest_Testscore*100))


# In[ ]:


#Misclassification error
RF_MSE=1-Random_Forest_Testscore
print("Misclassification error of Random Forest Classification model is {0:.1f}%".format(RF_MSE*100))


# In[ ]:


accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred)
recall_score=metrics.recall_score(Ytest,Ypred)
f1_score=metrics.f1_score(Ytest,Ypred)
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The F1 score of this model is {0:.2f}%".format(f1_score*100))
print(metrics.classification_report(Ytest,Ypred))


# In[ ]:


#Confusion Matrix
cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, cmap="BuPu")


# # The confusion matrix
# 
# True Positives (TP): we correctly predicted that they have taken Term Deposit is 490
# 
# True Negatives (TN): we correctly predicted that they have not taken Term Deposit is 10000
# 
# False Positives (FP): we incorrectly predicted that have taken Term Deposit (a "Type I error") 550 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they have not taken Term Deposit  (a "Type II error") 780 Falsely predict negative Type II error

# In[ ]:


#AUC ROC curve

rf_auc = roc_auc_score(Ytest, rf.predict(Xtest))
fpr, tpr, thresholds = roc_curve(Ytest, rf.predict_proba(Xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % rf_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('rf_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Ytest, rf.predict_proba(Xtest)[:,1])
print("The AUC score is {0:.2f}".format(auc_score))


# In[ ]:


log_entry = pd.DataFrame([["Random Forest Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)
log = log.append(log_entry)
log


# ## Ensemble Technique- Bagging Classifier

# In[ ]:


# Importing libraries
from sklearn.ensemble import BaggingClassifier

bg = BaggingClassifier(n_estimators=100, max_samples= .9, bootstrap=True, oob_score=True, random_state=22)
bg = bg.fit(Xtrain_res, Ytrain_res)
Ypred = bg.predict(Xtest)


# In[ ]:


Bagging_Trainscore=bg.score(Xtrain_res, Ytrain_res)
print("The score for Bagging-Training Data is {0:.2f}%".format(Bagging_Trainscore*100))
Bagging_Testscore=bg.score(Xtest,Ytest)
print("The score for Bagging-Test Data is {0:.2f}%".format(Bagging_Testscore*100))


# In[ ]:


#Misclassification error
BG_MSE=1-Bagging_Testscore
print("Misclassification error of Bagging Classification model is {0:.1f}%".format(BG_MSE*100))


# In[ ]:


accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred)
recall_score=metrics.recall_score(Ytest,Ypred)
f1_score=metrics.f1_score(Ytest,Ypred)
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The F1 score of this model is {0:.2f}%".format(f1_score*100))
print(metrics.classification_report(Ytest,Ypred))


# In[ ]:


#Confusion Matrix
cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, cmap="Greens")


# 
# 
# 
# # The confusion matrix
# 
# True Positives (TP): we correctly predicted that they have taken Term Deposit is 450
# 
# True Negatives (TN): we correctly predicted that they have not taken Term Deposit is 10000
# 
# False Positives (FP): we incorrectly predicted that have taken Term Deposit (a "Type I error") 550 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they have not taken Term Deposit  (a "Type II error") 820 Falsely predict negative Type II error

# In[ ]:


#AUC ROC curve

bg_auc = roc_auc_score(Ytest, bg.predict(Xtest))
fpr, tpr, thresholds = roc_curve(Ytest, bg.predict_proba(Xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Bagging Classifier (area = %0.2f)' % bg_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('bg_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Ytest, bg.predict_proba(Xtest)[:,1])
print("The AUC score is {0:.2f}".format(auc_score))


# ### Bagging Classifier Results:
# 
# #### The Accuracy of this model is 88.62%
# #### The Percission of this model is 44.88%
# #### The Recall score of this model is 35.51%
# #### The F1 score of this model is 39.65%
# #### The AUC score of this model is 76%

# In[ ]:


log_entry = pd.DataFrame([["Bagging Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)
log = log.append(log_entry)
log


# ## Ensemble Technique- AdaBoost Classifier

# In[ ]:


#Importing necessary libraries
from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier(n_estimators= 100, learning_rate=0.5, random_state=22)
ab = ab.fit(Xtrain_res, Ytrain_res)


# In[ ]:


Ypred=ab.predict(Xtest)


# In[ ]:


Adaboosting_Trainscore=ab.score(Xtrain_res,Ytrain_res)
print("The score for Adaboosting-Training Data is {0:.2f}%".format(Adaboosting_Trainscore*100))
Adaboosting_Testscore=ab.score(Xtest,Ytest)
print("The score for Adaboosting-Test Data is {0:.2f}%".format(Adaboosting_Testscore*100))


# In[ ]:


#Misclassification error
AB_MSE=1-Adaboosting_Testscore
print("Misclassification error of Bagging Classification model is {0:.1f}%".format(AB_MSE*100))


# In[ ]:


accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred)
recall_score=metrics.recall_score(Ytest,Ypred)
f1_score=metrics.f1_score(Ytest,Ypred)
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The F1 score of this model is {0:.2f}%".format(f1_score*100))
print(metrics.classification_report(Ytest,Ypred))


# In[ ]:


#Confusion Matrix
cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, cmap="Reds")


# ###### The confusion matrix
# 
# True Positives (TP): we correctly predicted that they have taken Term Deposit is 680
# 
# True Negatives (TN): we correctly predicted that they have not taken Term Deposit is 9500
# 
# False Positives (FP): we incorrectly predicted that have taken Term Deposit (a "Type I error") 1300 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they have not taken Term Deposit  (a "Type II error") 590 Falsely predict negative Type II error

# In[ ]:


#AUC ROC curve

ab_auc = roc_auc_score(Ytest, ab.predict(Xtest))
fpr, tpr, thresholds = roc_curve(Ytest, ab.predict_proba(Xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Bagging Classifier (area = %0.2f)' % ab_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('ab_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Ytest, ab.predict_proba(Xtest)[:,1])
print("The AUC score is {0:.2f}".format(auc_score))


# ### Adaboosting Classifier Results:
# 
# #### The Accuracy of this model is 84.57%
# #### The Percission of this model is 34.85%
# #### The Recall score of this model is 53.54%
# #### The F1 score of this model is 42.22%
# #### The AUC score is 77%

# In[ ]:


log_entry = pd.DataFrame([["Adaptive Boosting Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)
log = log.append(log_entry)
log


# ## Ensemble Technique- Gradient Boosting

# ### Hyper Parameterization- Tuning the model with RandomSearch CV

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
num_estimators = [100,200]
learn_rates = [0.2,0.3]

scoreFunction = {"recall": "recall", "precision": "precision"}

param_grid = {'n_estimators': num_estimators,
              'learning_rate': learn_rates,
}

random_search =RandomizedSearchCV(GradientBoostingClassifier(loss='deviance'), param_grid, scoring = scoreFunction,               
                                       refit = "precision", random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

random_search.fit(Xtrain_res, Ytrain_res)


# In[ ]:


random_search.best_params_


# In[ ]:


# Importing necessary libraries and fitting the data

from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.3, random_state=22)
gb = gb.fit(Xtrain_res, Ytrain_res)
Ypred = gb.predict(Xtest)


# In[ ]:


Gradient_Booosting_Trainscore=gb.score(Xtrain_res,Ytrain_res)
print("The score for Gradient_Booosting-Training Data is {0:.2f}%".format(Gradient_Booosting_Trainscore*100))
Gradient_Booosting_Testscore=gb.score(Xtest,Ytest)
print("The score for Gradient_Booostinge-Test Data is {0:.2f}%".format(Gradient_Booosting_Testscore*100))


# In[ ]:


#Misclassification error
GB_MSE=1-Gradient_Booosting_Testscore
print("Misclassification error of Gradient Boosting Classification model is {0:.1f}%".format(GB_MSE*100))


# In[ ]:


accuracy_score=metrics.accuracy_score(Ytest,Ypred)
percision_score=metrics.precision_score(Ytest,Ypred)
recall_score=metrics.recall_score(Ytest,Ypred)
f1_score=metrics.f1_score(Ytest,Ypred)
print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))
print("The Percission of this model is {0:.2f}%".format(percision_score*100))
print("The Recall score of this model is {0:.2f}%".format(recall_score*100))
print("The F1 score of this model is {0:.2f}%".format(f1_score*100))
print(metrics.classification_report(Ytest,Ypred))


# In[ ]:


#Confusion Matrix
cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, cmap="Blues")


# ##### The confusion matrix
# 
# True Positives (TP): we correctly predicted that they have taken Term Deposit is 500
# 
# True Negatives (TN): we correctly predicted that they have not taken Term Deposit is 10000
# 
# False Positives (FP): we incorrectly predicted that have taken Term Deposit (a "Type I error") 480 Falsely predict positive Type I error
# 
# False Negatives (FN): we incorrectly predicted that they have not taken Term Deposit  (a "Type II error") 770 Falsely predict negative Type II error

# In[ ]:


#AUC ROC curve

gb_auc = roc_auc_score(Ytest, gb.predict(Xtest))
fpr, tpr, thresholds = roc_curve(Ytest, gb.predict_proba(Xtest)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Gradient Boosting Classifier (area = %0.2f)' % gb_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('gb_ROC')
plt.show()
auc_score = metrics.roc_auc_score(Ytest, gb.predict_proba(Xtest)[:,1])
print("The AUC score is {0:.2f}".format(auc_score))


# In[ ]:


log_entry = pd.DataFrame([["Gradient Boosting Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)
log = log.append(log_entry)
log


# ## Final Insights
# #### 1. The aim of the data-set is to predict customers who would subscribe for a term deposit
# #### 2. Data had 16 independent variable and 1 target variable
# #### 3. Outliers were handled by converting numerical variables to z-score and removing rows greater than +/- 3
# #### 4. Label encoding and one-hot encoding was employed for Categorical columns
# #### 5. Since the target variable was highly imbalanced SMOTE oversampling on the training data was employed(accuracy was better without oversampling but f1 and AUC score were low)
# #### 6. Out of all models tried on, Gradient Boosing is considered the best as it has good Accuracy, f1 and AUC scores
# #### 7. Hyper-parameterization tuning was done on Gradient Boosting to find the best parameters('n_estimators': 200, 'learning_rate': 0.3). RamdomSearchCV was employed.
# #### 8. All Ensemble Techniques were better than the base models(Standard Classification Algorithm) Logistic Regression and Decision Tree Classifier

# In[ ]:




