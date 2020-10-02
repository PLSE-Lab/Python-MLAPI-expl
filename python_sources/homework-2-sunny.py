#!/usr/bin/env python
# coding: utf-8

# # Homework Assignment 2

# ## Loading the data

# In[ ]:


#Loading the data
import pandas as pd


# In[ ]:


train = pd.read_csv("../input/train-dataset/train.csv", low_memory = False) #imports data train.csv
train.head()
train = train.drop(["Unnamed: 0"],1)


# In[ ]:


test = pd.read_csv("../input/test-dataset/test.csv", low_memory = False) #Loading the test data
test = test.drop(['Unnamed: 0.1'],1)
test = test.drop(["Unnamed: 0"],1)
test.head()


# ### Q1 What percentage of your training set loans are in default?

# In[ ]:


percent= train.default.sum() / train.default.count() #Sum method counts the values and since False has a default value of 0, 
print("Percentage of training set loans in default {:.0%}".format(percent))


# ### Q2 Which ZIP code has the highest default rate in the training dataset?

# In[ ]:


train.groupby("ZIP").default.value_counts() #using the groupby for counting the value counts


# ### Q3 What is the default rate in the training set for the first year for which you have data?

# In[ ]:


train_year0 = train.loc[train["year"] == 0]
percent = train_year0.default.sum() / train_year0.default.count()
print("Percentage of default rate in training set for year {:.0%}".format(percent))


# ###  Q4 What is the correlation between age and income in the training dataset?

# In[ ]:


train_corr_matrix = train.corr() #stores all the values of the training correlation matrix


# In[ ]:


#train_corr_matrix["age"].sort_values(ascending= False)
train_corr_matrix


# #### We can use a different function as well

# In[ ]:


print("correlation between age and income in the training dataset is {:.0}".format(train.age.corr(train.income)))
train.columns


# ### Q5 What is the in-sample accuracy? That is, find the accuracy score of the fitted model for predicting the outcomes using the whole training dataset

# In[ ]:


y_train = train['default'] #creates dataset that includes only defaulted loans from train 
x_train = pd.concat([train.drop( ["default", 'occupation', 'ZIP', 'sex', 'minority'], axis=1), 
                     pd.get_dummies (train['occupation']),pd.get_dummies (train['ZIP'])], axis=1)
x_train.columns


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state=42,oob_score=True)
rfc.fit(x_train,y_train)
y_predict = rfc.predict(x_train)
train.head()


# In[ ]:


from sklearn import metrics
score = metrics.accuracy_score(y_train,y_predict)
print("Insample Score is {:.0%}".format(score))


# ## Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(rfc,x_train,y_train, cv = 3, scoring = "accuracy")
cv_score


# In[ ]:


from sklearn.model_selection import cross_val_predict
y_predict = cross_val_predict(rfc,x_train, y_train, cv = 3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train,y_predict)


# In[ ]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score


# In[ ]:


precision_score(y_train, y_predict)


# In[ ]:


recall_score(y_train,y_predict)


# In[ ]:


f1_score(y_train,y_predict)


# ### Exploratory Data Analysis

# In[ ]:


train_target = train.drop(["default"],1)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
train_target.hist(bins=10,figsize=(20,20))
plt.show()


# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt

plt.figure(figsize=(15,8))
sns.heatmap(train.corr(),annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm',fmt=".2%")


# ### Q6. What is the out of bag score for the model? The out of bag score is a fit and validation method that is calculated while the model is being trained.

# In[ ]:


print("Out of Bag Score:")
oob = rfc.oob_score_
print(oob)


# ### Q7 What is the out of sample accuracy? That is, find the accuracy score of the model using the test data without re-estimating the model parameters. 

# In[ ]:


y_test = test["default"] #creates dataset that includes only defaulted loans
x_test = pd.concat([test.drop( ["default", "occupation", "ZIP", "sex", "minority"], axis=1),
                     pd.get_dummies (test["occupation"]),pd.get_dummies (test["ZIP"])], axis=1)


# In[ ]:


y_predict_test = rfc.predict(x_test)
test["default_predict"] = y_predict_test
from sklearn import metrics
score_test = metrics.accuracy_score(y_test,y_predict_test)
print("Outofsample Score is {:.0%}".format(score_test))


# ### Q8 What is the predicted average default probability for all non-minority members in the test set? 

# In[ ]:


majority_default = (test[test.minority == 0].default_predict.sum()/len(test))*100
print("Predicted average default probability for non-minority members {:.2f}".format(majority_default))


# In[ ]:


test.groupby("minority").default_predict.value_counts()


# ### Q9 What is the predicted average default probability for all non-minority members in the test set? 

# In[ ]:


majority_default = (test[test.minority == 1].default_predict.sum()/len(test))*100
print("Predicted average default probability for non-minority members {:.2f}".format(majority_default))


# ### Q10 Is the loan granting scheme (the cutoff, not the model) group unaware? (hint: this question does not require a calculation)

# The Cutoff model for the default is based on probability which is same for the group
# 

# ### Q11 Has the loan granting scheme achieved demographic parity? Compare the share of approved female applicants to the share of approved male applicants. Do the same for minority and non-minority applicants. What does this indicate about the demographic parity achieved by the model?

# In[ ]:


test["loan_approved"] = ~(test.default_predict)
test.head()


# In[ ]:


share_approved_male = test[test.sex == 0].loan_approved.sum()/(len(test[test.sex == 0]))
print("Share of approved male applicant", (share_approved_male)*100)


# In[ ]:


share_approved_female = test[test.sex == 1].loan_approved.sum()/(len(test[test.sex == 1]))
print("Share of approved female applicant is" ,(share_approved_female)*100)


# In[ ]:


test.groupby("sex").loan_approved.value_counts()


# In[ ]:


test.groupby("minority").loan_approved.value_counts()


# In[ ]:


share_approved_minority = test[test.minority == 1].loan_approved.sum()/(len(test[test.minority == 1]))
share_approved_majority = test[test.minority == 0].loan_approved.sum()/(len(test[test.minority == 0]))
print("Share of approved majority applicant is {maj} and share of approved minority applicant is {min}".format(maj= share_approved_majority*100, min = share_approved_minority*100) )


# The model doesn't achieve Demographic partiy for males and neither it for minority applicants

# ### Q11 Is the loan granting scheme equal opportunity? Compare the share of successful non-minority applicants that would repay to the share of successful minority applicants that would repay. Do the same comparison of the share of successful female applicants that would repay versus successful male applicants that would repay. What do these shares indicate about the likelihood of securing a loan in different population groups that would repay a loan?

# In[ ]:


test["pay_cap"] = ~(test.default)
from sklearn.metrics import confusion_matrix #determmining the true positive rate for quantifying equal opportunity
cm_min= confusion_matrix(test[test.minority==1].pay_cap,test[test.minority==1].loan_approved)
cm_maj= confusion_matrix(test[test.minority==0].pay_cap,test[test.minority==0].loan_approved)
cm_male = confusion_matrix(test[test.sex==0].pay_cap,test[test.sex==0].loan_approved)
cm_female = confusion_matrix(test[test.sex==1].pay_cap,test[test.sex==1].loan_approved)


# In[ ]:


eq_op_min = cm_min[1][1]*100/(cm_min[1][1]+cm_min[1][0])
eq_op_maj = cm_maj[1][1]*100/(cm_maj[1][1]+cm_min[1][0])
eq_op_male = cm_male[1][1]*100/(cm_male[1][1]+cm_male[1][0])
eq_op_female =cm_female[1][1]*100/(cm_female[1][1]+cm_female[1][0])
print("Share of successful non-minority applicants that would repay {:.2f}".format(eq_op_maj))
print("Share of successsful minority applicants that would repay {:.2f}".format(eq_op_min))
print("Share of successsful male applicants that would repay {:.2f}".format(eq_op_male))
print("Share of successsful female applicants that would repay {:.2f}".format(eq_op_female))

