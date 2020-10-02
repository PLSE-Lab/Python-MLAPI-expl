#!/usr/bin/env python
# coding: utf-8

# Use classification technique to create a model to predict loan default or payoff and maximize expected value for Lending Club

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# In[ ]:


data = pd.read_csv('../input/loan.csv', low_memory=False)


# Look at 74 columns with 2 sample rows to get a glimpse of the data.<br>
# loan_status will serve as our target<br>
# <font color='green'>Possible applicant features listed below:</font> <br>
# emp_length, home_ownership, annual_inc<br>
# open_acc = number of open credit lines in the borrower's credit file<br>
# revol_bal = portion of CC spending thats unpaid at end of billing cycle<br>
# revol_util = amount of credit the borrower is using relative to all available revolving credit.<br>
# dti=debt to income & inq_last_6mths= credit inquiries<br>
# home_ownership

# What important loan metric is missing?  <font color='red'>FICO Score</font> 

# Drop some outliers

# In[ ]:


data.drop(484446, inplace = True)
data.drop(531886, inplace = True)
data.drop(475046, inplace = True)
data.drop(532701, inplace = True)
data.drop(540456, inplace = True)


# Debt-to-Income vs. Annual Income relationship makes sense

# In[ ]:


sns.set(font_scale=1.5)
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
sns.regplot(x='dti', y='annual_inc', data= data, line_kws={'color':'red'}, ax=ax)


# There is a strong corellation between the grade given and the interest rate given, but since this is assigned after the bank has performed their own prediction of risk, we can't use them as pre-application features.

# In[ ]:


data.boxplot(column='int_rate', by='grade', figsize=(12,6))


# Clean up home ownership into two categories as it is a relevant applicant feature

# In[ ]:


data.home_ownership.value_counts()


# In[ ]:


data=data.drop(data[data.home_ownership=='OTHER'].index)
data=data.drop(data[data.home_ownership=='ANY'].index)
data=data.drop(data[data.home_ownership=='NONE'].index)
data.home_ownership.replace('OWN','MORTGAGE', inplace=True)
data.home_ownership.value_counts()


# In[ ]:


data.loan_status.value_counts().plot(kind='barh', figsize=(7,5), title = "Loan Status", fontsize = 15)


# Drop all rows for loan_status types except 'Fully Paid' and 'Charged Off' as they are the only categories that have matured.

# In[ ]:


matureLoan = data[(data.loan_status=='Fully Paid') | (data.loan_status=='Charged Off')].copy()


# In[ ]:


matureLoan.loan_status.value_counts()


# Look at all potential features that are relevant to the applicant

# In[ ]:


possibleFeatures=matureLoan[['emp_length', 'home_ownership', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                                'mths_since_last_delinq','mths_since_last_record', 'open_acc', 'pub_rec',
                                'revol_bal', 'revol_util', 'total_acc', 'collections_12_mths_ex_med',
                                'mths_since_last_delinq', 'open_acc_6m', 'open_il_6m','open_il_12m',
                                'open_il_24m', 'mths_since_rcnt_il', 'il_util','open_rv_12m', 'open_rv_24m','max_bal_bc',
                                'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m','tot_coll_amt','tot_cur_bal','loan_status']]


# In[ ]:


possibleFeatures.isnull().sum()


# Drop all columns that have large NaN's<br>
# Use a dictionary to digitize emp_length column

# In[ ]:


matureLoan.emp_length.replace({'10+ years':10, '< 1 year':1, '1 year':1, '3 years':3, '8 years':8, '9 years':9, '4 years':4, '5 years':5, '6 years':6, '2 years':2, '7 years':7}, inplace=True)


# In[ ]:


matureLoan.emp_length.value_counts(dropna=False)


# In[ ]:


features = pd.get_dummies(matureLoan[['emp_length', 'home_ownership', 'annual_inc', 'dti', 'delinq_2yrs', 
                                       'inq_last_6mths', 'open_acc', 'pub_rec','revol_bal', 'revol_util', 'total_acc',
                                       'collections_12_mths_ex_med','tot_coll_amt','tot_cur_bal','loan_status']],
                                    drop_first = True)


# In[ ]:


features.isnull().sum()


# In[ ]:


features.dropna(inplace=True)


# Here is the features list after clean up & applying get_dummies

# In[ ]:


features.isnull().sum()


# In[ ]:


features.describe()


# Perform logistic regression using loan_status as my target

# In[ ]:


X=features.drop('loan_status_Fully Paid', axis=1)


# In[ ]:


y=features['loan_status_Fully Paid']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Split my training & test data using 70/30 split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Scale the dataset

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# Output the coefficients to help prioritize which features are important

# In[ ]:


name = features.columns

coef = logreg.coef_[0]

pd.DataFrame([name,coef],index = ['Name','Coef']).transpose()


# Keep the high coefficient values for our model and drop the rest

# In[ ]:


features1 = pd.get_dummies(matureLoan[['annual_inc', 'dti','inq_last_6mths', 'revol_util', 'total_acc','tot_cur_bal','loan_status']],
                                    drop_first = True)


# In[ ]:


features1.dropna(inplace=True)
features1.isnull().sum()


# Re-run logistic regression using the new features list

# In[ ]:


X1=features1.drop('loan_status_Fully Paid', axis=1)
y1=features1['loan_status_Fully Paid']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test = scaler.transform(X1_test)


# In[ ]:


logreg1= LogisticRegression()
logreg1.fit(X1_train,y1_train)


# In[ ]:


name = features1.columns

coef = logreg1.coef_[0]

pd.DataFrame([name,coef],index = ['Name','Coef']).transpose()


# Use the model to predict on x1_test

# In[ ]:


y_pred1 = logreg1.predict(X1_test)


# Calculate the accuracy of the model

# In[ ]:


metrics.accuracy_score(y1_test,y_pred1)


# Compare to the Null Accuracy<br>
# Null Accuracy = the proportion of the majority class in the testing set (aka, baseline)

# In[ ]:


y1_test.mean()


# If a dummy model were to predict the predominant class 100% of the time, it would be 81% correct, no difference from my model

# **Lets use a different classification technique just to confirm that we chose the right features**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

treeclf = DecisionTreeClassifier(max_depth=4, random_state=42)
treeclf.fit(X, y)


# In[ ]:


pd.DataFrame({'feature':X.columns, 'importance':treeclf.feature_importances_})


# Decision Tree method confirms we have the right features in our model

# Let's take a look at our logistic regression confusion matrix

# In[ ]:


cm = metrics.confusion_matrix(y1_test,y_pred1)
plt.clf()
plt.rcParams["figure.figsize"] = [6,6]
plt.imshow(cm, cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Loan Status Fully Paid')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# Looks like we have class imbalance, plot histogram

# In[ ]:


y_pred_prob = logreg1.predict_proba(X1_test)[:, 1]


# In[ ]:


plt.rcParams['font.size'] = 14
plt.rcParams["figure.figsize"] = [7,7]
plt.hist(y_pred_prob)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')


# In[ ]:


plt.rcParams["figure.figsize"] = [7,7]
plt.hist(y_pred_prob, label='prediction')
plt.hist(y1_test, label='test')
plt.xlim(0, 1)
plt.title('Histogram of test data vs. prediction')
plt.xlabel('Actual data vs. predicted probability')
plt.ylabel('Frequency')
plt.legend()


# Compared to our test data, our model only predicts 1.0 (Loan fully paid) and almost never predicts 0.0 (loan defaul).

# Since we have class imbalance, we have to change the model's threshold

# Threshold = 0.6

# In[ ]:


y_pred_class6 = np.where(y_pred_prob > 0.6, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class6)


# Threshold= 0.7

# In[ ]:


y_pred_class7 = np.where(y_pred_prob > 0.7, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class7)


# Threshold = 0.8

# In[ ]:


y_pred_class8 = np.where(y_pred_prob > 0.8, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class8)


# Threshold = 0.9

# In[ ]:


y_pred_class9 = np.where(y_pred_prob > 0.9, 1, 0)
metrics.confusion_matrix(y1_test,y_pred_class9)


# **How do we apply the different confusion matrices to the real world?**<br>
# We need to look at the business use case
# 
# Apply the Cost-Benefit Matrix

# Assume the Cost of a charge-off (average)= Loan amount - Total payment received

# In[ ]:


matureLoan['costChargeOff'] = matureLoan.loan_amnt - matureLoan.total_pymnt


# In[ ]:


cost=matureLoan.costChargeOff[matureLoan.loan_status=='Charged Off']
cost.mean()


# Assume the Benefit of a fully paid loan (average) = Total interest received

# In[ ]:


benefit = matureLoan.total_rec_int[matureLoan.loan_status=='Fully Paid']
benefit.mean()


# When looking at a business use case for Lending Club loans:<br><br>
# 
# Benefit of a True Positive = 1902 dollars<br>
# Benefit of a True Negative = 0 since they don't qualify for the loan<br>
# Cost of a False Positive = 8188 dollars<br>
# Cost of a False Negative = 0 since they don't qualify for the loan<br>
# 
# Cost Benefit Matrix=
# 
#    |0   8188|<br>
#    |0   1902|

# In[ ]:


# Benefit of a True Positive = $1902
BTP = 1902
# Benefit of a True Negative = $0 since they don't qualify for the loan
BTN = 0
# Cost of a False Positive = $8188
CFP = -8188
# Cost of a False Negative = $0 since they don't qualify for the loan
CFN = 0

# Calculate the probabilities for each confusion matrix entry
TP = 46168/56718
TN = 3/56718
FP = 10537/56718
FN = 10/56718

TP, TN, FP, FN


# Multiply Cost-Benefit Matrix & Confusion Matrix<br>
# 
# Cost Benefit Matrix * Confusion Matrix probabilities<br>
# 
# |0     8188|             |pTN    pFP|<br>
# |0     1902|             |pFN    pTP|<br><br>
# Expected Value = BTP * pTP + BTN * pTN + CFP * pFP + CFN * pFN

# In[ ]:


EV = BTP * TP + BTN * TN + CFP * FP + CFN * FN
EV


# If we use our 0.5 threshold model to predict which loans to give out, we'll make $27/customer

# Let's do the same calculation for the other confusion matrices if we change the thresholds

# Threshold = 0.5, Expected Value = 27 dollars<br>
# Threshold = 0.6, Expected Value = 37 dollars<br>
# Threshold = 0.7, Expected Value = 127 dollars<br>
# Threshold = 0.75, Expected Value = 229 dollars<br>
# Threshold = 0.8, Expected Value = 308 dollars<br>
# Threshold = 0.9, Expected Value = 113 dollars<br>

# In[ ]:


thresholds=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
MSE=[27, 37, 127, 229, 308, 113]

plt.plot(thresholds, MSE)
plt.xlabel('Threshold')
plt.ylabel('Expected Value ($)')


# **Conclusion:  Using a threshold of 0.8 allows our model to balance the loan qualifying/denying decision while maximizing expected value.**

# In[ ]:




