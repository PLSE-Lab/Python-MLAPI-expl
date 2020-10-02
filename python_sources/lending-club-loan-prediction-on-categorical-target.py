#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict # 10-Fold Cross Validation
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier # Logistic Regression
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbour
from sklearn.ensemble import RandomForestClassifier, VotingClassifier # Ensemble  

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier  



import datetime as dt 
import sys, matplotlib, warnings, math, sklearn
warnings.filterwarnings("ignore")

# Libraries versions 
print("Numpy : " + np.__version__)
print("Pandas : " + pd.__version__)
print("Seaborn : " + sns.__version__) 
print("Matplotlib : " + matplotlib.__version__)
print("SkLearn : " + sklearn.__version__)
print("Python : " + sys.version)


# In[ ]:


loanstats = '../data/lending_club/loan.csv'

try:
    loan_df = pd.read_csv(loanstats, skipinitialspace=True, low_memory=False)
except Exception as e:
    print(e)


# In[ ]:


loanstats = '/kaggle/input/lending-club-loan-data/loan.csv'

loan_df = pd.read_csv(loanstats, skipinitialspace = True, low_memory = False)


# In[ ]:


df1 = loan_df.copy()  


# In[ ]:


loan_df = df1.copy()


# In[ ]:


# Getting the percentage of NA values 
pd.set_option('display.max.rows', len(loan_df.columns))
print(loan_df.shape)
print((loan_df.isnull().sum() / loan_df.shape[0] * 100))


# In[ ]:





# In[ ]:


def get_nan_cols(df, nan_percent=0.8):
    threshold = len(df.index) * nan_percent
    return [c for c in df.columns if sum(df[c].isnull()) >= threshold]

loan_df.drop(get_nan_cols(loan_df, 0.9), axis=1, inplace= True)


# In[ ]:


# Removing un-necessary columns after visualization of data 
cols = ["hardship_flag", "application_type", "policy_code", "out_prncp_inv", "out_prncp", "initial_list_status",
        "title", "pymnt_plan", "emp_title", "chargeoff_within_12_mths", 
        "collections_12_mths_ex_med"] # zip_code

loan_df.drop(cols, axis=1, inplace= True)


# In[ ]:


pd.set_option('display.max.rows', len(loan_df.columns))
print((loan_df.isnull().sum() / loan_df.shape[0] * 100))


# In[ ]:


# loan_df.describe()
pd.set_option('display.max.columns', len(loan_df.columns))
loan_df.head()


# ## Exploratory Data Analysis
# 
#     NOTE: This has been done after data visualization
# 
# There are four categories for 'loan_status' and the data for the two which are "Does not meet the credit policy" are very less in numbers. It can be renamed to two categories, i.e. "Fully Paid" and "Charged Off". This will improve the model accuracy. Creating separate column for "Does Not Meet Credit Policy".

# In[ ]:


def dnmcpStatus(x):
    if x in ['Fully Paid','Charged Off']:
        return 0 
    else :
        return 1

loan_df['dnmcp'] = loan_df.loan_status.apply(lambda x: dnmcpStatus(x))


# In[ ]:


try:
    # ["term"]
    loan_df['term'] = loan_df['term'].str.replace('months','')
    loan_df['term'] = loan_df['term'].str.replace('36','3')
    loan_df['term'] = loan_df['term'].str.replace('60','5')
    loan_df['term'] = pd.to_numeric(loan_df['term']) 
except Exception as e:
    print(e.with_traceback())


# In[ ]:


try:
    # ["int_rate"]
#     loan_df['int_rate'] = loan_df['int_rate'].str.replace('%','')
    loan_df['int_rate'] = pd.to_numeric(loan_df['int_rate']) 
except Exception as e:
    print(e.with_traceback())


# In[ ]:


try:
    # ["loan_status"]
    loan_df["loan_status"]=loan_df["loan_status"].str.replace("Does not meet the credit policy. Status:Fully Paid",
                                                                "Fully Paid")
    loan_df["loan_status"]=loan_df["loan_status"].str.replace("Does not meet the credit policy. Status:Charged Off",
                                                                "Charged Off")
except Exception as e:
    print(e.with_traceback())


# In[ ]:


try:
    # ["emp_length"]
    loan_df["emp_length"] = loan_df["emp_length"].str.replace('< 1 year','0')
    
    # ["home_ownership"]
    loan_df["home_ownership"] = loan_df["home_ownership"].replace(['ANY', 'NONE','OTHER'], 'RENT')
    
    
    # ['purpose']
    loan_df['purpose'] = loan_df['purpose'].str.replace('renewable_energy', 'other')
except Exception as e:
    print(e.with_traceback())


# In[ ]:


try:
    # ["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d"]
    loan_df['issue_d'] = pd.to_datetime(loan_df['issue_d'])
    loan_df['last_pymnt_d'] = pd.to_datetime(loan_df['last_pymnt_d'])
    loan_df['earliest_cr_line'] = pd.to_datetime(loan_df['earliest_cr_line'])
    loan_df['last_credit_pull_d'] = pd.to_datetime(loan_df['last_credit_pull_d'])
except Exception as e:
    print(e.with_traceback())


# In[ ]:


try:
    # ['issue_yr']
    loan_df['issue_yr'] = loan_df['issue_d'].dt.year
    
    # ['issue_month']
    loan_df['issue_month'] = loan_df['issue_d'].dt.month_name()    

    # ['issue_qtr']
    loan_df["issue_qtr"] = loan_df["issue_d"].dt.quarter
except Exception as e:
    print(e.with_traceback())


# In[ ]:


try:
    loan_df['last_credit_pull_d'].fillna(method='ffill', inplace=True)
    loan_df['earliest_cr_line'].fillna(method='ffill', inplace=True)
    loan_df['last_pymnt_d'].fillna(method='ffill', inplace=True)

    loan_df["revol_util"].fillna(loan_df["revol_util"].mean(), inplace=True)
    loan_df["open_acc"].fillna(int(loan_df['open_acc'].mean()), inplace=True)
    loan_df["annual_inc"].fillna(round(loan_df["annual_inc"].mean(),2), inplace=True)
    loan_df["inq_last_6mths"].fillna(math.ceil(loan_df["inq_last_6mths"].mean()), inplace=True)
    loan_df["total_acc"].fillna(math.floor(loan_df["total_acc"].mean()), inplace=True)
    loan_df["pub_rec_bankruptcies"].fillna(math.ceil(loan_df["pub_rec_bankruptcies"].mean()), inplace=True)
    
    loan_df['emp_length'].fillna('0', inplace=True)
    
    loan_df['tax_liens'].fillna(0,inplace=True)
    loan_df['pub_rec_bankruptcies'].fillna(0,inplace=True)
    loan_df['delinq_amnt'].fillna(0, inplace=True)
    loan_df['acc_now_delinq'].fillna(0, inplace=True)
    loan_df['delinq_2yrs'].fillna(0, inplace=True)
    loan_df['pub_rec'].fillna(0, inplace=True)
    loan_df['dti'].fillna(0, inplace=True)
    
except Exception as e:
    print(e)


# In[ ]:





# In[ ]:


# Separating 2008 data for testing 
test_2008 = loan_df[loan_df.issue_yr == 2008]
loan_df = loan_df[loan_df.issue_yr != 2008]


# ## Visualization & Outliers Removal
#     NOTE: Data preprocessing is done after the visualization of data on various columns

# In[ ]:


title_font, xlabel_font, ylabel_font = 20, 12, 12  
colors=['lightcoral','yellowgreen', 'gold', 'skyblue','red','cyan']


# ## Univariate Analysis

# In[ ]:


# ['verification_status']

verification_status = [(loan_df['verification_status']=='Verified').sum(),
                      (loan_df['verification_status']=='Source Verified').sum(),
                      (loan_df['verification_status']=='Not Verified').sum()]

plt.figure(figsize=(5,5),dpi=120)
plt.pie(verification_status, labels = ('Verified','Source Verified', 'Not Verified'), 
        explode = (0.05, 0.05, 0.05), colors = colors[0:3],shadow=True,startangle = 360, autopct='%1.2f%%')
plt.title("Loan Verification", fontsize = title_font)
plt.xlabel('Status',fontsize = xlabel_font)
plt.show()


# In[ ]:


# ['home_ownership']
ownership_status = [(loan_df['home_ownership'] == 'RENT').sum(),
                      (loan_df['home_ownership'] == 'OWN').sum(),
                      (loan_df['home_ownership'] == 'MORTGAGE').sum()]

plt.figure(figsize=(5,5),dpi=120)
plt.pie(ownership_status, labels = ('RENT','OWN', 'MORTGAGE'), 
        explode = (0.05, 0.05, 0.05), colors = colors[-5:-2],shadow=True,startangle = 180, autopct='%1.2f%%')
plt.title("Home Ownership", fontsize = title_font)
plt.xlabel('Status',fontsize = xlabel_font)
plt.show()


# In[ ]:


# ['loan_status'] -> DNMCP :- Does not meet the credit policy
loan_status = [(loan_df['loan_status'] == 'Fully Paid').sum(), (loan_df['loan_status'] == 'Charged Off').sum()]

plt.figure(figsize=(5,5),dpi=120) 
plt.pie(loan_status, labels = ('Fully Paid', 'Charged Off'), 
        explode = (0.08, 0.08), colors = colors[0:2],shadow=True,startangle = 180, autopct='%1.2f%%')
plt.title("Loan Status", fontsize = title_font)
plt.xlabel('Status',fontsize = xlabel_font)
plt.show()


# In[ ]:


# ['purpose']
plt.figure(figsize=(10,10),dpi=120)
sns.countplot(y='purpose', data=loan_df ,hue='loan_status')
plt.show()


# In[ ]:


# ['loan_status']
plt.figure(figsize = (10,4), dpi=120)
sns.countplot(x='issue_yr', data=loan_df, hue='loan_status')
plt.xlabel('Year')
plt.legend(loc='upper left')
plt.title("Loan Status by Years", fontsize=title_font)
plt.show()


# In[ ]:


plt.figure(figsize = (12,4), dpi=120)
sns.countplot(x='term', data=loan_df, hue='loan_status')
plt.xlabel('Years',fontsize=xlabel_font)
plt.ylabel('Count',fontsize=ylabel_font)
plt.title('Loan Term', fontsize=title_font)
plt.show()


# In[ ]:


# Analyzing the "int_rate" column
plt.figure(figsize = (16,6), dpi=120)
sns.countplot(x = np.rint(loan_df['int_rate']),data=loan_df)
plt.xlabel("Interest % Rate", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Interest Rate Normal Distribuition", fontsize=20)
plt.show()


# In[ ]:


# Analyzing the "dti" column
plt.figure(figsize=(16,16), dpi=120)

plt.subplot(2,2,1)
g = sns.distplot(loan_df["dti"])
g.set_xlabel("Debit Income Ratio", fontsize = xlabel_font)
g.set_ylabel("Distribution", fontsize = xlabel_font)
g.set_title("Debit Income Ratio Distribuition", fontsize = title_font)

plt.subplot(2,2,2)
g1 = sns.violinplot(y="dti", data=loan_df, inner="quartile", palette="hls")
g1.set_xlabel("", fontsize = xlabel_font)
g1.set_ylabel("Debit Income Ratio", fontsize = ylabel_font)
g1.set_title("Debit Income Ratio Distribuition", fontsize= title_font)

plt.subplot(2,2,3)
g2 = sns.boxplot(x='dti', data=loan_df, orient='v')
g2.set_xlabel("", fontsize = xlabel_font)
g2.set_ylabel("Debit Income Ratio", fontsize = ylabel_font)
g2.set_title("Debit Income Ratio Distribuition", fontsize= title_font)

plt.show()


# In[ ]:


q = loan_df["loan_amnt"].quantile(0.9)
loan_df = loan_df[loan_df["loan_amnt"] < q] 
loan_df["loan_amnt"].describe()


# In[ ]:


# Analyzing the "loan_amnt" column
plt.figure(figsize=(16,16), dpi=120)

plt.subplot(2,2,1)
g = sns.distplot(loan_df["loan_amnt"])
g.set_xlabel("Amount", fontsize = xlabel_font)
g.set_ylabel("Distribution", fontsize = xlabel_font)
g.set_title("Loan Amount Distribuition", fontsize = title_font)

plt.subplot(2,2,2)
g1 = sns.violinplot(y="loan_amnt", data=loan_df, inner="quartile", palette="hls")
g1.set_xlabel("", fontsize = xlabel_font)
g1.set_ylabel("Amount", fontsize = ylabel_font)
g1.set_title("Loan Amount Distribuition", fontsize= title_font)

plt.subplot(2,2,3)
g2 = sns.boxplot(x='loan_amnt', data=loan_df, orient='v')
g2.set_xlabel("", fontsize = xlabel_font)
g2.set_ylabel("Loan Amount", fontsize = ylabel_font)
g2.set_title("Loan Amount Distribuition", fontsize= title_font)

plt.show()


# ### Outlier removal

# In[ ]:


loan_df = loan_df[loan_df['int_rate'] <= 22]
loan_df["int_rate"].describe()


# In[ ]:


# Analyzing the "int_rate" column
plt.figure(figsize=(16,16), dpi=120)

plt.subplot(2,2,1)
g = sns.distplot(loan_df["int_rate"])
g.set_xlabel("Interest Rate", fontsize = xlabel_font)
g.set_ylabel("Distribution", fontsize = xlabel_font)
g.set_title("Interest Rate Distribuition", fontsize = title_font)

plt.subplot(2,2,2)
g1 = sns.violinplot(y="int_rate", data=loan_df, inner="quartile", palette="hls")
g1.set_xlabel("", fontsize = xlabel_font)
g1.set_ylabel("Interest Rate", fontsize = ylabel_font)
g1.set_title("Interest Rate Distribuition", fontsize= title_font)

plt.subplot(2,2,3)
g2 = sns.boxplot(x='int_rate', data=loan_df, orient='v')
g2.set_xlabel("", fontsize = xlabel_font)
g2.set_ylabel("Interest Rate", fontsize = ylabel_font)
g2.set_title("Interest Rate Distribuition", fontsize= title_font)

plt.show()


# ### Outlier Removal

# In[ ]:


q = loan_df["annual_inc"].quantile(0.948)
loan_df = loan_df[loan_df["annual_inc"] < q]
loan_df["annual_inc"].describe()


# In[ ]:


# Analyzing the "annual_inc" column
plt.figure(figsize=(16,16), dpi=120)

plt.subplot(2,2,1)
g = sns.distplot(loan_df["annual_inc"])
g.set_xlabel("Annual Income", fontsize = xlabel_font)
g.set_ylabel("Distribution", fontsize = xlabel_font)
g.set_title("Annual Income Distribuition", fontsize = title_font)

plt.subplot(2,2,2)
g1 = sns.violinplot(y="annual_inc", data=loan_df, inner="quartile", palette="hls")
g1.set_xlabel("", fontsize = xlabel_font)
g1.set_ylabel("Annual Income", fontsize = ylabel_font)
g1.set_title("Annual Income Distribuition", fontsize= title_font)

plt.subplot(2,2,3)
g2 = sns.boxplot(x='annual_inc', data=loan_df, orient='v')
g2.set_xlabel("", fontsize = xlabel_font)
g2.set_ylabel("Annual Income", fontsize = ylabel_font)
g2.set_title("Annual Income Distribuition", fontsize= title_font)

plt.show()


# ## Bivariate Analysis 

# In[ ]:


plt.figure(figsize=(12,20))
sns.boxplot(data =loan_df, y='purpose', x='loan_amnt', hue ='loan_status')
plt.xlabel('Loan Amount',fontsize=xlabel_font+6)
plt.ylabel('Purpose',fontsize=ylabel_font+6)
plt.title('Purpose vs Loan Amount',fontsize=title_font)
plt.show()


# In[ ]:


# ['Employment Length vs Loan Amount']

loanstatus=loan_df.pivot_table(index=['loan_status','purpose','emp_length'],values='loan_amnt',
                               aggfunc=('count')).reset_index()

loanstatus=loan_df.loc[loan_df['loan_status']=='Charged Off']

plt.figure(figsize=(12, 26),dpi=120)
sns.boxplot(y='emp_length', x='loan_amnt', hue='purpose', data=loanstatus)
plt.title('Employment Length vs Loan Amount for different pupose of Loan',fontsize=title_font)
plt.ylabel('Employment Length',fontsize=xlabel_font)
plt.xlabel('Loan Amount',fontsize=ylabel_font)
plt.show()


# ## Correation

# In[ ]:


plt.figure(figsize=(16,16),dpi=120)
plt.figure(figsize=(20,20))
sns.set_context(font_scale=2)
sns.heatmap(loan_df.corr(), annot=False, cmap='bwr', square=True, linewidths=0.8)
plt.show()


# ##  Binning

# In[ ]:


# ['dti']
bins = [0, 5, 10, 15, 20, 30]
slot = ['0-5', '5-10', '10-15', '15-20', '20 and above']
loan_df['dti_range'] = pd.cut(loan_df['dti'], bins, labels=slot)
test_2008['dti_range'] = pd.cut(test_2008['dti'], bins, labels=slot)

# ['loan_amnt']
bins = [0, 5000, 10000, 15000, 20000, 25000 ,40000]
slot = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000','25000 and above']
loan_df['loan_amnt_range'] = pd.cut(loan_df['loan_amnt'], bins, labels=slot)
test_2008['loan_amnt_range'] = pd.cut(test_2008['loan_amnt'], bins, labels=slot)

# ['annual_inc']
bins = [0, 25000, 50000, 75000, 100000, 1000000]
slot = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000 and above']
loan_df['annual_inc_range'] = pd.cut(loan_df['annual_inc'], bins, labels=slot)
test_2008['annual_inc_range'] = pd.cut(test_2008['annual_inc'], bins, labels=slot)

# ['int_rate']
bins = [0, 7.5, 10, 12.5, 15,20]
slot = ['0-7.5', '7.5-10', '10-12.5', '12.5-15', '15 and above']
loan_df['int_rate_range'] = pd.cut(loan_df['int_rate'], bins, labels=slot)
test_2008['int_rate_range'] = pd.cut(test_2008['int_rate'], bins, labels=slot)


# ## Conclusion from Visualization
# ### Target Variable
#     Loan Status
# 
# ### Major variables to consider for loan prediction:
#     Loan Amount
#     Purpose of Loan
#     Employment Length
#     Grade
#     Interest Rate
#     Term
#     Annual Income
#     Debit to Income Ratio(dti)
#     Verification Status
#     Home Ownership
#     Does Not Meet Credit Policy

# ## Model Development

# In[ ]:


cols = ['loan_amnt_range','purpose','emp_length','grade','int_rate_range','term','annual_inc_range',
        'loan_status','dti', 'verification_status','home_ownership', 'dnmcp']


# In[ ]:


test_2008_df = test_2008[cols]
train_test_data = loan_df[loan_df.issue_yr != 2008]  
df = train_test_data[cols]


# In[ ]:


new_df = df.assign(
    loan_amnt_range = df.loan_amnt_range.astype('category').cat.codes,
    purpose = df.purpose.astype('category').cat.codes,
    emp_length = df.emp_length.astype('category').cat.codes,
    grade = df.grade.astype('category').cat.codes,
    int_rate_range = df.int_rate_range.astype('category').cat.codes,
    term = df.term.astype('category').cat.codes,
    annual_inc_range = df.annual_inc_range.astype('category').cat.codes,
    home_ownership = df.home_ownership.astype('category').cat.codes,
    verification_status = df.verification_status.astype('category').cat.codes
)


# In[ ]:


# Train-test split
X = new_df.drop(['loan_status'],axis=1)
X = preprocessing.normalize(X)
y = new_df['loan_status']


# In[ ]:


y_original = y  


# In[ ]:


Y = y_original.astype('category').cat.codes 


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2,random_state = 42)


# In[ ]:


print("----------------------------Logistic Regression------------------------------")

l_model = LogisticRegression()
l_model.fit(X_train,y_train)
l_score = cross_val_score(l_model,X_train,y_train,cv=10)
y_pred = l_model.predict(X_test)

print(classification_report(y_pred,y_test))
print("\n\n Confusion Matrix\n")
print(confusion_matrix(y_pred,y_test))
print() 
print("Model Accuracy Avg: ", l_score.mean()*100)
print("Test Accuracy: ",accuracy_score(y_pred,y_test)*100)


# In[ ]:


print("----------------------------Stochastic Gradient Descent------------------------------")

sgd_model = SGDClassifier(class_weight='balanced', loss='modified_huber')
sgd_model.fit(X_train,y_train)
sgd_score = cross_val_score(sgd_model,X_train,y_train,cv=10)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_pred,y_test))
print("\n\n Confusion Matrix\n")
print(confusion_matrix(y_pred,y_test))
print() 
print("Model Accuracy Avg: ", l_score.mean()*100)
print("Test Accuracy: ",accuracy_score(y_pred,y_test)*100)


# In[ ]:


# print("----------------------------Stochastic Gradient Descent with Bootstrap Aggregation------------------------------")

# seed = 8
# kfold = model_selection.KFold(n_splits = 2, random_state = seed) 


# # initialize the base classifier 
# base_cls = SGDClassifier(class_weight='balanced',loss='modified_huber') 
  
# # no. of base classifier 
# num_est = 800 #400 , 600, 700, 1000
  
# # bagging classifier 
# sgdl_model = BaggingClassifier(base_estimator = base_cls, 
#                           n_estimators = num_est, 
#                           random_state = seed) 

# sgdl_model.fit(X_train,y_train)

# y_pred = sgdl_model.predict(X_test)
# prob_score = sgdl_model.predict_proba(X_test)

# print("\nConfusion Matrix\n")
# print(confusion_matrix(y_test,y_pred))
# print() 
# print("Accuracy: ",accuracy_score(y_test,y_pred)*100)


# ## Testing models against data for year 2008 from Quarters 1 to 4
# 

# In[ ]:


test_2008_df = test_2008_df.assign(
    loan_amnt_range = test_2008_df.loan_amnt_range.astype('category').cat.codes,
    purpose = test_2008_df.purpose.astype('category').cat.codes,
    emp_length = test_2008_df.emp_length.astype('category').cat.codes,
    grade = test_2008_df.grade.astype('category').cat.codes,
    int_rate_range = test_2008_df.int_rate_range.astype('category').cat.codes,
    term = test_2008_df.term.astype('category').cat.codes,
    annual_inc_range = test_2008_df.annual_inc_range.astype('category').cat.codes,
    home_ownership = test_2008_df.home_ownership.astype('category').cat.codes,
    verification_status = test_2008_df.verification_status.astype('category').cat.codes
) 

test_q1 = test_2008_df[test_2008.issue_qtr == 1]
test_q2 = test_2008_df[test_2008.issue_qtr == 2]
test_q3 = test_2008_df[test_2008.issue_qtr == 3]
test_q4 = test_2008_df[test_2008.issue_qtr == 4]


# In[ ]:


lr_test_score = []
sgd_test_score = [] 


# ###  Test for 2008 Quarter 1

# In[ ]:


X1 = test_q1.drop(['loan_status'],axis=1)
X1 = preprocessing.normalize(X1)
y1 = test_q1['loan_status']
print("Testing Data: ", test_q1.shape[0])

# Logistic Regression Model
y_pred =  cross_val_predict(l_model,X1,y1, cv=10)
acc = accuracy_score(y1,y_pred)
lr_test_score.append(acc)
print("\nLogistic Regression : ", acc*100)

# Logistic Regression Model
y_pred =  cross_val_predict(sgd_model,X1,y1, cv=10)
acc = accuracy_score(y1,y_pred)
lr_test_score.append(acc)
print("\nSGD: ", acc*100)


# ###  Test for 2008 Quarter 2

# In[ ]:


X2 = test_q2.drop(['loan_status'],axis=1)
X2 = preprocessing.normalize(X2)
y2 = test_q2['loan_status']
print("Testing Data: ", test_q2.shape[0])

# Logistic Regression Model
y_pred =  cross_val_predict(l_model,X2,y2, cv=10)
acc = accuracy_score(y2,y_pred)
lr_test_score.append(acc)
print("\nLogistic Regression : ", acc*100)

# Logistic Regression Model
y_pred =  cross_val_predict(sgd_model,X2,y2, cv=10)
acc = accuracy_score(y2,y_pred)
lr_test_score.append(acc)
print("\nSGD: ", acc*100)


# ###  Test for 2008 Quarter 3

# In[ ]:


X3 = test_q3.drop(['loan_status'],axis=1)
X3 = preprocessing.normalize(X3)
y3 = test_q3['loan_status']
print("Testing Data: ", test_q3.shape[0])

# Logistic Regression Model
y_pred =  cross_val_predict(l_model,X3,y3, cv=10)
acc = accuracy_score(y3,y_pred)
lr_test_score.append(acc)
print("\nLogistic Regression : ", acc*100)

# Logistic Regression Model
y_pred =  cross_val_predict(sgd_model,X3,y3, cv=10)
acc = accuracy_score(y3,y_pred)
lr_test_score.append(acc)
print("\nSGD: ", acc*100)


# ###  Test for 2008 Quarter 4

# In[ ]:


X4 = test_q3.drop(['loan_status'],axis=1)
X4 = preprocessing.normalize(X4)
y4 = test_q3['loan_status']
print("Testing Data : ", test_q4.shape[0])

# Logistic Regression Model
y_pred =  cross_val_predict(l_model,X4,y4, cv=10)
acc = accuracy_score(y4,y_pred)
lr_test_score.append(acc)
print("\nLogistic Regression : ", acc*100)

# Logistic Regression Model
y_pred =  cross_val_predict(sgd_model,X4,y4, cv=10)
acc = accuracy_score(y4,y_pred)
lr_test_score.append(acc)
print("\nSGD: ", acc*100)


# In[ ]:




