#!/usr/bin/env python
# coding: utf-8

# ![](https://github.com/rajat5ranjan/AV-LTFS-Data-Science-FinHack-ML-Hackathon/raw/2853f792147b4305cad1b40d75893dab112e6611/ltfs.jpg)

# Hello Everyone !
# 
# This kernel consists of my work for the **AV - LTFS Hackathon** where we were supposed to predict the loan defaulters in the first month of EMI payment.
# 
# I have tried some feature engineering first up, followed by parameter tuning of CatBoost and then a 1-Layer Stacking of the different base models.
# 
# Other than CatBosst, XGBoost,LightGBM,RF,NNs were also tried, but they were giving sub-optimal results.
# 
# On hind sight, a bit more extensive feature engineering would have helped in boosting the score further up.
# 
# This kernel gets a 
# 
# **CV Score - 0.6752**
# 
# **Public LB Score - 0.6636**       ( Rank - 53rd / 1352 )
# 
# **Private LB Score - 0.667127**    ( Rank - 47th / 1352 )
# 
# (AUC-ROC Metric)

# 

# **Basic Overview of the things done in the kernel before jumping into the coding part - 
# **
# 
# 1. FEATURE ENGINEERING - 
#       *   **Anomalous Branch** - Keeps track of the branches, from where, certain loans have been sanctioned and then the buy has been done at a showroom  which is far from that bank,possibly even in a **different state or city**. This is tracked by seeing the usual showrooms from where buys take place if a loan is sanctioned from a particular branch. Certain anomalies detected in this list have been tracked in this feature.
#       *   The super-messy Perform_CNS Score categorical data have been **re-binned** to give a cleaner idea of the CIBIL scores. There have been 2 new binnings made. One on the basis of some background knowledge about banking, and the way banks segregate the users and the second according to the data provided in the dataset.
#       *   The number of ID Proofs a person has submitted at the time of taking the loan - Assumption being, **the more number of IDs shown, the more the credibility of the borrower.**
#       *   The number of Primary and Secondary accounts a person already has defaulted, overall as well as over the last 6 months.
#       *   The borrower's age, his/her average account age,i.e, on an average how much time he/she takes to give back all the lent money.
#       *   Whether the borrower is a **"Student"** or a **"Senior Citizen"** from the age and the employment status.
#       *   Since the model was suppposed to predict who would be defaulting in the **FIRST MONTH** of taking loan, so, keeping track of which all users defaulted in the **first month only**, rather than who all defaulted over-all in the train set makes more sense as the model would then be able to recognize the trends and behaviour more easily.
#       
# 2. MISSING DATA HANDLING -
#       *   The missing "Employment" were treated as **"Unemployed"** as of that moment.
#       *   The UNEMPLOYED borrowers were categorized into "Students" and "Senior Citizens" taking a hint from their ages.
#       
# 3. STRATIFICATION - 
# 
#       *   Stratification done on the basis of **similarity between the train and test set**, rather than doing on the basis of the classes.
#     
# 4. TRAINING -
# 
#       *   **Five-Fold Cross Validation** was used and the predictions on the test set were taken over the model trained on each fold and were finally averaged over all the folds to get the final prediction over the test set.
#       *   Heavy **Parameter Tuning done on CatBoost Classifier**, LightGBM, Random Forest and XGBoost, with CatBoost out-performing the rest. Hence, finally CatBoost was used for submission.
#       *   **Stacking** was done, with 20 CatBoost models and a meta learner (Logistic Regression) was used.
#      
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split,cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from tqdm import tqdm_notebook as tqdm
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **READING THE FILES**

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test_bqCt9Pv.csv')


# In[ ]:


#Lets have a look at the data

train.head()


# In[ ]:


#Lets see how skewed the data is wrt the number of data points belonging to each class.

train['loan_default'].value_counts()


# **FEATURE ENGINEERING - MAKING UP AS MANY NEW INNOVATIVE FEATURES AS I COULD COME UP WITH**

# The logic behind this feature is that, if a person takes a loan from a particular branch, in normal cases, we would expect him to buy the vehicle from a showroom/retailer which is located in the same city ( or the same state in worst case scenario ). 
# 
# So, my assumption was that every branch serves to customers who then go to one of the showrooms of a disjoint set, i.e, ideally, there should be a set of showrooms from where if a customer is buying a vehicle, then he must be getting it funded from a particular branch.
# 
# Though this seemed to me to be somewhat logical assumption, it *didnt* really turn out to be that good a differentiator.

# In[ ]:


branchList = train['branch_id'].unique()
branchSupId = train.groupby('branch_id')['supplier_id'].unique()

branchSupIdList = []
anomalousBranch = []

for bra in range(len(branchList)):
    branchId = branchList[bra]
    branchSupIdList.append(branchSupId[branchId])

for i in range(len(branchSupIdList)):
  for j in range(len(branchSupIdList)):
    if(i != j):
      #print(len(list(set(branchSupIdList[i]).intersection(set(branchSupIdList[j])))))
      if ((len(list(set(branchSupIdList[i]).intersection(set(branchSupIdList[j]))))) != 0):  
        if (len(list(set(branchSupIdList[i]).intersection(set(branchSupIdList[j]))))) >= 3:  
          #Both branches in the same locality.
          continue
        else:
          anomalousBranch.append(branchList[i])
      else:
        #Disjoint Branches
        continue
    else:
      continue  


# In[ ]:


def isBranchAnomalous(x):
  if (x in anomalousBranch):
    return 1
  else:
    return 0


# **CIBIL features are made from a bit of background knowledge. This is the usual score used by financial institutions in order to decide whether to lend money to a person or not.**

# * The 'PERFORM_CNS.SCORE.DESCRIPTION' column has a lot of bins and there are many different kinds of bins which essentially represent more or less the same set of people/distribution of customers. 
# 
# **Ex - Different types of "High Risk", "Low Risk" etc.**

# In[ ]:


def CIBIL_norm(x):
    a=''
    if((x=='A-Very Low Risk') or (x=='B-Very Low Risk') or (x=='C-Very Low Risk') or (x=='D-Very Low Risk')):
        a = 'Very Low Risk'
    elif((x=='M-Very High Risk')):
        a = 'Very Very High Risk'
    elif((x=='L-Very High Risk')):
        a='Very High Risk'
    elif((x=='E-Low Risk') or (x=='F-Low Risk') or (x=='G-Low Risk')):
        a = 'Low Risk'
    elif((x=='H-Medium Risk') or (x=='I-Medium Risk')):
        a = 'Medium Risk'
    elif((x=='J-High Risk') or (x=='K-High Risk')):
        a = 'High Risk'
    elif((x=='Not Scored: No Activity seen on the customer (Inactive)') or (x=='Not Scored: No Updates available in last 36 months')):
        a = 'Inactive'
    elif((x=='Not Scored: Only a Guarantor')):
        a='Guarantor'
    elif((x=='Not Scored: More than 50 active Accounts found')):
        a='SuperActive'
    else:
        a='Others'
    return a


# In[ ]:


def CIBIL_other(x):
    a=''
    if((x=='A-Very Low Risk') or (x=='B-Very Low Risk') or (x=='C-Very Low Risk') or (x=='D-Very Low Risk')):
        a = 'Very Low Risk'
    elif((x=='M-Very High Risk')):
        a = 'Very Very High Risk'
    elif((x=='L-Very High Risk')):
        a='Very High Risk'
    elif((x=='E-Low Risk') or (x=='F-Low Risk') or (x=='G-Low Risk')):
        a = 'Low Risk'
    elif((x=='H-Medium Risk') or (x=='I-Medium Risk')):
        a = 'Medium Risk'
    elif((x=='J-High Risk') or (x=='K-High Risk')):
        a = 'High Risk'
    elif((x=='Not Scored: No Activity seen on the customer (Inactive)') or (x=='Not Scored: No Updates available in last 36 months')):
        a = 'Inactive'
    elif((x=='Not Scored: Only a Guarantor')):
        a='Guarantor'
    elif((x=='Not Scored: More than 50 active Accounts found')):
        a='SuperActive'
    elif((x=='No Bureau History Available') or (x=='Not Scored: Sufficient History Not Available') or (x=='Not Scored: Not Enough Info available on the customer')):  
        a='NoHistory'
    else:
        a='Others'
    return a


# In[ ]:


def CIBIL_trend(x):
    a=''
    if(x==300):
        a='Very Poor'
    elif((x>300) and (x<=550)):
        a='Poor'
    elif((x>550) and (x<=650)):
        a='Fair'
    elif((x>650) and (x<=750)):
        a='Good'
    elif((x>750) and (x<=900)):
        a='Excellent'
    else:
        a='Others'
    return a


# The number of ID proofs submitted by a person while taking a loan.  - > Thought behind this is, **More the number of ID proofs a person submits, more is the chance of that person being a genuine person** and not someone who is intentionally going to default in EMI Payments.

# In[ ]:


def NumIds(x):
    a=''
    if(x==1):
        a = 'One'
    elif(x==2):
        a='Two'
    elif(x==3):
        a='Three'
    else:
        a='Four'
    return a


# **Age Calculation from the DOB column**

# In[ ]:


def calcAge(x):
    year = int(x.split('-')[2])
    if(year<=19):
        age = 20-year
    else:
        age = 100 + (20-year)
    return age


# **Remarks on the basis of the number of Primary and Secondary Defaulted accounts of that person.**

# In[ ]:


def PrimaDefault(x):
    a=''
    if(x==-1):
        a='First'
    elif(x==0):
        a='Great'
    elif(x<=0.2):
        a='Normal'
    elif(x<=0.4):
        a='Bothersome'
    elif(x<=0.6):
        a='Trouble'
    elif(x<=0.8):
        a='Danger'
    else:
        a='High Alert'
    return a

def SecDefault(x):
    a=''
    if(x==-1):
        a='First'
    elif(x==0):
        a='Great'
    elif(x<=0.2):
        a='Normal'
    elif(x<=0.4):
        a='Bothersome'
    elif(x<=0.6):
        a='Trouble'
    elif(x<=0.8):
        a='Danger'
    else:
        a='High Alert'
    return a

def TotDefault(x):
    a=''
    if(x==-1):
        a='First'
    elif(x==0):
        a='Great'
    elif(x<=0.2):
        a='Normal'
    elif(x<=0.4):
        a='Bothersome'
    elif(x<=0.6):
        a='Trouble'
    elif(x<=0.8):
        a='Danger'
    else:
        a='High Alert'
    return a


# Keeping track of the number of primary default accounts in the last 6 months and assigning a remark to that.

# In[ ]:


def PrimaDefaultLastSix(x):
    a=''
    if(x==0):
        a='Great'
    elif(x==1):
        a='Normal'
    elif(x==2):
        a='Bothersome'
    elif(x>=3):
        a='Trouble'
    else:
        a='Others'
    return a


# Defining a new feature based on the Number of outstanding Balance accounts the customer has. 
# 
# The idea behind this being, **more the number of accounts a customer has with outstanding balance**, **the less reliable** he would be expected to be.

# In[ ]:


def CurrOutstandingBal(x):
    a=''
    if (x==0):
        a='Good'
    elif((x<0) and (x!=-1)):
        a='Very Good'
    elif(x>0 and x<=1):
        a='Both'
    elif(x>1):
        a='Problematic'
    else:
        a='Other'
    return a


# In[ ]:


def AvgAcctAge(x):
    year = int(x.split(" ")[0].split("y")[0])
    month = int(x.split(" ")[1].split("m")[0])
    time_int = (12*year) + month
    return time_int


# In[ ]:


def oneMonth(x):
    if(int(x.split('-')[1])==10):
        return 1
    else:
        return 0


# **Feature representing the number of identity cards given by the customer.**

# In[ ]:


oneHot = train[['Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag','Passport_flag']]
oneHot['sum'] = oneHot['Aadhar_flag'] + oneHot['PAN_flag'] + oneHot['VoterID_flag'] + oneHot['Driving_flag'] + oneHot['Passport_flag'] 
train['NumIDs'] = oneHot['sum']

oneHotTest = test[['Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag','Passport_flag']]
oneHotTest['sum'] = oneHotTest['Aadhar_flag'] + oneHotTest['PAN_flag'] + oneHotTest['VoterID_flag'] + oneHotTest['Driving_flag'] + oneHotTest['Passport_flag'] 
test['NumIDs'] = oneHotTest['sum']


# **Applying the functions defined above, and thus, the final steep in actually implementing the features planned above.**

# Certain Scores on CNS Score like 11,14,15,16,17,18 were all made 0 as they all were corresponding to cases with less/no history of the borrower being available.

# In[ ]:


#Simply Calling the functions defined above for feature engineering.

#Train set dataframe manipulation

train['NumIDsCnt'] = train['NumIDs'].apply(NumIds)
train['IDsCount'] = np.where(train['NumIDs']>1,1,0)

train['Age']=train['Date.of.Birth'].apply(calcAge)

train['isStudent'] = np.where(train['Age']<=25,1,0)
train['isSenior'] = np.where(train['Age']>=60,1,0)

train['Employment.Type'] = np.where(train['Employment.Type'].isnull(),'Unemployed',train['Employment.Type'])

train['leftover'] = train['asset_cost'] - train['disbursed_amount']
train['loanRatio'] = (train['disbursed_amount']/train['asset_cost'])*100

train['CIBIL_Descr'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].apply(CIBIL_norm)
train['CIBIL_Other'] = train['PERFORM_CNS.SCORE.DESCRIPTION'].apply(CIBIL_other)
train['CIBIL_Trend'] = train['PERFORM_CNS.SCORE'].apply(CIBIL_trend)

train['PriOverduePercentage'] = np.where(train['PRI.NO.OF.ACCTS'] != 0,train['PRI.OVERDUE.ACCTS']/train['PRI.NO.OF.ACCTS'],-1) 
train['SecOverduePercentage'] = np.where(train['SEC.NO.OF.ACCTS'] != 0,train['SEC.OVERDUE.ACCTS']/train['SEC.NO.OF.ACCTS'],-1) 

train['PrimaDefaultRemark'] = train['PriOverduePercentage'].apply(PrimaDefault)
train['SecoDefaultRemark'] = train['SecOverduePercentage'].apply(SecDefault)
train['totalDefaultPercent'] = np.where((train['PRI.NO.OF.ACCTS'] + train['SEC.NO.OF.ACCTS']) != 0,(train['PRI.OVERDUE.ACCTS'] + train['SEC.OVERDUE.ACCTS'])/(train['PRI.NO.OF.ACCTS'] + train['SEC.NO.OF.ACCTS']),-1)    
train['TotaDefaultRemark'] = train['totalDefaultPercent'].apply(TotDefault) 

train['AcctsLastSixRemarks'] = train['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].apply(PrimaDefaultLastSix)

train['PRIcritRatio'] = np.where(train['PRI.DISBURSED.AMOUNT'] != 0,train['PRI.CURRENT.BALANCE']/train['PRI.DISBURSED.AMOUNT'],-1)
train['SECcritRatio'] = np.where(train['SEC.DISBURSED.AMOUNT'] != 0,train['SEC.CURRENT.BALANCE']/train['SEC.DISBURSED.AMOUNT'],-1)

train['TOT.DISBURSED.AMOUNT'] = train['PRI.DISBURSED.AMOUNT'] + train['SEC.DISBURSED.AMOUNT']
train['TOT.CURRENT.BALANCE'] = train['PRI.CURRENT.BALANCE'] + train['SEC.CURRENT.BALANCE']
train['TOTcritRatio'] = np.where(train['TOT.DISBURSED.AMOUNT'] != 0,train['TOT.CURRENT.BALANCE']/train['TOT.DISBURSED.AMOUNT'],-1)

train['PriRatioRemark'] = train['PRIcritRatio'].apply(CurrOutstandingBal)
train['SecRatioRemark'] = train['SECcritRatio'].apply(CurrOutstandingBal)
train['TotRatioRemark'] = train['TOTcritRatio'].apply(CurrOutstandingBal)

train["AvgAcctAge"] = train['AVERAGE.ACCT.AGE'].apply(AvgAcctAge)
train['CredAcctAge'] = train['CREDIT.HISTORY.LENGTH'].apply(AvgAcctAge)

train['OneMonthDef'] = train['DisbursalDate'].apply(oneMonth)

train['PERFORM_CNS.SCORE'] = np.where(train['PERFORM_CNS.SCORE']==11,0,train['PERFORM_CNS.SCORE'])
train['PERFORM_CNS.SCORE'] = np.where(train['PERFORM_CNS.SCORE']==14,0,train['PERFORM_CNS.SCORE'])
train['PERFORM_CNS.SCORE'] = np.where(train['PERFORM_CNS.SCORE']==15,0,train['PERFORM_CNS.SCORE'])
train['PERFORM_CNS.SCORE'] = np.where(train['PERFORM_CNS.SCORE']==16,0,train['PERFORM_CNS.SCORE'])
train['PERFORM_CNS.SCORE'] = np.where(train['PERFORM_CNS.SCORE']==17,0,train['PERFORM_CNS.SCORE'])
train['PERFORM_CNS.SCORE'] = np.where(train['PERFORM_CNS.SCORE']==18,0,train['PERFORM_CNS.SCORE'])

train['isBranchAnomalous'] = train['branch_id'].apply(isBranchAnomalous)

#Test Set dataframe manipulation


test['isBranchAnomalous'] = test['branch_id'].apply(isBranchAnomalous)

test['NumIDsCnt'] = test['NumIDs'].apply(NumIds)
test['IDsCount'] = np.where(test['NumIDs']>1,1,0)

test['Age']=test['Date.of.Birth'].apply(calcAge)

test['isStudent'] = np.where(test['Age']<=25,1,0)
test['isSenior'] = np.where(test['Age']>=60,1,0)

test['Employment.Type'] = np.where(test['Employment.Type'].isnull(),'Unemployed',test['Employment.Type'])

test['leftover'] = test['asset_cost'] - test['disbursed_amount']
test['loanRatio'] = (test['disbursed_amount']/test['asset_cost'])*100

test['CIBIL_Descr'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].apply(CIBIL_norm)
test['CIBIL_Other'] = test['PERFORM_CNS.SCORE.DESCRIPTION'].apply(CIBIL_other)
test['CIBIL_Trend'] = test['PERFORM_CNS.SCORE'].apply(CIBIL_trend)

test['PriOverduePercentage'] = np.where(test['PRI.NO.OF.ACCTS'] != 0,test['PRI.OVERDUE.ACCTS']/test['PRI.NO.OF.ACCTS'],-1) 
test['SecOverduePercentage'] = np.where(test['SEC.NO.OF.ACCTS'] != 0,test['SEC.OVERDUE.ACCTS']/test['SEC.NO.OF.ACCTS'],-1) 

test['PrimaDefaultRemark'] = test['PriOverduePercentage'].apply(PrimaDefault)
test['SecoDefaultRemark'] = test['SecOverduePercentage'].apply(SecDefault)
test['totalDefaultPercent'] = np.where((test['PRI.NO.OF.ACCTS'] + test['SEC.NO.OF.ACCTS']) != 0,(test['PRI.OVERDUE.ACCTS'] + test['SEC.OVERDUE.ACCTS'])/(test['PRI.NO.OF.ACCTS'] + test['SEC.NO.OF.ACCTS']),-1)    
test['TotaDefaultRemark'] = test['totalDefaultPercent'].apply(TotDefault) 

test['AcctsLastSixRemarks'] = test['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'].apply(PrimaDefaultLastSix)

test['PRIcritRatio'] = np.where(test['PRI.DISBURSED.AMOUNT'] != 0,test['PRI.CURRENT.BALANCE']/test['PRI.DISBURSED.AMOUNT'],-1)
test['SECcritRatio'] = np.where(test['SEC.DISBURSED.AMOUNT'] != 0,test['SEC.CURRENT.BALANCE']/test['SEC.DISBURSED.AMOUNT'],-1)

test['TOT.DISBURSED.AMOUNT'] = test['PRI.DISBURSED.AMOUNT'] + test['SEC.DISBURSED.AMOUNT']
test['TOT.CURRENT.BALANCE'] = test['PRI.CURRENT.BALANCE'] + test['SEC.CURRENT.BALANCE']
test['TOTcritRatio'] = np.where(test['TOT.DISBURSED.AMOUNT'] != 0,test['TOT.CURRENT.BALANCE']/test['TOT.DISBURSED.AMOUNT'],-1)

test['PriRatioRemark'] = test['PRIcritRatio'].apply(CurrOutstandingBal)
test['SecRatioRemark'] = test['SECcritRatio'].apply(CurrOutstandingBal)
test['TotRatioRemark'] = test['TOTcritRatio'].apply(CurrOutstandingBal)

test["AvgAcctAge"] = test['AVERAGE.ACCT.AGE'].apply(AvgAcctAge)
test['CredAcctAge'] = test['CREDIT.HISTORY.LENGTH'].apply(AvgAcctAge)

test['OneMonthDef'] = test['DisbursalDate'].apply(oneMonth)

test['PERFORM_CNS.SCORE'] = np.where(test['PERFORM_CNS.SCORE']==11,0,test['PERFORM_CNS.SCORE'])
test['PERFORM_CNS.SCORE'] = np.where(test['PERFORM_CNS.SCORE']==14,0,test['PERFORM_CNS.SCORE'])
test['PERFORM_CNS.SCORE'] = np.where(test['PERFORM_CNS.SCORE']==15,0,test['PERFORM_CNS.SCORE'])
test['PERFORM_CNS.SCORE'] = np.where(test['PERFORM_CNS.SCORE']==16,0,test['PERFORM_CNS.SCORE'])
test['PERFORM_CNS.SCORE'] = np.where(test['PERFORM_CNS.SCORE']==17,0,test['PERFORM_CNS.SCORE'])
test['PERFORM_CNS.SCORE'] = np.where(test['PERFORM_CNS.SCORE']==18,0,test['PERFORM_CNS.SCORE'])


# In[ ]:


train.drop(columns=['Date.of.Birth','DisbursalDate','UniqueID'],inplace=True)
test.drop(columns=['Date.of.Birth','DisbursalDate','UniqueID'],inplace=True)


# **Making a function for easier Training and Cross-Validation - Using 5 fold stratified cross-validation**

# In[ ]:


def scoreOfModel(clf,X,y,flag,shuffleBool=False,nFolds=5):
    score = 0
    finalPreds = np.zeros(112392)
    trainPreds = np.zeros(233154)
    folds = StratifiedKFold(n_splits=nFolds, shuffle=shuffleBool, random_state=42)
    #train_pred = cross_val_predict(clf, X, y, cv=12,method='predict_proba')
    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(X,stratCol))):
        X_train,X_val = X.loc[trn_idx,:],X.loc[val_idx,:]
        y_train,y_val = y[trn_idx],y[val_idx]
        clf.fit(X_train,y_train)
        yPreds = clf.predict_proba(X_val)
        yPredsTweaked = yPreds[:,1]
        trainPreds[val_idx] = yPredsTweaked
        score += roc_auc_score(y_val,yPredsTweaked)
        p = clf.predict_proba(x_test)
        #Adding the probabilities of belonging to the class "1".
        for k in range(len(p)):
            finalPreds[k] += p[k][1]
        print("**********"+ str(score/(1+fold_)) + "******************Iteration "+str(fold_)+" Done****************")    
    return str(score/nFolds),(trainPreds),(finalPreds/nFolds)  


# In[ ]:


labelEnc = ['AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH',
'branch_id', 'supplier_id','manufacturer_id','State_ID','PERFORM_CNS.SCORE.DESCRIPTION','Current_pincode_ID',
                                          'isStudent','isSenior','Employment.Type',
                                                  'CIBIL_Trend','AcctsLastSixRemarks','OneMonthDef',
                                                  'NumIDsCnt','CIBIL_Descr','CIBIL_Other',
                                                  'PrimaDefaultRemark','SecoDefaultRemark','TotaDefaultRemark',
                                                  'PriRatioRemark','SecRatioRemark','TotRatioRemark',  'Employee_code_ID'
                                                    ]


# In[ ]:


X = train.drop(columns=['loan_default'])
y = train['loan_default']


# **Using "similarity between the train and test columns" as the stratification. **
# 
# Instead of using the class labels as stratification,using the similarity between the train and test set as a parameter for stratification tends to give a better model, considering that the model gets an idea about how similar/dissimilar the data points in train and test set are.

# In[ ]:


data = pd.concat([X, test], axis = 0)

X_newLGB = data.copy()
#test_newLGB = test.copy()
for col in labelEnc:
  le = LabelEncoder()
  data[col] = le.fit_transform(data[col])
  #X_newLGB[col] = le.fit_transform(data[col])
  #test_newLGB[col] = le.transform(test[col])

data['is_test'] = np.zeros(345546)

#(data.iloc[:233154,:])['is_test'] = 0
data.iloc[233154:,-1] = 1

train_examples = train.shape[0]

data_x = data.drop('is_test', axis=1)
data_y = data['is_test']

is_test_probs = cross_val_predict(RandomForestClassifier(max_depth = 7,n_estimators=200), data_x, data_y, method='predict_proba')[:train_examples]

is_test_Probs = is_test_probs[:,1]

from scipy.stats import rankdata

data.iloc[:233154,-1] = rankdata(is_test_Probs)
bins = np.histogram(data.iloc[:233154,-1])[1][:-1]
#train['is_test_bins'] = np.digitize(X_newLGB['is_test'], bins)
stratCol = np.digitize(data.iloc[:233154,-1], bins)


# In[ ]:


x_train = data.iloc[:233154,:]
x_test = data.iloc[233154:,:]


# **Categorical Features**

# In[ ]:


catFeatures = ['AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','branch_id', 'supplier_id','manufacturer_id','State_ID','PERFORM_CNS.SCORE.DESCRIPTION','Current_pincode_ID',
                                          'isStudent','isSenior','Employment.Type',
                                                  'CIBIL_Trend','AcctsLastSixRemarks','OneMonthDef',
                                                  'NumIDsCnt','CIBIL_Descr','CIBIL_Other',
                                                  'PrimaDefaultRemark','SecoDefaultRemark','TotaDefaultRemark',
                                                  'PriRatioRemark','SecRatioRemark','TotRatioRemark',  'Employee_code_ID' 
                                         ]


# **Training different base models with different hyper-paramter settings for stacking.**

# For the final submission, I have stacked all the 20 base models with the hyper-parameters which I have mentioned in the comments below as well. 
# 
# Due to the restriction on the maximum time a kernel can run for getting committed on Kaggle, I am unable to run a complete stack of 20 models here.
# 
# Here I have trained only 3 models and stacked on those.
# 
# Hence, this result can be expected to be a bit *sub-optimal* than the max scores which I have actually achieved in the hackathon.

# In[ ]:


# catClf1 = CatBoostClassifier(learning_rate = 0.02147,iterations = 9997, l2_leaf_reg = 9985,scale_pos_weight = 3.662,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

catClf2 = CatBoostClassifier(learning_rate = 0.03185,iterations = 2000, l2_leaf_reg = 999.6,scale_pos_weight = 1.915,eval_metric='AUC',
                            silent = True,cat_features=catFeatures)

catClf3 = CatBoostClassifier(learning_rate = 0.03998,iterations = 1497, l2_leaf_reg = 49.97,scale_pos_weight = 2.207,eval_metric='AUC',
                            silent = True,cat_features=catFeatures)

# catClf4 = CatBoostClassifier(learning_rate = 0.02838,iterations = 5174, l2_leaf_reg = 6311,scale_pos_weight = 3.926,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf5 = CatBoostClassifier(learning_rate = 0.02373,iterations = 3174, l2_leaf_reg = 2739,scale_pos_weight = 2.228,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf6 = CatBoostClassifier(learning_rate = 0.02,iterations = 5336, l2_leaf_reg = 7763,scale_pos_weight = 2.012,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf7 = CatBoostClassifier(learning_rate = 0.03624,iterations = 5995, l2_leaf_reg = 9994,scale_pos_weight = 0.8615,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

catClf8 = CatBoostClassifier(learning_rate = 0.03365,iterations = 2001, l2_leaf_reg = 9943,scale_pos_weight = 4.617,eval_metric='AUC',
                            silent = True,cat_features=catFeatures)

# catClf9 = CatBoostClassifier(learning_rate = 0.03132,iterations = 9985, l2_leaf_reg = 9988,scale_pos_weight = 0.6724,eval_metric='AUC',
#                            silent = True,cat_features=catFeatures)

#catClf10 = CatBoostClassifier(learning_rate = 0.0379,iterations = 2001, l2_leaf_reg = 5651,scale_pos_weight = 3.447,eval_metric='AUC',
#                            silent = True,cat_features=catFeatures)

#catClf11 = CatBoostClassifier(learning_rate = 0.02852,iterations = 2015, l2_leaf_reg = 2005,scale_pos_weight = 1.301,eval_metric='AUC',
#                            silent = True,cat_features=catFeatures)

# catClf12 = CatBoostClassifier(learning_rate = 0.02705,iterations = 6150, l2_leaf_reg = 9998,scale_pos_weight = 4.336,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf13 = CatBoostClassifier(learning_rate = 0.02706,iterations = 9996, l2_leaf_reg = 6891,scale_pos_weight = 0.596,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf14 = CatBoostClassifier(learning_rate = 0.0396,iterations = 5582, l2_leaf_reg = 2002,scale_pos_weight = 2.781,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf15 = CatBoostClassifier(learning_rate = 0.03604,iterations = 9958, l2_leaf_reg = 10000,scale_pos_weight = 3.879,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf16 = CatBoostClassifier(learning_rate = 0.02844,iterations = 7360, l2_leaf_reg = 6280,scale_pos_weight = 0.6643,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf17 = CatBoostClassifier(learning_rate = 0.03624,iterations = 5995, l2_leaf_reg = 9994,scale_pos_weight = 0.8615,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf18 = CatBoostClassifier(learning_rate = 0.03098,iterations = 2002, l2_leaf_reg = 6313,scale_pos_weight = 5.341,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)

# catClf19 = CatBoostClassifier(learning_rate = 0.02764,iterations = 7314, l2_leaf_reg = 7162,scale_pos_weight = 5.475,eval_metric='AUC',
#                             silent = True,cat_features=catFeatures)


# In[ ]:


#scr_catClf1,trainPredsProbas1,catClfPreds1 = scoreOfModel(catClf1,x_train,y,3)
scr_catClf2,trainPredsProbas2,catClfPreds2 = scoreOfModel(catClf2,x_train,y,3)
scr_catClf3,trainPredsProbas3,catClfPreds3 = scoreOfModel(catClf3,x_train,y,3)
#scr_catClf4,trainPredsProbas4,catClfPreds4 = scoreOfModel(catClf4,x_train,y,3)
#scr_catClf5,trainPredsProbas5,catClfPreds5 = scoreOfModel(catClf5,x_train,y,3)
#scr_catClf6,trainPredsProbas6,catClfPreds6 = scoreOfModel(catClf6,x_train,y,3)
#scr_catClf7,trainPredsProbas7,catClfPreds7 = scoreOfModel(catClf7,x_train,y,3)
scr_catClf8,trainPredsProbas8,catClfPreds8 = scoreOfModel(catClf8,x_train,y,3)
#scr_catClf9,trainPredsProbas9,catClfPreds9 = scoreOfModel(catClf9,x_train,y,3)
#scr_catClf10,trainPredsProbas10,catClfPreds10 = scoreOfModel(catClf10,x_train,y,3)
#scr_catClf11,trainPredsProbas11,catClfPreds11 = scoreOfModel(catClf11,x_train,y,3)
# scr_catClf12,trainPredsProbas12,catClfPreds12 = scoreOfModel(catClf12,x_train,y,3)
# scr_catClf13,trainPredsProbas13,catClfPreds13 = scoreOfModel(catClf13,x_train,y,3)
# scr_catClf14,trainPredsProbas14,catClfPreds14 = scoreOfModel(catClf14,x_train,y,3)
# scr_catClf15,trainPredsProbas15,catClfPreds15 = scoreOfModel(catClf15,x_train,y,3)
# scr_catClf16,trainPredsProbas16,catClfPreds16 = scoreOfModel(catClf16,x_train,y,3)
# scr_catClf17,trainPredsProbas17,catClfPreds17 = scoreOfModel(catClf17,x_train,y,3)
# scr_catClf18,trainPredsProbas18,catClfPreds18 = scoreOfModel(catClf18,x_train,y,3)
# scr_catClf19,trainPredsProbas19,catClfPreds19 = scoreOfModel(catClf19,x_train,y,3)


# **Stacking the base models**

# **Defining a new Dataframe with the prediction values from our previous base models and then, we will use this DataFrame to train a meta-learner ( Logistic Regression in this kernel ) to get a boost in the prediction levels.**

# In[ ]:


stackedDF = pd.DataFrame({#'One' : trainPredsProbas1,
                          'Two' : trainPredsProbas2,'Three' : trainPredsProbas3, 
                          # 'Four' : trainPredsProbas4, 'Five' : trainPredsProbas5, 'Six' : trainPredsProbas6,
                          #'Seven':trainPredsProbas7,
                           'Eight':trainPredsProbas8,
                            #'Nine':trainPredsProbas9,
                          #'Ten':trainPredsProbas10,'Eleven':trainPredsProbas11
                            #,'Twelve':trainPredsProbas12,
                          #'Thirteen':trainPredsProbas13,'Fourteen':trainPredsProbas14,
                          #'Fifteen':trainPredsProbas15,'Sixteen':trainPredsProbas16,'Seventeen':trainPredsProbas17,
                          #'Eighteen':trainPredsProbas18,'Nineteen':trainPredsProbas19
                         })

stackedTest = pd.DataFrame({#'One' : catClfPreds1,
                            'Two' : catClfPreds2, 'Three' : catClfPreds3,
                            #'Four' : catClfPreds4,'Five' : catClfPreds5, 'Six' : catClfPreds6,
                           #'Seven':catClfPreds7,
                           'Eight':catClfPreds8,
                            #'Nine':catClfPreds9,
                         # 'Ten':catClfPreds10,'Eleven':catClfPreds11
                            #,'Twelve':catClfPreds12,
                          #'Thirteen':catClfPreds13,'Fourteen':catClfPreds14,
                          #'Fifteen':catClfPreds15,'Sixteen':catClfPreds16,'Seventeen':catClfPreds17,
                          #'Eighteen':catClfPreds18,'Nineteen':catClfPreds19
                         })


# In[ ]:


Stacker = LogisticRegression(C = 0.003728,solver='liblinear')


# Training the Stacker with **5 fold CV stratified**.

# In[ ]:


LRprobas = np.zeros(112392)
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(stackedDF,y))):
    X_train,X_val = stackedDF.loc[trn_idx,:],stackedDF.loc[val_idx,:]
    y_train,y_val = y[trn_idx],y[val_idx]
    
    Stacker.fit(X_train,y_train)
    
    #LRpreda = Stacker.predict_proba(X_val)
    #LRtrainprobas[val_idx] = LRpreda[:,1]
    #LRpredaT = LRpreda[:,1]
    #LRscore = LRscore + roc_auc_score(y_val,LRpredaT)
    LRpreds = Stacker.predict_proba(stackedTest)
    LRprobas = LRprobas + LRpreds[:,1]

LRprobas = LRprobas/5


# In[ ]:


sub = pd.read_csv('../input/sample_submission_24jSKY6.csv')

sub['loan_default'] = LRprobas


# In[ ]:


sub.to_csv('Submission.csv',index = False)


# **Now that we have reached the end of the kernel, I am assuming you liked the kernel, since you didnt close it mid-way.**
# 
# **If you did like it, please UPVOTE the kernel. That keeps me going !**
# 
# **Any suggestions and criticism are welcome.**
# 
# **Cheers !**
