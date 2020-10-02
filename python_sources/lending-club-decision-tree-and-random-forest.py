#!/usr/bin/env python
# coding: utf-8

# # **Lending club Data Set**
# 

# #### In this Data set , I am using Decision Tree and Random Decision Tree to predict weather the payment will be paid in full , I also analyzed some other factor as fico (credit Score), interes rate , etc.
# * I used to work for **Capital one USA**, so I was familiar with this factors , however there are many other that pick my curiosity
# 1. It is interesting to notice that t**he interest rate varies depending on the kind of need** (see column "Purpose), this might be because in U**SA Credit card rate are less risky-lower than loan for small business**, this because CC cash generate more income that loans daily
# 2. I found some **Outliers** according to the quarter rules, actually less than 10 so I evaluated the model **with/without** them 
# 3. I would asume that **people with better fisco will always pay in full at least the installment** (according to my experience in Capital one customer always paid in full beacuse they were worry about the Credit Score) or recived better interest rate , but no in this Company
# 4. There is no a Factor that can lead drastically the company to predict if they willl recieved the payment in full . 
# 5. Other banks consider the Credit Score inquire as risk factor - If other bank has checked it will reduce your Credit core because it means you applied for more loan and the chances of getting their payment is less
# 6.Rndom Forest has a excellent performance over Decision Tree

# # **OSEMN Methodology **

# In[ ]:


#basic library 
import pandas as pd
import numpy as np


#visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 
import cufflinks as cf 
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot #iplot for interactive graphs
init_notebook_mode(connected=True)

cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')


# ## **Obtain data**

# In[ ]:


df=pd.read_csv('../input/loan-data/loan_data.csv')
df.head(5)


# # **Some columns Information**
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# In[ ]:


df.describe().head(4)


# ## **Scrub (filtering, extracting , replacing , handle missing values)** 
# 1. Looking for Missing Values 
# 2. "0" Values , etc.

# In[ ]:


NAN_value = (df.isnull().sum() / len(df)) * 100
Missing = NAN_value[NAN_value==0].index.sort_values(ascending=False)
Missing_data = pd.DataFrame({'Missing Ratio' :NAN_value})
Missing_data.head(20)


# > * **There is no** Missing Values`

# ### * Explore (Understand Data , Create Visualization, Deriving Statistic)

# > Finding Coorelation between variables 
# 1. A start by having a General view to better understand the data
# 2. Identify any issue
# 3. Find any relationshp between variables 
# 4. Verify some assumption 

# In[ ]:


f,ax = plt.subplots(figsize = (15,10))
sns.heatmap(df.corr(),cmap='viridis',annot=True, ax=ax)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
print("Correlacion between variables")


# 1. Exploring Interest Rate

# In[ ]:


f,(ax1,ax2,ax3)= plt.subplots(1,3,figsize=(25,10))
sns.distplot(df['int.rate'], bins= 30,ax=ax1)
sns.boxplot(data =df, x ='credit.policy', y= df['int.rate'],ax=ax2).legend().set_visible(False)
sns.boxplot(data = df['int.rate'], ax=ax3)
print("Interest Rate Distribution, Credit Policy range based on the Credit policy , General Interest rate")


# In[ ]:


#Finding Relationship between fisco ~ interest rate ~ installment 
sns.jointplot(y='int.rate', x='fico',data= df)
print("Interest rate - FICO ")


# # Interpretation
# 1. The most common interest rate is arrounf 0.75 - 0.125 respectively
# 2. The frist boxplot showed that People who is eligible after the underwriting Process "1" , have a better interest rate (I asumme this might be based on previous application, and those who has 0 go to a second review but with higher interest rate 
# 3. The median of interest rate es 0.12 , it means the average interest rate for customer is 0.12
# 4. As expected if your fico (credit Score) it is high your interest rate will be low and vice versa.

# 2. Exploring Interest Rate and Needs

# In[ ]:


f,(ax4,ax5,ax6)= plt.subplots(1,3,figsize=(30,10))

sns.countplot(x='purpose',data=df, 
              hue='not.fully.paid',palette='Set1',ax=ax4)

sns.boxplot(x='purpose', y='int.rate',
            data= df, hue='not.fully.paid',palette='Set2',ax=ax5).legend().set_visible(False)

sns.boxplot(x='purpose', y='int.rate',
            data= df,ax=ax6)


print('Data1 : Reason of the Loan.    Data2: Interest based on the Reason')


# # Interpretation
# 1. As I assumed Debt consolidation knowing as "balance Transfer" and Credita card are in the top 3 Reason 
# 2. Interest rate accoding to the Needs
# 3. Tis is intersting to notice , in terms of risk " small business loan" seems to be more risky, perhaps this could be a reason why USA has no that much start up company as before, instead major purchases (House, car, etc) interest rate is the lowest (" I could be because is the one that everyone is looking for or simply incentive the economy )

# 3. Exploring FICO and Full Payment 

# In[ ]:


sns.lmplot(data=df,palette='Set1',x='fico',y='int.rate', hue='credit.policy',col='not.fully.paid')


# # Interpretation
# 1. People with fico > 675 are more likely to pay in full (but cosider the amount of people who has fico > 675 who are not paying in full) 

# In[ ]:


# Load the example mpg dataset
mpg = sns.load_dataset("mpg")

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="fico", y="int.rate",hue= 'not.fully.paid', sizes=(30, 200), alpha=.4, size ='installment',
         height=6, data=df )


# # Interpretation
# 1. Most of the them do not make full payment 
# 2. the moajority of them make payment lower than 800 
# 3. It is interesting to notice that people with lower fico (750) are paying in full , perhaps they are working on their Credit score 

# In[ ]:


plt.figure(figsize=(11,6))
fico_0 = df[df['not.fully.paid']==0]['fico'].hist(alpha=0.4,color='red',bins=30,label='not.fully.paid=1')
fico_1 = df[df['not.fully.paid']==1]['fico'].hist(alpha=0.4,color='blue',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(11,6))
fico_0 = df[df['not.fully.paid']==0]['fico'].hist(alpha=0.4,color='red',bins=30,label='not.fully.paid=1')
fico_1 = df[df['not.fully.paid']==1]['fico'].hist(alpha=0.4,color='blue',bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# ### * Modeling the Data 
# 1. Creating Categorical Data

# In[ ]:


# Attention somo outliers
out_1 = df[(df['fico']>750) & (df['int.rate']>0.175)].index.to_list()
out_2 = df[(df['fico']<700) & (df['int.rate']<0.075)].index.to_list()
Outliers = out_1 + out_2
df.iloc[Outliers]
loan_1 = df.drop(Outliers)


# 1. Following the Q rules I found some parameters, I will drop them and run the same model and make sure how it affects the final result 

#  # "DECISION TREE"

# ## Decision tree with outlier

# In[ ]:


#Purpose to Categorical Data
final_loan  = pd.get_dummies(loan_1,drop_first = True) # without outliers
final_data = pd.get_dummies(df,drop_first = True)

# Applying Machine Learning 
from sklearn.model_selection import train_test_split
X=final_data.drop('not.fully.paid', axis=1)
y= final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)

# Measuring Accurancy 
from sklearn.metrics import classification_report,confusion_matrix
dtree_score=classification_report(y_test,y_pred)
print(classification_report(y_test,y_pred))


# ## Decision tree without outlier

# In[ ]:


X=final_loan.drop('not.fully.paid', axis=1)
y= final_loan['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dtree_loan = DecisionTreeClassifier()
dtree_loan.fit(X_train,y_train)
y_pred_loan = dtree_loan.predict(X_test)
dtree_outlier = classification_report(y_test,y_pred_loan)
print(classification_report(y_test,y_pred_loan))


#  # "Random Forest"
#  1. With oulier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
X=final_data.drop('not.fully.paid', axis=1)
y= final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
RFC = RandomForestClassifier(n_estimators = 50)
RFC.fit(X_train,y_train)
y_pred_RFC = (y_test)
RFC_report=classification_report(y_test,y_pred_RFC)
print(classification_report(y_test,y_pred_RFC))


# 2. without Outliers

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
X=final_data.drop('not.fully.paid', axis=1)
y= final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
RFC = RandomForestClassifier(n_estimators = 300)
RFC.fit(X_train,y_train)
y_pred_RFC = (y_test)
RFC_outliers = classification_report(y_test,y_pred_RFC)
print(classification_report(y_test,y_pred_RFC))


# In[ ]:


#Error_Rate = []
#for i in range (1,310):
    
 #   RFC_Error = RandomForestClassifier(n_estimators = i)
 #  RFC_Error.fit(X_train,y_train)
 #  pred_i = RFC_Error.predict(X_test)
 # Error_Rate.append(np.mean(pred_i != y_test))
    
#plt.figure(figsize=(10,6))
#plt.plot(range(1,310), Error_Rate , color = 'blue', linestyle = 'dashed', marker = 'o')


# >### I dont know what is the best number of n_estimator so I would choose the one who has less Error_rate
# >### please if you know the answer let me know in the comments 
# >### Any other advance technique or approach of this Data "please let me know and post the link" 

# In[ ]:




