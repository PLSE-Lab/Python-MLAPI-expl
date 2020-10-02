#!/usr/bin/env python
# coding: utf-8

# In[294]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt 
import seaborn as sb
import random 
import time
from  sklearn.model_selection  import StratifiedKFold as kfold
from  sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# In[295]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
Full_Data=pd.concat([train,test])
Full_Data.shape


# In[296]:


#Generalised Function which can give the Percentage of missing values present in DataSet.
def Train_missing_values(training_dataset):
    Missing_Data_Percent=pd.DataFrame(training_dataset.isna().sum())
    Missing_Data_Percent.reset_index(inplace=True)
    Missing_Data_Percent.columns=['Feild_Name','Missing_value_count']
    Missing_Data_Percent['Percent_missing_values']=Missing_Data_Percent['Missing_value_count'].                                                apply(lambda Missing_value_count:(Missing_value_count/len(training_dataset))*100)
    return Missing_Data_Percent.sort_values(['Percent_missing_values'],ascending=False)


Train_missing_values(Full_Data)


# ### Observations :
# * Loan status having almost 37 % missing values but that missing values are due to test set. So Ideally we should not worry about them 
# * We have Very Small set of missing values as well as Dataset , So we will Prefer to go for imputation of Missing values instead of removeing them set 

# In[297]:


Full_Data.Credit_History.unique()


# ### Fixing the Imputation value for Credit History
# *  We have almost 10 % data where Credit history is Blank or Null.So we have to impute that. So First We will Check the Target  distributions for those Missing NaN values .This Will help us to impute the Missing values for that particular 
# * by Seeing below chart , We can Conclude that more that 60 percent of people who have a Blanks Credit History Got the loan from the Banks .
# * So we will imput Value as '1' For all the records which are having credit history as NaN

# In[298]:


Credit_History_Analysis=Full_Data[Full_Data['Credit_History'].isnull()==True]
a=dict(Credit_History_Analysis['Loan_Status'].value_counts())
plt.bar(range(len(a)), list(a.values()), align='center')
plt.xticks(range(len(a)), list(a.keys()))


# * by Seeing above chart , We can Conclude that more that 60 percent of people who have a Blanks Credit History Got the loan from the Banks.
# * So we will imput Value as '1' For all the records which are having credit history as NaN

# In[299]:


a=dict(Full_Data['Self_Employed'].value_counts())
plt.bar(range(len(a)), list(a.values()), align='center')
plt.xticks(range(len(a)), list(a.keys()))


# ### Imputation Final Values 
# 
# * Credit History NULL values should  be replace by 1(i.e that is Credit history is Present). Reason is 60% of people who dont have credit history they still got loan or eligible for loan
# * For Self Employment Column , We will Go for Mode Imputation where most of Popultion who are applying for the Loan Not Self Employed. 

# In[300]:


sb.boxplot(x=Full_Data["Loan_Amount_Term"],color=".25")


# In[301]:


#Normalise the Loan Term Duration nby dividuing it 12
Full_Data['Loan_term_in_year']=Full_Data['Loan_Amount_Term']/12
Full_Data.drop('Loan_Amount_Term',axis=1,inplace=True)
Full_Data.dropna(subset=['Married'],inplace=True)


# In[302]:


Full_Data['Gender'].describe()


# In[303]:


#Imputation for NA values
#The Imputation done below are based on Some analysis which is not Shown Above
Full_Data.fillna({'Self_Employed':'No','Credit_History':1,'Loan_term_in_year':30,'Dependents':0 ,                   'LoanAmount':Full_Data['LoanAmount'].mean() ,'Gender':'Male'} ,inplace=True)


# In[304]:


sb.pairplot(Full_Data,hue='Loan_Status',markers='+')
plt.show()


# In[305]:


Full_Data.columns


# In[306]:


Full_Data.head()


# In[307]:


Full_Data.describe()


# In[308]:


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


# In[309]:


#Get One Hot Encoding For Data
Full_Data=one_hot(Full_Data,['Education','Gender','Married','Self_Employed','Property_Area'])


# In[310]:


#Normalisation of Data
Full_Data['LoanAmount_log']=np.log(Full_Data['LoanAmount']+1)
Full_Data['ApplicantIncome_log']=np.log(Full_Data['ApplicantIncome']+1)
Full_Data['CoapplicantIncome_log']=np.log(Full_Data['CoapplicantIncome']+1)
Full_Data['Loan_term_in_year_log']=np.log(Full_Data['Loan_term_in_year']+1)


# In[311]:


#Removed Columns 
Full_Data.drop(['Education','Gender','Married','Self_Employed','Property_Area','LoanAmount',               'ApplicantIncome','CoapplicantIncome', 'Loan_term_in_year'],axis=1,inplace=True)
Full_Data.set_index('Loan_ID',inplace=True)


# In[312]:


#Label Encoding
Full_Data['Dependents'].replace({'0':0,0:0,'1':1,'2':2,'3+':3},inplace=True)
Full_Data['Dependents'].unique()


# In[313]:


Final_Test_Set=Full_Data[Full_Data.Loan_Status.isnull()]
Final_Train_Set=Full_Data[Full_Data.Loan_Status.isnull()==False]


# In[314]:


Final_Train_Set.columns


# In[315]:


x_train=Final_Train_Set[['ApplicantIncome_log', 'CoapplicantIncome_log', 'Credit_History', 'Dependents',        'LoanAmount_log', 'Loan_term_in_year_log', 'Education_Graduate',       'Education_Not Graduate', 'Gender_Female', 'Gender_Male', 'Married_No',       'Married_Yes', 'Self_Employed_No', 'Self_Employed_Yes',       'Property_Area_Rural', 'Property_Area_Semiurban',       'Property_Area_Urban']]
y_train=Final_Train_Set['Loan_Status']


# In[316]:


x_train.describe()


# In[317]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
LogisticRegression=LogisticRegressionCV(cv=10,random_state=1,multi_class='auto')
#cross_val_score(model,x_train, y_train,cv=10)
Final_model=LogisticRegression.fit(x_train, y_train)


# In[318]:


X_test=Final_Test_Set[['ApplicantIncome_log', 'CoapplicantIncome_log', 'Credit_History', 'Dependents',        'LoanAmount_log', 'Loan_term_in_year_log', 'Education_Graduate',       'Education_Not Graduate', 'Gender_Female', 'Gender_Male', 'Married_No',       'Married_Yes', 'Self_Employed_No', 'Self_Employed_Yes',       'Property_Area_Rural', 'Property_Area_Semiurban',       'Property_Area_Urban']]


# In[319]:


X_test['y_test_pred']=Final_model.predict(X_test)


# In[320]:


X_test['y_test_pred'].value_counts()


# In[321]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = X_test

# create a link to download the dataframe
create_download_link(df)

