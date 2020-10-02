#!/usr/bin/env python
# coding: utf-8

# 
# 

#              

# 

# 

# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


df = pd.read_csv('../input/train_u6lujuX_CVtuZ9i (1).csv')
#df2 = pd.read_csv('D:/datacamp/files/loan predication/test.csv',index_col=['Loan_ID'])
#df=pd.merge(df1,df2)


# In[7]:


# printing first five rows
df.head()


# In[ ]:





# In[4]:


df.shape


# In[151]:


df.isnull().sum()


# In[163]:


list = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']


# In[160]:


def category_var(dataframe,ls):
    for var in ls:
        dataframe[var] =  dataframe[var].astype('category')
    


# In[164]:


category_var(df,list)


# In[165]:


df.info()


# In[166]:


df.isnull().sum()


# In[11]:


def fill_missing(df):
    for  col in df:
        #print(df[col].isnull().sum())
        if df[col].dtypes=='category':
            print('hello')


# In[172]:


# filling missing values 
category =['Gender','Married','Education','Self_Employed','Property_Area','Dependents','Credit_History']

for var in category:
    df[var].fillna(method='ffill',inplace=True)
    


# In[13]:


# filling missing values
def fill_missing(frame,list):
    for var in list:
        if(frame[var].dtypes=='category'):
            frame[var].fillna(method='ffill')
        elif((frame[var]=='int64') or (frame[var]=='float64')):
            frame[var].fillna(np.mean,inplace=True)


# In[173]:


#filling missing values
numeric = ['LoanAmount','Loan_Amount_Term']

for var in numeric:
    mean = np.around(np.mean(df[var]),decimals=0)
    df[var].fillna(mean,inplace=True)


# In[174]:


df.isnull().sum()


# In[16]:


df.boxplot(column='ApplicantIncome',by='Education')


# In[17]:


df.columns


# In[ ]:





# In[18]:


df.describe()


# In[176]:


#peoples who are not graduated,unmarried  and got a loan a
df[(df['Education']!='Graduate') & (df['Married']=='No') &  (df['Loan_Status']=='Y')]['Self_Employed'].value_counts()
# conclusion-1 -- 82% peoples which are not self emplyoyed and also not graduated and unmarried


# In[177]:


df.head()


# In[ ]:





# In[ ]:





# In[22]:


data_sorted = df.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)
data_sorted[['ApplicantIncome','CoapplicantIncome']].head(10)


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df.boxplot(column="ApplicantIncome",by="Loan_Status")


# In[24]:


df.groupby(['Loan_Status','Gender']).sum()
#conclusion3 -- males have high crdit history and more income then females 


# In[25]:


df.head()


# In[26]:


df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})


# In[181]:


with plt.style.context(('dark_background')):
    pd.crosstab(df['Property_Area'],df['Loan_Status']).plot.bar()
#pd.crosstab(df['Property_Area'],df['Loan_Status'],margins=True).apply(convert_percentage,axis=1)


# HERE 0 DENOTES LOAN-STATUS=='NO'AND 1 DENOTES LOAN-STATUS='YES'
#conclusion--in rural and urban areas loan denied have no signifiacntly differnce 
#and semiurban areas have more peoples of  getting loan


# In[31]:


df.head()


# In[120]:


income=  df[['ApplicantIncome','CoapplicantIncome','Loan_Status']]
a =income.loc[(income['ApplicantIncome']>=6277) & (income['ApplicantIncome']<=6500)]
#print(a)
plt.figure()
pd.crosstab(a['ApplicantIncome'],a['Loan_Status']).plot.bar()
#a.plot(['ApplicantIncome','Loan_Status'])

pd.crosstab(a['CoapplicantIncome'],a['Loan_Status']).plot.bar()

###############################
##########      min applicant income to get loan is 6000-7000
######################## and there are no much dependency on coaaplicantincome


# In[ ]:





# In[196]:



plt.style.use('ggplot')
plt.figure(figsize=(30,5))
pd.crosstab(df['Education'],df['Loan_Status']).plot.bar()
###################################
#####################conclusion-4 ------- graduate have higher chance of gettinpd.
plt.style.use('fivethirtyeight')

pd.crosstab(df['Credit_History'],df['Loan_Status']).plot.bar()
#####################################################################
####################################################################
#########  THERE IS SIGNIFIACNT DIFERRNCE  ON BASIS OF CREDIT HISTORY
#THE PEOPLES WHICH HAVE CREDIT HISTORY THA HAVE HIGH CHANCES OF GETTING LOAN


# In[197]:





# In[295]:


plt.style.use('classic')
plt.rc('figure', figsize=(20,5))
ax =pd.crosstab(df['Married'],df['Loan_Status']).plot.bar()



ax.text(0, 330, 'married and unmarried  loan status',fontsize=30,
        bbox={'facecolor':'red','color':'yellow', 'alpha':0.5, 'pad':5})

ax.set_xlabel('married',fontsize=30,color='r',bbox={'facecolor':'blue'})

plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14, rotation=90)
#############################################################################################################
#conclusion-6 -------  married peoples have high probablity of getting loan then unmarried peoples
########################################################################################################



# In[299]:


pd.crosstab(df['Dependents'],df['Loan_Status']).plot.bar()
#######################
# peoples which  have 0 dependents means 0 childerns or siblings have high probbality of getting loan


# # conclusion ---------
# ## 1- the peoples  which have  credit history that  chances are more to getting loan
# ## 2- peoples which live in semiurban areas that  have more chances to get loan
# ###  relationship--(semiurban>urban>rural)
# ## 3-minimum applicant income to get loan in between 6000-7000 and there is no dependency on coapplicantincome
# ## 4-graduated peoples have more chances to get loan
# ### realtionship --(graduated>non-graduated)
# ## 5-married peoples have more chances to get loan
# ## 6 - peoples  which have 0 childrens or siblings that have more chances to get loan

# # final -conclusion
# ## loan -approval-probablity
# ### 1- credit history>>no_credit_history
# ### 2-no_dependents>>depdents
# ### 3-graduated>non_graduated
# ### 4-married>unmarried
# ### 5-applicant_income>=6000
# ### 6 - semiurban>>urban>rural

# In[ ]:




