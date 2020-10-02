#!/usr/bin/env python
# coding: utf-8

# In[516]:


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


# **Inputing files and seeing the first 5 rows**

# In[517]:


df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[518]:


df_train = pd.read_csv('../input/train.csv')
df_train.head()


# **Looking into structure and data types of the test.csv file**
# 
# Now all the operations and analysis has been done for this file only

# In[519]:


print(df_train.shape)
#it contains 614 rows along with 13 columns
print(df_train.dtypes)
#Observe the dtypes of Gender, Married, Self_Employed and Loan_Status. From first look it seems that they can be stored as Container types instead of strings
#Lets check that hypothesis by looking at the unique vallues


# In[520]:


print(df_train.loc[:,'Gender'].unique())
print(df_train.loc[:,'Married'].unique())
print(df_train.loc[:,'Education'].unique())
print(df_train.loc[:,'Self_Employed'].unique())
print(df_train.loc[:,'Loan_Status'].unique())


# In[521]:


#total memory usage
df_train.memory_usage(deep = True).sum()


# In[522]:


#Education indeed has only 2 values and hence can be effectively converted to Container type
df_train['Education'] = df_train.Education.astype('category', categories = ['Graduate','Not Graduate'], ordered = True)
df_train.head()


# In[523]:


#Since Educated has been converted the memory usage has reduced
df_train.memory_usage(deep = True).sum()


# In[524]:


#Finding all columns with null values
df_train.columns[df_train.isnull().any()].tolist()


# **Let us now find the mean median and modes of all the numeric columns and similar data for the string columns**

# In[525]:


df_train.describe()


# In[526]:


df_train.Gender.describe()
#most of the gender applicants are males


# In[527]:


df_train.Married.describe()


# In[528]:


df_train.Education.describe()


# In[529]:


df_train.Self_Employed.describe()


# In[530]:


df_train.Property_Area.describe()


# In[531]:


#import matplotlib to plot graphs
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')


# **Will now plot graphs to get various inferences**

# In[532]:


plt.scatter(df_train.ApplicantIncome, df_train.LoanAmount )
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()
#It can be seen from the plot that the number of points that the company gets most of the clients with lower income and lower loan amount.
#So the company should focus on this part rather than chasing big clients and spending time money and effort
#let us create a new column assigning people classes according to their income


# In[533]:


#Creating a new column with total income and treating nan values as 0. Inorder not to change the initial file, i copied the data into another series and then ued it for calculation
ap = df_train['ApplicantIncome'].copy()
coap = df_train['CoapplicantIncome'].copy()
df_train['Total_Income'] = ap.replace(np.nan, 0) + coap.replace(np.nan, 0)
df_train.head()


# In[534]:


df_train['Total_Income'].describe()


# In[535]:


#Creating the income Category according to Total Income
df_train['Income_Category'] = 'Lower Class'
df_train.loc[ (df_train.Total_Income > 4166) & (df_train.Total_Income <=5416.5) ,'Income_Category'] = 'Lower Middle Class'
df_train.loc[ (df_train.Total_Income > 5416.5) & (df_train.Total_Income <=7521.75) ,'Income_Category'] = 'Upper Middle Class'
df_train.loc[ (df_train.Total_Income > 7521.75) ,'Income_Category'] = 'Upper Class'
df_train.head()


# In[536]:


lower = df_train.loc[df_train.Income_Category == 'Lower Class', 'LoanAmount']
lowermid = df_train.loc[df_train.Income_Category == 'Lower Middle Class', 'LoanAmount']
uppermid = df_train.loc[df_train.Income_Category == 'Upper Middle Class', 'LoanAmount']
upper = df_train.loc[df_train.Income_Category == 'Upper Class', 'LoanAmount']
legend = ['Lower', 'Lower-Mid', 'Upper-Mid', 'Upper']
plt.hist([lower,lowermid,uppermid,upper],histtype='bar')
plt.xlabel('Loan Amount')
plt.ylabel('No of Loans Given')
plt.title('Histogram showing distribution of Loan Amount')
plt.legend(legend)
plt.show()
#It can be seen that the most applicants come from the Lower Middle and Lower classes. Also most of the loans sanctioned are of small amounts.
# I am unable to figure out the runtime warnings 


# In[537]:


ap = df_train['LoanAmount'].copy()
coap = df_train['Total_Income'].copy()
df_train['Ratio_LoantoIncome'] = ap.replace(np.nan, 0) / coap.replace(np.nan, 0)
df_train.head()


# In[538]:


plt.hist(df_train.Ratio_LoantoIncome, 30)
plt.xlabel('Ratio of Loan Amount to Income')
plt.ylabel('No of Loans')
plt.legend()
plt.show()


# In[539]:


#Finding success ratio of different classes
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Income_Category']).count().Loan_ID
total = df_train.groupby(['Income_Category']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Lower', 'Lower-Mid', 'Upper', 'Upper-Mid']
plt.bar(legend,success_rate, color = 'k',width = 0.2)
plt.xlabel('Class Of Loan Applicant')
plt.ylabel('Success Percentage of that Class')
plt.title('Succes Percentage of All Classes')
plt.show()


# In[540]:


#Finding success ratio of both genders
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Gender']).count().Loan_ID
total = df_train.groupby(['Gender']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Female','Male']
plt.bar(legend,success_rate, color = 'chartreuse',width = 0.07)
plt.xlabel('Gender Of Loan Applicant')
plt.ylabel('Success Percentage of that Gender')
plt.title('Succes Percentage of Both Genders')
plt.show()


# In[541]:


#Finding success ratio according to marital status
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Married']).count().Loan_ID
total = df_train.groupby(['Married']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Unmarried','Married']
plt.bar(legend,success_rate, color = 'r',width = 0.2)
plt.xlabel('Marital Status Of Loan Applicant')
plt.ylabel('Success Percentage of the Marital Status')
plt.title('Succes Percentage according to Marital Status')
plt.show()


# In[542]:


#Finding success ratio according to Education
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Education']).count().Loan_ID
total = df_train.groupby(['Education']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Graduate' ,'Not Graduate']
plt.bar(legend,success_rate, color = 'm',width = 0.2)
plt.xlabel('Education Of Loan Applicant')
plt.ylabel('Success Percentage of that Education Status')
plt.title('Succes Percentage according to Education')
plt.show()


# In[543]:


payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Credit_History']).count().Loan_ID
total = df_train.groupby(['Credit_History']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Poor' ,'Good']
plt.bar(legend,success_rate, color = 'burlywood',width = 0.2)
plt.xlabel('Credit History Of Loan Applicant')
plt.ylabel('Success Percentage according to Credit History')
plt.title('Succes Percentage according to Credit History')
plt.show()


# In[544]:


payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Property_Area']).count().Loan_ID
total = df_train.groupby(['Property_Area']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Rural' ,'Semi-Urban','Urban']
plt.bar(legend,success_rate, color = (0.9,0.2,0.6),width = 0.2)
plt.xlabel('Credit History Of Loan Applicant')
plt.ylabel('Success Percentage according to Credit History')
plt.title('Succes Percentage according to Credit History')
plt.show()


# In[545]:


#It can be seen from above graphs that the Educated, Married and people having good credit history are likely to pay their loans as compared to others


# **We will now move on to outlier detection**

# In[546]:


#Function for Finding and replacing outliers with mean
def change_outlier(df_in, col_name):
    q1 = df_in[col_name].describe()['25%']
    q3 = df_in[col_name].describe()['75%']
    m = df_in[col_name].describe()['mean']
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-(1.5*iqr)
    fence_high = q3+(1.5*iqr)
    df_in.loc[(df_in[col_name] <= fence_low) | (df_in[col_name] >= fence_high),col_name] = m
    return df_in


# In[547]:


#Outliers not changed for Income as that will remove the Upper Class entirely. The Upper Class is not incorrect data so the need to change them is not there
change_outlier(df_train,'LoanAmount')
change_outlier(df_train,'Loan_Amount_Term')
df_train.describe()


# **Will now deal with Missing values in various columns by replacing them with the most frequent value**

# In[548]:


#Find the most frequent value and set the rest to it
m = df_train.Married.describe()['top']
df_train.loc[(df_train.Married != 'Yes') & (df_train.Married !='No'),'Married'] = m


# In[549]:


m = df_train.Gender.describe()['top']
df_train.loc[(df_train.Gender != 'Male') & (df_train.Gender !='Female'),'Gender'] = m

m = df_train.Dependents.describe()['top']
df_train.loc[(df_train.Dependents != '0') & (df_train.Dependents !='1'),'Dependents'] = m

m = df_train.Self_Employed.describe()['top']
df_train.loc[(df_train.Self_Employed != 'Yes') & (df_train.Self_Employed !='No'),'Self_Employed'] = m


# In[550]:


#replacing credit history nan with median and not mean as it can only be zero or one
m = df_train.Credit_History.describe()['50%']
df_train.loc[(df_train.Credit_History != 1) & (df_train.Credit_History !=0),'Credit_History'] = m

#LoanAmount and LoanAmountTerm replaced by mean
m = df_train.LoanAmount.describe()['mean']
df_train.loc[(df_train.LoanAmount.isnull()),'LoanAmount'] = m
m = df_train.Loan_Amount_Term.describe()['mean']
df_train.loc[(df_train.Loan_Amount_Term.isnull()),'Loan_Amount_Term'] = m


# In[551]:


#check whether any any elements left
df_train.columns[df_train.isnull().any()].tolist()


# **Make all the graphs again with the new data and note changes**

# In[552]:


plt.scatter(df_train.ApplicantIncome, df_train.LoanAmount )
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.legend()
plt.show()
#In camparison to the previous plot the data is even more scewed towards the lower income range


# In[553]:


lower = df_train.loc[df_train.Income_Category == 'Lower Class', 'LoanAmount']
lowermid = df_train.loc[df_train.Income_Category == 'Lower Middle Class', 'LoanAmount']
uppermid = df_train.loc[df_train.Income_Category == 'Upper Middle Class', 'LoanAmount']
upper = df_train.loc[df_train.Income_Category == 'Upper Class', 'LoanAmount']
legend = ['Lower', 'Lower-Mid', 'Upper-Mid', 'Upper']
plt.hist([lower,lowermid,uppermid,upper],histtype='bar')
plt.xlabel('Loan Amount')
plt.ylabel('No of Loans Given')
plt.title('Histogram showing distribution of Loan Amount')
plt.legend(legend)
plt.show()
#It contrast to the previous graph it is seen that there are significant applicants from Upper class as well
#The runtime warning here is gone again dont know the reason


# In[554]:


plt.hist(df_train.Ratio_LoantoIncome, 30)
plt.xlabel('Ratio of Loan Amount to Income')
plt.ylabel('No of Loans')
plt.legend()
plt.show()
#This plot remains similar with most of the ratios lying between 0.01-0.03


# **Finding success ratios again with different parameters**

# In[558]:


#Finding success ratio of different classes
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Income_Category']).count().Loan_ID
total = df_train.groupby(['Income_Category']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Lower', 'Lower-Mid', 'Upper', 'Upper-Mid']
plt.bar(legend,success_rate, color = 'k',width = 0.2)
plt.xlabel('Class Of Loan Applicant')
plt.ylabel('Success Percentage of that Class')
plt.title('Succes Percentage of All Classes')
plt.show()


# In[559]:


#Finding success ratio of both genders
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Gender']).count().Loan_ID
total = df_train.groupby(['Gender']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Female','Male']
plt.bar(legend,success_rate, color = 'chartreuse',width = 0.07)
plt.xlabel('Gender Of Loan Applicant')
plt.ylabel('Success Percentage of that Gender')
plt.title('Succes Percentage of Both Genders')
plt.show()


# In[560]:


#Finding success ratio according to marital status
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Married']).count().Loan_ID
total = df_train.groupby(['Married']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Unmarried','Married']
plt.bar(legend,success_rate, color = 'r',width = 0.2)
plt.xlabel('Marital Status Of Loan Applicant')
plt.ylabel('Success Percentage of the Marital Status')
plt.title('Succes Percentage according to Marital Status')
plt.show()


# In[561]:


#Finding success ratio according to Education
payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Education']).count().Loan_ID
total = df_train.groupby(['Education']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Graduate' ,'Not Graduate']
plt.bar(legend,success_rate, color = 'm',width = 0.2)
plt.xlabel('Education Of Loan Applicant')
plt.ylabel('Success Percentage of that Education Status')
plt.title('Succes Percentage according to Education')
plt.show()


# In[562]:


payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Credit_History']).count().Loan_ID
total = df_train.groupby(['Credit_History']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Poor' ,'Good']
plt.bar(legend,success_rate, color = 'burlywood',width = 0.2)
plt.xlabel('Credit History Of Loan Applicant')
plt.ylabel('Success Percentage according to Credit History')
plt.title('Succes Percentage according to Credit History')
plt.show()


# In[563]:


payed = df_train.loc[df_train.Loan_Status == 'Y',:].groupby(['Property_Area']).count().Loan_ID
total = df_train.groupby(['Property_Area']).count().Loan_ID

success_rate = (payed/total) * 100
legend = ['Rural' ,'Semi-Urban','Urban']
plt.bar(legend,success_rate, color = (0.9,0.2,0.6),width = 0.2)
plt.xlabel('Credit History Of Loan Applicant')
plt.ylabel('Success Percentage according to Credit History')
plt.title('Succes Percentage according to Credit History')
plt.show()


# Most of the plots did not show much variance. The analysis remains similar that Educated, Married and Good Credit History are desirable traits in a Loan Applicant as they are more likely to clear their loans
