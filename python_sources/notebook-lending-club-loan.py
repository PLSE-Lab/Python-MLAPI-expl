#!/usr/bin/env python
# coding: utf-8

# # Checking the Available Dataset

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Loading Modules

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# # Loading Dataset

# In[ ]:


# Display all columns(features)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

df_loan = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv')


# # General Data Exploration

# The dataset consist of 2,260,668 observations and 145 features (including the target feature *"loan_status"*). Most of those features contain too many missing values.

# In[ ]:


df_loan.shape


# In[ ]:


df_loan.head()


# In[ ]:


# descriptive statistics of numerical features
df_loan.describe()


# In[ ]:


# descriptive statistics of categorical (string) features
df_loan.describe(include=['O'])


# In[ ]:


df_loan.isnull().sum()


# # Exploratory Data Analysis

# In[ ]:


total_data=len(df_loan)


# * ##  Loan Status

# > Mostly, the loan has status fuly paid by 46%, current 41%, charged off 12%, then followed by other status. While default status portion just about 0.001% or happened 1 out of 1000 cases.

# ### Loan Status Distribution

# In[ ]:


plt.figure(figsize=(11,6))
sns.set(style='darkgrid')
g=sns.countplot(x="loan_status", data=df_loan ,order = df_loan['loan_status'].value_counts().index ,color='r')
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Loan Status", fontsize=15)
g.set_ylabel("Count", fontsize=15)
g.set_title("Loan Status Distribution", fontsize=20)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.3f}%'.format(height/total_data*100),ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# * ## State

# > Typically, all states loan dominated by status fully paid (red), current (orange), and charged off (blue). Meanwhile, default just occur in few cases. <br> IA is the state with largest portion of fully paid status by 85.7%. Also the state with largest percentage of charged off beside AL by 14%. While ME is the state with largest portion of default by 0.02%.

# In[ ]:


status=['Current','Fully Paid','Charged Off','Default']
df=df_loan[df_loan['loan_status'].isin(status)]


# In[ ]:


plt.figure(figsize=(20,10))
tab=pd.crosstab(df['addr_state'], df['loan_status'],normalize='index') * 100
tab.plot(kind='bar',legend=None,stacked=True)
plt.xlabel('State',fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.ylabel('Share of Loan Status (%)',fontsize=10)
plt.title('Share of Loan Status by State', fontsize=12)
#plt.legend(loc='upper right')


# In[ ]:


tab.style.background_gradient(cmap = sns.light_palette("green", as_cmap=True))


# * ## Verification Status

# > Verification status does not strongly correlated referenced by fully paid (red), current (orange), and charged off (blue) status. <br> Altough, loan with verified mark has slightly better fully paid chance rate but it is not significant compared to other category of verification status.

# In[ ]:


tab=pd.crosstab(df['verification_status'], df['loan_status'],normalize='index') * 100
tab.plot(kind='bar',legend=None,stacked=True)
plt.xlabel('Verification Status',fontsize=10)
plt.ylabel('Share of Loan Status (%)',fontsize=10)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.title('Share of Loan Status by Verification Status',fontsize=12)
#plt.legend(loc='upper right')


# In[ ]:


tab.style.background_gradient(cmap = sns.light_palette("orange", as_cmap=True))


# * ## Purpose

# > The borrowers mostly request the loan for debt consolidation by more than half portion (56.53%). The second and third largest are credit card and home improvement respectively by much lesser percentage. <br>
# Loan with wedding purpose has the chance rate of fully paid. Meanwhile many education purposes loan has the great record chance of does not meet the credit policy status. Loan with small business purpose has the biggest chance rate of charged off status.

# ### Purpose Distribution

# In[ ]:


plt.figure(figsize=(11,6))

g = sns.countplot(x='purpose', data=df_loan,order = df_loan['purpose'].value_counts().index,color='g')
g.set_title("Borrower Purposes for Loan Credit", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Purpose", fontsize=15)
g.set_ylabel('Count', fontsize=15)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=11) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# ### Purpose by Loan Status

# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan['purpose'], df_loan['loan_status'],normalize='index') * 100,2).style.background_gradient(cmap = cm)


# > * ## Loan Amount

# > The highest frequency of loan amount requested by borrowes are about USD 10,000. <br> Generally, the loan borrower that does not meet the credit policy are  request for lower loan amount than with other loan status.

# ### Loan Amount Distribution

# In[ ]:


plt.figure(figsize=(11,6))
sns.set(style='dark')
g = sns.distplot(df_loan["loan_amnt"], color='c')
g.set_xlabel("Loan Amount", fontsize=15)
g.set_ylabel("Frequency", fontsize=15)
g.set_title("Loan Amount Distribution", fontsize=20)
plt.show()


# ### Loan Amount by Loan Status

# In[ ]:


plt.figure(figsize=(11,6))

g2 = sns.boxplot(x="loan_status", y="loan_amnt", data=df_loan, order = df_loan['loan_status'].value_counts().index,color='r')
g2.set_xticklabels(g2.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g2.set_xlabel("Loan Status", fontsize=15)
g2.set_ylabel("Loan Amount Distribution", fontsize=15)
g2.set_title("Loan Amount by Loan Status", fontsize=20)

plt.show()


# * ## Year Issued

# > The count of total loan significantly increase between 2012 and 2015. Afterward, the count keep increasing with constant rise until recent year.

# ### Total Loan by Year Issued

# In[ ]:


df_loan['issue_month'], df_loan['issue_year'] = df_loan['issue_d'].str.split('-', 1).str

plt.figure(figsize=(11,6))
sns.set(style='darkgrid')

g = sns.countplot(x='issue_year', data=df_loan, color='m')
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'center'},rotation=45)
g.set_xlabel("Year", fontsize=15)
g.set_ylabel("Total Loan", fontsize=15)
g.set_title("Total Loan by Year Issued", fontsize=20)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# * ## Application Type

# > Almost all of the loans are from individual application by 94.7% while the rest with small portion is from joint application (5.3%). <br>
# Loan application which does not meet credit policy come from individual application.<br>
# The distribution of loan amount of joint application nearly the same for all loan status and also similar to from individual app. <br>
# Those patter nearly the same with categorized by purpose except the wedding and educational purpose only come from individual app.

# ### Application Type Distribution

# In[ ]:


plt.figure(figsize=(11,6))

g = sns.countplot(x='application_type', data=df_loan,order = df_loan['application_type'].value_counts().index,color='gray')
g.set_title("Aplication Type Distribution", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Application Type", fontsize=15)
g.set_ylabel('Count', fontsize=15)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=11) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# ### Loan Amount & Loan Status by Application Type

# In[ ]:


# Now will start exploring the Purpose variable
plt.figure(figsize=(11,6))

g = sns.violinplot(x="loan_status",y="loan_amnt",data=df_loan,hue="application_type", order = df_loan['loan_status'].value_counts().index,split=True, palette='RdBu')
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_title("Loan Status - Loan Amount Distribution by Application Type", fontsize=20)
g.set_xlabel("Loan Status", fontsize=15)
g.set_ylabel("Loan Amount Distribution", fontsize=15)

plt.show()


# ### Loan Amount & Purpose by Application Type

# In[ ]:


# Now will start exploring the Purpose variable
plt.figure(figsize=(11,6))

g = sns.violinplot(x="purpose",y="loan_amnt",data=df_loan,hue="application_type",order = df_loan['purpose'].value_counts().index ,split=True, palette='hls')
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_title("Purpose - Loan Amount Distribution by Application Type", fontsize=20)
g.set_xlabel("Purpose", fontsize=15)
g.set_ylabel("Loan Amount Distribution", fontsize=15)

plt.show()


# * ## Annual Income

# > There are outliers value in annual income for each loan status. <br>
# The quantile 25%, 50%, and 75% of annual income for each loan status has similiar value distribution. <br>
# What differentiate is the value distribution of outlier values for each loan status.

# ### Loan Status by Annual Income

# In[ ]:


plt.figure(figsize=(15,7))
gg = sns.boxplot(x="loan_status", y="annual_inc", data=df_loan, order = df_loan['loan_status'].value_counts().index,color='r')
gg.set_xticklabels(gg.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
plt.show()


# In[ ]:


q3=df_loan['annual_inc'].quantile([0.999]).values
# Exclude data outside quantile 9.999%
plt.figure(figsize=(15,7))
gg = sns.boxplot(x="loan_status", y="annual_inc", data=df_loan, order = df_loan['loan_status'].value_counts().index,color='r')
gg.set_xticklabels(gg.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
plt.ylim([0,q3])
plt.show()


# * ## Home Ownership

# > The top three home ownership for the loan application throughout 2007-2018: Mortgage (49%), Rent (39.6%), Own (11.2%).
# <br><br>Generally, the loan amount value distribution from joint application is higher than from individual application.
# <br><br>The Home Ownership tends to does not strongly correlated with status loan since the chance rate for all status loan by all home ownership is relatively similar each other.

# ### Home Ownership Distribution

# In[ ]:


plt.figure(figsize=(11,6))

g = sns.countplot(x='home_ownership', data=df_loan,order = df_loan['home_ownership'].value_counts().index,color='cyan')
g.set_title("Home Ownership Distribution", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Home Ownership", fontsize=15)
g.set_ylabel('Count', fontsize=15)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=11) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# ### Home Ownership & Loan Amount by Application Type

# In[ ]:


plt.figure(figsize = (11,6))

g = sns.boxenplot(x="home_ownership",y="loan_amnt",data=df_loan,hue="application_type")
g.set_title("Home Ownership - Loan Amount", fontsize=20)
g.set_xlabel("Home Ownership", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)
plt.show()


# ### Loan Status by Home Ownership

# In[ ]:


cm = sns.light_palette("cyan", as_cmap=True)
round(pd.crosstab(df_loan['loan_status'], df_loan['home_ownership'],normalize='columns')*100,2).fillna(0).style.background_gradient(cmap = cm)


# * ## Grade and Sub-Grade

# In[ ]:





# > In Fully Paid cases, all grade has the similar chance rate by about 40%-50%. In charged off cases, grade E,F,G have the highest chance rate of occurency.<br>
# While F1,F2,F3,F4,F5,F1,G2,G3,G4,and G5 sub-grade have the highest chance rate of charged off by moret than 30%.

# ### Loan Status by Grade

# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan['loan_status'], df_loan['grade'], normalize='columns')*100,2).style.background_gradient(cmap = cm)


# ### Loan Status by Sub-Grade

# In[ ]:


cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_loan['loan_status'], df_loan['sub_grade'], normalize='columns')*100,2).style.background_gradient(cmap = cm)


# * ## Revolving Line Utilization Rate

# > Overall, default cases occur to borrower with revolving line utilization rate more than 50%. While in fully paid cases, mostly the revolving utilization rate less than 50%.

# In[ ]:


cm = sns.light_palette("purple", as_cmap=True)
round((pd.crosstab(df_loan['loan_status'], df_loan['purpose'],values=df_loan['revol_util'], aggfunc='mean')).fillna(0),2).style.background_gradient(cmap = cm)


# * ## Employment Title

# > The top five borrower by the employment title: Teacher, Manager, Owner, Nurse, and Drive. <br>
# > Overall, the chance rate of fully paid for all emplyment title is similar by 40-50%, except Owner (34%) and Drive (37%). <br>
# > Driver, Truck Driver, and Store Manager have the largest chance rate of charged off by more than 14%, while other employment title lower than 12% rate.

# ### Employment Title Distribution

# In[ ]:


# Into lowercase
df_loan['emp_title']=df_loan['emp_title'].str.lower()

plt.figure(figsize=(20,8))
sns.set(style='darkgrid')

g = sns.countplot(x='emp_title', data=df_loan, palette='hls',order = df_loan['emp_title'].value_counts().index.values[:20])
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Employment Title", fontsize=15)
g.set_ylabel("Total Loans", fontsize=15)
g.set_title("Number of Loans by Employment Title", fontsize=20)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 1000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# ### Loan Status by Employment Title

# In[ ]:


emp_title = df_loan.emp_title.value_counts()[:30].index.values 
cm = sns.light_palette("red", as_cmap=True)

round(pd.crosstab(df_loan[df_loan['emp_title'].isin(emp_title)]['emp_title'], df_loan[df_loan['emp_title'].isin(emp_title)]['loan_status'], normalize='index')*100,2).style.background_gradient(cmap = cm)


# ### Grade by Employment Title

# In[ ]:


emp_title = df_loan.emp_title.value_counts()[:30].index.values 
cm = sns.light_palette("blue", as_cmap=True)

round(pd.crosstab(df_loan[df_loan['emp_title'].isin(emp_title)]['emp_title'], df_loan[df_loan['emp_title'].isin(emp_title)]['grade'], normalize='index')*100,2).style.background_gradient(cmap = cm)


# ### Sub-Grade by Employment Title

# In[ ]:


cm = sns.light_palette("blue", as_cmap=True)

round(pd.crosstab(df_loan[df_loan['emp_title'].isin(emp_title)]['emp_title'], df_loan[df_loan['emp_title'].isin(emp_title)]['sub_grade'], normalize='index')*100,2).style.background_gradient(cmap = cm)


# * ## Employment Length

# > Over 1/3 of the loans throughout 2007-2018 are requested by employee with working experienced 10 or above years. The rest portion evenly distributed by less than 1 year to 9 year employees. <br>
# > Employee with experience less than 1 year have the least chance rate of fully paid plus have the largest rate of current status.
# > Employee with experience 7,8, and 9 years has the best chance rate of fully paid status.
# > The loan amount for each employment length category is similar each other.

# ### Employment Length Distribution

# In[ ]:


orders = ['< 1 year', '1 year', '2 years', '3 years','4 years', '5 years', '6 years', '7 years','8 years', '9 years', '10+ years']
plt.figure(figsize=(20,8))
sns.set(style='darkgrid')

g = sns.countplot(x='emp_length', data=df_loan, color='blue',order = orders)
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Employment Length", fontsize=15)
g.set_ylabel("Total Loans", fontsize=15)
g.set_title("Number of Loans by Employment Length", fontsize=20)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# ### Loan Amount by Employment Length

# In[ ]:


plt.figure(figsize = (11,6))

g = sns.violinplot(x="emp_length", y="loan_amnt",data=df_loan, color='gray',order=orders)
g.set_title("Loan Amount Distribution by Employment Length", fontsize=20)
g.set_xlabel("Employment Length", fontsize=15)
g.set_ylabel("Loan Amount", fontsize=15)

plt.legend(loc='upper left')
plt.show()


# ### Loan Status by Employment Length

# In[ ]:


cm = sns.light_palette("red", as_cmap=True)
round(pd.crosstab(df_loan['emp_length'],df_loan['loan_status'],   normalize='index') * 100,3)[:15].style.background_gradient(cmap = cm)


# * ## Loan Credit Title

# > Similar to Purpose feature, the title feature has the same pattern with debt consolidation, credit card refinancing, and home improvement have the largest share.

# ### Credit Loan Title Distribution

# In[ ]:


# Into lowercase
df_loan['title']=df_loan['title'].str.lower()

plt.figure(figsize=(20,8))
sns.set(style='darkgrid')

g = sns.countplot(x='title', data=df_loan, color='red',order = df_loan['title'].value_counts().index.values[:20])
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Credit Loan Title", fontsize=15)
g.set_ylabel("Total Loans", fontsize=15)
g.set_title("Number of Loans by Credit Loan Title", fontsize=20)

sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,height + 10000,'{:1.2f}%'.format(height/total_data*100),ha="center", fontsize=12) 
g.set_ylim(0, max(sizes) * 1.10)
plt.show()


# * ## DTI Ratio (Debt to Income Ratio)

# > There are outliers DTI value for each loan status. <br>
# The quantile 25%, 50%, and 75% of DTI value distribution for each loan status has similiar value distribution. <br>
# What differentiate is the value distributino of outlier values for each loan status.

# In[ ]:


q3=df_loan['dti'].quantile([0.999]).values
plt.figure(figsize=(11,6))

g = sns.boxplot(x="loan_status", y="dti", data=df_loan, order = df_loan['loan_status'].value_counts().index,color='purple')
g.set_xticklabels(g.get_xticklabels(),fontdict={'horizontalalignment':'right'},rotation=45)
g.set_xlabel("Loan Status", fontsize=15)
g.set_ylabel("DTI Distribution", fontsize=15)
g.set_title("Loan Status by DTI", fontsize=20)
plt.ylim([0,q3])
plt.show()


# ### Loan Status & Purpose by DTI 

# In[ ]:


cm = sns.light_palette("yellow", as_cmap=True)
round((pd.crosstab(df_loan['loan_status'], df_loan['purpose'],values=df_loan['dti'], aggfunc='mean')).fillna(0),2).style.background_gradient(cmap = cm)


# # Feature Selection (Part 1)
# ## Dealing with the Dominant Missing Values

# In[ ]:


# Create a new dataframe to plot how dominant the missing values are
data={"feature":df_loan.columns,
     "missing_value":df_loan.isnull().sum()}
df1=pd.DataFrame(data=data)
plt.figure(figsize=(20,6))
sns.barplot(x='feature',y='missing_value',data=df1)
plt.ylim(0,df_loan.shape[0])


# In[ ]:


df1['missing_value'].describe()


# Important information regarding the distribution of missing values throughout all features:
# - q0 (min) = 0
# - q1 (25%) = 29
# - q2 (50%) = 70276
# - q3 (75%) = 2139958
# - q4 (max) = 2260668

# Most likely, features that are dominated by missing values will not help much as features for the prediction model, even those features may be noises in the modeling. Hence, we specify the threshold for maximum containing missing value as 0.09% (2000 observations).

# In[ ]:


# select features that pass the threshold
chosen_feature=df1['feature'][df1['missing_value']<=2000].values


# In[ ]:


# Create a new dataframe containing selected features
df_data=df_loan[chosen_feature]


# Dataframe consists 2,260,668 observations and 41 features with data type <u>*int, float, and object (string)*</u>. All features for prediction models need to convert into numeric <u>*(int or float)*</u> datatype. Thus, the features with type object need to converted into numeric type first.

# In[ ]:


df_data.shape


# In[ ]:


df_data.head()


# # Convert Loan Status into Good or Bad Loan

# In[ ]:


bad_loan = ["Charged Off","In Grace Period", "Default","Late (16-30 days)", "Late (31-120 days)","Does not meet the credit policy. Status:Charged Off"]
dat_temp=df_data.loan_status

def loan_condition(status):
    if status in bad_loan:
        return 'Bad Loan'
    else:
        return 'Good Loan'

df_data['loan_status'] = df_data['loan_status'].apply(loan_condition)


# # Convert the Object Type into Numerical Type
# ## Feature Selection for Object Type

# There are 17 features with object type. We will not convert all of those, but just the features that will have high impact for the model and feature with categorical values.

# In[ ]:


# Create a new dataframe specifically for object type
df_obj=df_data.select_dtypes(include='object')


# In[ ]:


df_obj.shape


# In[ ]:


df_obj.describe(include=['O'])


# Based on that descriptive statistics above, several categorical features need to be dropped because the level of values is too varied or not significantly influece the target variable (<i>loan_status</i>), as following:
# - issue_d
# - zip_code
# - addr_state
# - earliest_cr_line
# - last_credit_pull_d
# - disbursement_method
# - debt_settlement_flag

# In[ ]:


# drop some features
features_dropped = ['issue_d','zip_code','addr_state','earliest_cr_line','last_credit_pull_d','disbursement_method','debt_settlement_flag']
df_obj.drop(features_dropped,axis=1, inplace=True)
df_data.drop(features_dropped,axis=1, inplace=True)


# ## Converting into Numerical Type (by using LabelEncoder)

# In[ ]:


features=df_obj.columns
for feature in features:
    le = LabelEncoder()
    le.fit(df_obj[feature])
    df_obj[feature]=list(le.transform(df_obj[feature]))
    df_data[feature]=df_obj[feature]


# In[ ]:


df_obj.info()


# In[ ]:


# update the dataframe by dropping the 'police_code' feature
df_data.drop('policy_code',axis=1, inplace=True)


# In[ ]:


temp_features=df_data.drop('loan_status',axis=1).columns
x1=list(temp_features[:14])
x1.append('loan_status')
x2=list(temp_features[14:27])
x2.append('loan_status')
x3=list(temp_features[27:len(temp_features)])
x3.append('loan_status')


# Because there are so many features within dataframe <i>df_data</i>, we divide by three parts data for creating the heatmap. 

# In[ ]:


plt.figure(figsize=(35,25))
sns.set(font_scale=2)
sns.heatmap(df_data[x1].corr(), vmin=-0.42,vmax=0.42,square=True, annot=True,cmap='YlGnBu', cbar=True)


# In[ ]:


plt.figure(figsize=(35,25))
sns.heatmap(df_data[x2].corr(), vmin=-0.42,vmax=0.42,square=True, annot=True,cmap='YlGnBu', cbar=True)


# In[ ]:


plt.figure(figsize=(35,25))
sns.heatmap(df_data[x3].corr(), vmin=-0.42,vmax=0.42,square=True, annot=True,cmap='YlGnBu', cbar=True)


# # Feature Selection (Part 2)

# Based on domain knowledge sense and output from heatmap above, here are the features we would like to retain:
# - loan amount (loan_amnt)
# - interest rate (int_rate)
# - term (term)
# - grade (grade)
# - sub-grade (sub_grade)
# - home ownership (home_ownership)
# - annual income (annual_inc)
# - purpose (purpose)
# - DTI (dti)
# - revolving utilization rate (revol_util)
# - total accumulation of credit line (total_acc)
# - last payment amount (last_pymnt_amnt)
# - total payment to date (total_pymnt)
# - loan status (loan_status)

# In[ ]:


features_selected = ['loan_amnt','int_rate','term','grade','sub_grade','home_ownership','annual_inc','purpose',
                    'dti','revol_util','total_acc','last_pymnt_amnt','total_pymnt','loan_status']
df_data=df_data[features_selected]


# In[ ]:


df_data.isnull().sum()


# ## Handle the missing value

# In[ ]:


df_data.annual_inc.fillna(df_data.annual_inc.median(), inplace=True)     
df_data.dti.fillna(df_data.dti.median(), inplace=True)
df_data.revol_util.fillna(df_data.revol_util.median(), inplace=True)
df_data.total_acc.fillna(df_data.total_acc.median(), inplace=True)


# In[ ]:


df_data.isnull().sum()


# # Model Building
# ## Splitting into Training and Testing Set

# In[ ]:


# Splitting the dataset in X (features) and Y (target variable)
X=df_data.drop(['loan_status'],axis=1)
Y=df_data['loan_status']

# Splitting the dataset in training (75%) and testing set (25%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[ ]:


#Feature Scaling
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.fit_transform(X_test)


# ## Model Fitting

# ###### So far the candidate for model algorithms as follow:
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - Bagging Classifier

# ### Import the Modules

# In[ ]:


from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# ### Logistic Regression

# > In training set, Logistic Regression algorithm provide result with accuracy around 88%

# In[ ]:


# Logistic Regression
clf_log_reg = LogisticRegression()
clf_log_reg.fit(X_train, Y_train)
acc_log_reg=round((cross_val_score(clf_log_reg,X_train, Y_train, cv=3,scoring='accuracy').mean())*100,2)
print("Accuracy: "+str(acc_log_reg)+"%")


# In[ ]:


"""
# Support Vector Machine
clf_svc=SVC() 
clf_svc.fit(X_train, Y_train)
acc_svc=round((cross_val_score(clf_svc,X_train, Y_train, cv=3,scoring='accuracy').mean())*100,2)
print("Accuracy: "+str(acc_svc)+"%")
"""


# ### Decision Tree

# > In training set, Decision Tree algorithm provide result with accuracy more than 94%

# In[ ]:


# Decision Tree
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, Y_train)
acc_decision_tree=round((cross_val_score(clf_dt,X_train, Y_train, cv=3,scoring='accuracy').mean())*100,2)
print("Accuracy: "+str(acc_decision_tree)+"%")


# ### Random Forest

# > In training set, Random Forest algorithm provide result with accuracy about 92%

# In[ ]:


# Random Forest
clf_rf = RandomForestClassifier(n_estimators=3)
clf_rf.fit(X_train, Y_train)
acc_random_forest=round((cross_val_score(clf_rf,X_train, Y_train, cv=3,scoring='accuracy').mean())*100,2)
print("Accuracy: "+str(acc_random_forest)+"%")


# In[ ]:


"""
# Bagging (estimator KNN)
clf_bagging = BaggingClassifier(
    KNeighborsClassifier(
        n_neighbors=3,
        weights='distance'
        ),
    oob_score=True,
    max_samples=0.5,
    max_features=1.0
    )
clf_bagging.fit(X_train, Y_train)
acc_bagging=round((cross_val_score(clf_bagging,X_train, Y_train, cv=3,scoring='accuracy').mean())*100,2)
print("Accuracy: "+str(acc_bagging)+"%")
"""


# ## Model Prediction

# In[ ]:


# Predicting & Model Evaluation

# Predicting the results
log_pred = clf_log_reg.predict(X_test)
#svc_pred = clf_svc.predict(X_test)
dt_pred = clf_dt.predict(X_test)
rf_pred = clf_rf.predict(X_test)
#bag_pred = clf_bagging.predict(X_test)


# ## Model Validation
# ### Confusion Matrix from Logistic Regression

# In[ ]:


# Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(Y_test,log_pred)
accuracy_1 = (cm_log[0,0]+cm_log[1,1])/len(Y_test)


# ### Confusion Matrix from Decision Tree

# In[ ]:


cm_dt = confusion_matrix(Y_test,dt_pred)
accuracy_4 = (cm_dt[0,0]+cm_dt[1,1])/len(Y_test)
cm_dt


# ### Confusion Matrix from Random Forest

# In[ ]:


cm_rf = confusion_matrix(Y_test,rf_pred)
accuracy_5 = (cm_rf[0,0]+cm_rf[1,1])/len(Y_test)
cm_rf


# In[ ]:


print("Accuracy Logistic Regression in Testing Set:",accuracy_1*100,'%',
      #"\nAccuracy SVC:",accuracy_2*100,'%',
      "\nAccuracy Decision Tree in Testing Set:",accuracy_4*100,'%',
      "\nAccuracy Random Forest in Testing Set:",accuracy_5*100,'%')
      #"\nAccuracy Bagging Classifier:",accuracy_6*100,'%'

