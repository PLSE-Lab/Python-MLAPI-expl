# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# ---
# title: "Loans analysis"
# author: "Sindhu Rao"
# date: "20 Feb 2018"
# ---
# Any results you write to the current directory are saved as output.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
loandata = pd.read_csv('../input/loan.csv', sep=',', low_memory=False, parse_dates=['issue_d'])

print(loandata.head(5))
print(loandata.columns)
#print(loandata[['issue_d']])
print(loandata.shape)
#print(loandata.isnull().sum()) # number of missing values in each column

print(loandata.isnull().sum().value_counts()) # number of missing values and number of columns
# example: 0 35: 35 columns have 0 missing values
# example: 29 7: 7 columns have 29 missing values

# number of loans issued 
fig, ax = plt.subplots(1, 2)
sns.distplot(loandata['loan_amnt'], ax = ax[0])
ax[0].set_title('Loan Applied by the Borrower')
sns.distplot(loandata['funded_amnt'], ax = ax[1])
ax[1].set_title('Loan Funded by the investors')

# average loan issued by year
loandata['year'] = loandata.issue_d.dt.year
loandata['year'].head()
loandata.groupby(['year'])['loan_amnt'].mean().plot('bar')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average loan amount issued', fontsize=14)
plt.title('Issuance of Loans', fontsize=16)

# total loan issued by year
loandata.groupby(['year'])['loan_amnt'].sum().plot('bar')

# number of loans issued in each category
loandata.groupby('grade')['id'].count().plot('bar')
plt.xlabel('Grade', fontsize=14)
plt.ylabel('Number of loans issued', fontsize=14)
plt.title('Issuance of Loans by Grade', fontsize=16)

# Average loan amount by grade
loandata.groupby('grade')['loan_amnt'].mean().plot('bar')
plt.xlabel('Grade', fontsize=14)
plt.ylabel('Average loan amount by grade', fontsize=14)

# Avg interest rate in each category
subgrade = loandata['sub_grade'].str.get(1)
loandata['subgrade'] = subgrade
print(loandata.columns)

#using matplotlib
# loandata['int_rate','grade'].groupby('grade').mean()
#using snsplot
sns.distplot(loandata['int_rate'], bins = 30, kde = True, rug = False)
sns.kdeplot(loandata['int_rate'] )
sns.kdeplot(loandata['int_rate'], bw=.2, label="bw: 0.2")
sns.kdeplot(loandata['int_rate'], bw=2, label="bw: 2")
plt.legend()

#Box plot
sns.boxplot(x='grade', y="int_rate", data=loandata)
sns.boxplot(x='grade', y="int_rate", hue="subgrade", data=loandata)

# loans by status
loandata['loan_status'].value_counts()
BadLoan = ["Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Default", "Does not meet the credit policy. Status:Charged Off"]
def loancondition(status):
    if status in BadLoan:
        return 'Bad Loan'
    else:
        return 'Good Loan'
loandata['loan_condition'] = loandata['loan_status'].apply(loancondition)

# bar plot loans by status and year
sns.barplot(x='year', y='loan_amnt', hue='loan_condition', data=loandata, estimator=lambda x: len(x) / len(loandata) * 100)
ax[1].set(ylabel="(%)")

#loans by home ownership and loan condition
a = loandata.groupby(['home_ownership', 'loan_condition'])['id']
a.count().unstack()

#loans by state
loandata['addr_state'].unique()
west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
south_west = ['AZ', 'TX', 'NM', 'OK']
south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']

def findregion(state):
    if state in west:
        return 'west'
    elif state in south_west:
        return 'south_west'
    elif state in south_east:
        return 'south_east'
    elif state in mid_west:
        return 'mid_west'
    elif state in north_east:
        return 'north_east'
        
loandata['region'] = loandata['addr_state'].apply(findregion)

loandata['loan_amntk'] = loandata['loan_amnt']/1000

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

loandata.groupby(['issue_d', 'region'])['loan_amntk'].sum().unstack().plot(kind='area', stacked=True, grid=False, legend=False, ax=ax1, figsize=(16,12))
ax1.set_title('Average Loan Amount by Region', fontsize=14)
loandata.groupby(['issue_d', 'region'])['int_rate'].mean().unstack().plot(kind='area', stacked=True, grid=False, legend=False, ax=ax2, figsize=(16,12))
ax2.set_title('Average Interest Rate by Region', fontsize=14)
loandata.groupby(['issue_d', 'region'])['annual_inc'].mean().unstack().plot(kind='area', stacked=True, grid=False, legend=False, ax=ax3, figsize=(16,12))
ax3.set_title('Average Annual Income by Region', fontsize=14)
loandata.groupby(['issue_d', 'region'])['dti'].mean().unstack().plot(kind='area', stacked=True, grid=False, legend=False, ax=ax4, figsize=(16,12))
ax4.set_title('Average DTI by Region', fontsize=14)
ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=5,prop={'size':12}, ncol=5, mode="expand", borderaxespad=0.)


# employment length vs loan condition
loandata.groupby(['emp_length', 'loan_condition'])['id'].count().unstack().plot()

# loan amount and int rate by yera and grade
f, ((ax1, ax2)) = plt.subplots(1, 2)
loandata.groupby(['year', 'grade'])['loan_amnt'].mean().unstack().plot(legend=False, ax=ax1, figsize=(16,8))
ax1.set_title('Average Loan Amount by grade', fontsize=14)
loandata.groupby(['year', 'grade'])['int_rate'].mean().unstack().plot(legend=False, ax=ax2, figsize=(16,8))
ax2.set_title('Average Interest Rate by grade', fontsize=14)
ax2.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=5,prop={'size':12}, ncol=7, mode="expand", borderaxespad=0.)

#Number of loans by condition and grade
ax1 = plt.subplot(221)
loandata.groupby(['grade', 'loan_condition'])['loan_amnt'].size().unstack().plot(kind='bar', stacked=True,legend=True, ax=ax1, figsize=(16,8))
ax1.set_title('Number of loans by grade and loan condition', fontsize=14)
ax2 = plt.subplot(222)
loandata.groupby(['sub_grade', 'loan_condition'])['loan_amnt'].size().unstack().plot(kind='bar',stacked=True, legend=True, ax=ax2, figsize=(16,8))
ax2.set_title('Number of loans by sub grade and loan condition', fontsize=14)
ax3 = plt.subplot(212)
loandata.groupby(['year', 'loan_condition'])['int_rate'].mean().unstack().plot(legend=True, ax=ax3, figsize=(16,8))
ax3.set_title('Average interest rate by year and loan condition', fontsize=14)
ax3.set_ylabel('Interest Rate (%)', fontsize=12)

















 































































