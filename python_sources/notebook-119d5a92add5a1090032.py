#!/usr/bin/env python
# coding: utf-8

# # Loan Size and ROI
# 
# ### Lets look at the loans themselves and how they preform for Lending Club
# 
# As I am analyzing this data the most recent rates are:   
#    - 30-Year Fixed 3.64%   
#    - 15-Year Fixed 2.76%
#    
# To look at the value of loans, lets consider the funded_amnt_inv variable which is described as 
# "The total amount committed by investors for that loan at that point in time". This represents the
# principle amount of the loan.
# 
# By understanding the loan amounts we will be able to see what types of loans do the best for LC and
# which have the highest rates of return.

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#read the loan data
#parse dates for easy time slicing
ld = pd.read_csv('../input/loan.csv',low_memory=False, parse_dates = True)

#determine the percet full for each variable
pct_full = ld.count()/len(ld)
names = list(pct_full[pct_full > 0.75].index)

#reduce to mostly full data
loan = ld[names]


# In[ ]:


import seaborn as sns
import matplotlib

#I swear it makes graphs more meaningful
plt.style.use('fivethirtyeight')

#lets look at the distirbution of the loan amount
amount_hist = loan.funded_amnt_inv.hist()
amount_hist.set_title('Histogram of Loan Amount')


# In[ ]:


#the average loan is a little less than $15,000.00
loan.funded_amnt_inv.describe()
#np.median(loan.funded_amnt)


# From this initial look at the loan size we can see that the majority of the loans are around
# $10,000 and that the loan amount distribution has right skew and fat tails. By segmenting the loan 
# amounts by subgroups, we can paint a better picture of LC lending practices.

# In[ ]:


#look at difference between the length of the loans 36 vs. 60 month loans
termGroup = loan.groupby('term')
termGroup['funded_amnt_inv'].agg([np.count_nonzero, np.mean, np.std])


# We can see that there the majority of the loans(a little over 3/4ths) are 3 year loans and are for less 
# principal than the 5year loans as we would expect.

# ## Loans Over Time
# 
# Lets look at the loan sizing and number of loans overtime

# In[ ]:


#summarize loans by month

#hide the ugly warning
#!usually should set on copy of original data when creating variables!
pd.options.mode.chained_assignment = None 

#make new variable to groupby for month and year
loan['issue_mo'] = loan.issue_d.str[0:3]
loan['issue_year'] = loan.issue_d.str[4:]

loan_by_month = loan.groupby(['issue_year','issue_mo'])

avgLoanSizeByMonth = loan_by_month['funded_amnt_inv'].agg(np.mean).plot()
avgLoanSizeByMonth.set_title('Avg. Loan Size By Month')


# In[ ]:


NumLoansPerMo = loan_by_month.id.agg(np.count_nonzero).plot()
NumLoansPerMo.set_title('Number of Loans By Month')
NumLoansPerMo.set_xlabel('Issue Month')


# In[ ]:


#less granular look at loan volume
loanByYr = loan.groupby('issue_year')
loanYrPlt = loanByYr.id.agg(np.count_nonzero).plot(kind = 'bar')
loanYrPlt.set_title('Num Loans By Year')
loanYrPlt.set_xlabel('Issue Year')


# In[ ]:


import calendar
#get the counts by month
loanByMo = loan.groupby(['issue_d', 'issue_mo'])
numByDate = loanByMo.agg(np.count_nonzero).reset_index()

#average the monthly counts across years
counts_by_month = numByDate.groupby('issue_mo')
avg_loan_vol = counts_by_month.id.agg(np.mean)


moOrder = calendar.month_abbr[1:13]
mo_plt = sns.barplot(x = list(avg_loan_vol.index),y = avg_loan_vol, order = moOrder, palette = "GnBu_d")
mo_plt.set_title('Avg. Loan Volume Per Month')


# ## Loans By Month and Year
# There are a few conclusions that can be reached looking at the last few charts:   
# 
# - We can se that the number of loans that LC is giving overtime is increasing as we would expect 
# with a company that is rapidly growing. I assume that such a rapid growth curve in the number of 
# loans will be accompanied by a decrease in the overall quality of the loans.
# 
# - There does appear to be some seasonal trends in the number of loans given within the year. 
# There peaks occuring in July and October which show small trending build ups between months. It is hard
# to say if these seasonal trends are the same by year, we would have to break the monthly loans down
# into years. This will be done as follows.

# ## Loan Volume Over Time: Intrayear

# In[ ]:


#get the counts by mo/year
loanByMo = loan.groupby(['issue_d','issue_year','issue_mo'])
numByDate = loanByMo.agg(np.count_nonzero).reset_index()

#get the individual years
years = np.unique(loan.issue_year)

#just looking at the first year
tmp_agg = numByDate[numByDate.issue_year == '2007']
tmp_plt = sns.barplot(x = tmp_agg.issue_mo,y = tmp_agg.id, order = moOrder, palette = "GnBu_d")
tmp_plt.set_title('Loans By Month: 2007')


# In[ ]:


#plot the years in stacked graphs
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, 1, figsize=(5, 8), sharex=True)

#being lazy and not wanting to figure out a better way to do this
#Please let me know if any of you know a better way
y1 = numByDate[numByDate.issue_year == '2007']
y2 = numByDate[numByDate.issue_year == '2008']
y3 = numByDate[numByDate.issue_year == '2009']
y4 = numByDate[numByDate.issue_year == '2010']
y5 = numByDate[numByDate.issue_year == '2011']
y6 = numByDate[numByDate.issue_year == '2012']
y7 = numByDate[numByDate.issue_year == '2013']
y8 = numByDate[numByDate.issue_year == '2014']
y9 = numByDate[numByDate.issue_year == '2015']

sns.barplot(y1.issue_mo, y1.id, order = moOrder, palette="BuGn_d", ax=ax1)
ax1.set_ylabel("2007")

sns.barplot(x = y2.issue_mo,y = y2.id, order = moOrder, palette="BuGn_d", ax=ax2)
ax2.set_ylabel("2008")

sns.barplot(x = y3.issue_mo,y = y3.id, order = moOrder, palette="BuGn_d", ax=ax3)
ax3.set_ylabel("2009")

sns.barplot(x = y4.issue_mo,y = y4.id, order = moOrder, palette="BuGn_d", ax=ax4)
ax4.set_ylabel("2010")

sns.barplot(x = y5.issue_mo,y = y5.id, order = moOrder, palette="BuGn_d", ax=ax5)
ax5.set_ylabel("2011")

sns.barplot(x = y6.issue_mo,y = y6.id, order = moOrder, palette="BuGn_d", ax=ax6)
ax6.set_ylabel("2012")

sns.barplot(x = y7.issue_mo,y = y7.id, order = moOrder, palette="BuGn_d", ax=ax7)
ax7.set_ylabel("2013")

sns.barplot(x = y8.issue_mo,y = y8.id, order = moOrder, palette="BuGn_d", ax=ax8)
ax8.set_ylabel("2014")

sns.barplot(x = y9.issue_mo, y = y9.id, order = moOrder, palette="BuGn_d", ax=ax9)
ax9.set_ylabel("2015")

#look better
sns.despine(bottom=True)
plt.setp(f.axes, yticks = [], xlabel = '')
plt.tight_layout()


# So although these graphs aren't layed out in the best way, we can see that when plotted per year,
# the seasonality is essentially non-existent but because the volume in the past few years has been so much 
# higher the monthly averages are skewed by the last few years. The previously precieved seasonality was just
# a concequence of some higher volumes in 2014-2015. It might be the case that a sesonal pattern is 
# developing or could be exposed through standardizing the loan volumes.

# In[ ]:


loan['pct_paid'] = loan.out_prncp / loan.loan_amnt

loan[loan.loan_status == 'Current'].pct_paid.hist(bins = 50)

