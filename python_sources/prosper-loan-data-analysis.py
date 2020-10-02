#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/prosperLoanData.csv')


# In[ ]:


# Shape of entire dataset
df.shape


# In[ ]:



df.dtypes


# In[ ]:


# Summary statistics
df.describe()


# In[ ]:


# Duplicates data entry in loan data
df.duplicated().sum()


# In[ ]:


df.isnull().sum()


# In[ ]:


df_loan = df.copy()


# In[ ]:


df_loan.info()


# In[ ]:


# Changing Loan orgination date into date time format
df_loan['LoanOriginationDate'] = pd.to_datetime(df_loan['LoanOriginationDate'])


# In[ ]:


df_loan.dtypes


# In[ ]:


# filter out loans without ProsperScores
df_loan = df_loan[df_loan['ProsperScore'].isnull()==False]


# ## Univariate Exploration

# In[ ]:


# Loan by term
base_color = sb.color_palette()[0]
sb.countplot(data=df_loan,x= 'Term',color=base_color);
plt.title('Terms of loan (Months)')
plt.xlabel('Term (Months)');


# Most common term of loans is 36 months

# In[ ]:


type_count = df_loan['LoanStatus'].value_counts()
type_order = type_count.index


# In[ ]:


# Count of Loan by Loan Status
n_loan =df_loan.shape[0]
max_type_count = type_count[0]
max_prop = max_type_count/n_loan


# In[ ]:


tick_props = np.arange(0,max_prop,0.1)
tick_names = ['{:0.2f}'.format(v) for v in tick_props]


# In[ ]:


tick_names


# In[ ]:


sb.countplot(data=df_loan,y='LoanStatus',color=base_color,order=type_order);
plt.xticks(tick_props*n_loan,tick_names)
plt.xlabel('proportion');
plt.title('Proportion of Loan Status')


# Around 25% of total loan are completed but still majority of loans are in currentor pending state(around 80%)

# In[ ]:


df['ProsperScore'].describe()


# In[ ]:


# Distribution of Prosper rating
sb.countplot(data=df_loan,x='ProsperRating (numeric)',color=base_color);
plt.title('Count of Prosper ratings')


# Most of borrowers has got 4 prosper ratings that means most of borrowers has risk associated on the higher end

# In[ ]:


df_loan['Year'] = df_loan['LoanOriginationQuarter'].str[-4:]


# In[ ]:


# Number of loans per year
sb.countplot(data=df_loan,x='Year',color=base_color);
plt.title('Numner of Loans Sanctioned per year')
plt.xticks(rotation=90);


# In 2009 there was lowest number of loans sanctioned whereas 2013 has got highest number of loans sanctioned

# In[ ]:


df['LoanOriginalAmount'].describe()


# In[ ]:


# Distribution of orginal Loan amount
bins = np.arange(1000,35000,2000)
plt.hist(data=df_loan,x='LoanOriginalAmount',color=base_color,bins=bins);
plt.title('Distribution of orginal Loan amount')
plt.xticks(rotation=90);


# In[ ]:


# Histogram for Credit Score ranges

plt.figure(figsize = [13, 5]) 


plt.subplot(1, 2, 1)
bins = np.arange(550, df_loan['CreditScoreRangeLower'].max(), 20)
plt.hist(data = df_loan, x = 'CreditScoreRangeLower', bins = bins)
plt.xticks(np.arange(550, 1000, 100))
plt.title('CreditScoreRangeLower Count')
plt.xlabel('CreditScoreRangeLower')
plt.ylabel('count');

plt.subplot(1, 2, 2)
bins = np.arange(550, df_loan['CreditScoreRangeUpper'].max(), 20)
plt.hist(data = df_loan, x = 'CreditScoreRangeUpper', bins = bins)
plt.xticks(np.arange(550, 1000, 100))
plt.title('CreditScoreRangeUpper Count')
plt.xlabel('CreditScoreRangeUpper')
plt.ylabel('count');


# These two histograms shows similar trend. As both the upper and lower score are ranges of credit score

# In[ ]:


df_loan['LenderYield'].describe()


# In[ ]:


# Distribution of lender yield
bins = np.arange(.03,.34,.01)
plt.hist(data=df_loan,x='LenderYield',color=base_color,bins=bins);
plt.title('Distribution of Lender yield')
plt.xticks(rotation=90);


# Data is positively skewed, suggests that for investors got good yield for loans. Data is spiked at 34%.

# In[ ]:


# Income range of borrower
order = ['$0','$1-24,999','$25,000-49,999','$50,000-74,999','$75,000-99,999','$100,000+']
sb.countplot(data=df_loan,x='IncomeRange',color=base_color,order=order);
plt.title('Count of Income Range')
plt.xticks(rotation=90);


# ###  What is/are the change(s)  made to tidy dataset?
# 
# We have removed the rows where there is no data about prosper score
# 
# ###  What is/are the main feature(s) of interest in your dataset?
# 
# The Borrower's APR will be analyzied with many factors such as the borrower's rating, score, Employment Status and income that could influence change in borrower's APR. 
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# The Prosper Rating and score could show low Borrower's APR because higher rating reflect the borrower's personality to be more trustworthy. 

# ## Bivariate Analysis

# In this section, we will deeply analyzed the relationship of key variables of out dataset

# In[ ]:


# correlation plot 

num_vars = ['BorrowerAPR', 'ProsperScore', 'LenderYield', 
            'StatedMonthlyIncome',  'CreditScoreRangeUpper','ProsperRating (numeric)','DebtToIncomeRatio']
plt.figure(figsize = [8, 5])
sb.heatmap(df_loan[num_vars].corr(), annot = True, fmt = '.3f',
           cmap = 'vlag_r', center = 0)
plt.title('Correlation Plot') 
plt.show()


# Strong positive correlations  between Lender yield and Borrower APR. prosper score and prosper rating are also positive correlation .Credit score upper range has also some weak +ve correlation with prosper score. 
# 
# Negative correlation between prosper score & APR, and prosper score & Lender yield.
# Negative correlation between prosper ratings & APR, and prosper score & Lender yield.
# 
# Let's investigate further.

# In[ ]:


# plot matrix: only 300 random loans are used to see the pattern more clearer


num_vars = ['BorrowerAPR', 'ProsperScore', 'LenderYield', 
            'StatedMonthlyIncome',  'CreditScoreRangeUpper','ProsperRating (numeric)','DebtToIncomeRatio']

samples = np.random.choice(df_loan.shape[0], 300, replace = False)
loan_samp = df_loan.loc[samples,:]

g = sb.PairGrid(data = loan_samp, vars = num_vars)
g.map_offdiag(plt.scatter)
plt.title('Matrix Plot');


# Borrower APR is negatively related with prosper score and credit upper score. However borrower APR and lending yield are postively correlated as higher the APR will be, higher will be yield for lender.
# 
# Prosper score is negatively related with Borrower APR and lender yield
# 
# Debt To Income Ratio and Monthly income is not seems to be related with any variable. We have to analyzed this further.

# In[ ]:


# scatter and heat plot for comparing ProsperScore and BorrowerAPR. 
plt.figure(figsize = [15, 5]) 

plt.subplot(1, 2, 1)
plt.scatter(data = df_loan, x = 'BorrowerAPR', y = 'ProsperScore', alpha =  0.005)
plt.yticks(np.arange(0, 12, 1))
plt.title('BorrowerAPR vs. ProsperScore')
plt.xlabel('BorrowerAPR')
plt.ylabel('ProsperScore')


plt.subplot(1, 2, 2)
bins_x = np.arange(0, df_loan['BorrowerAPR'].max()+0.05, 0.03)
bins_y = np.arange(0, df_loan['ProsperScore'].max()+1, 1)
plt.hist2d(data = df_loan, x = 'BorrowerAPR', y = 'ProsperScore', bins = [bins_x, bins_y], 
               cmap = 'viridis_r', cmin = 0.5)
plt.colorbar()
plt.title('BorrowerAPR vs. ProsperScore')
plt.xlabel('BorrowerAPR (l)')
plt.ylabel('ProsperScore');


# Here the relationship is evident, higher the prosper score is lower is Borrower APR and this makes sense because lower the risk attached with the borrower lower will be the APR.

# In[ ]:


# scatter and heat plot for comparing BorrowerAPR and credit score upper range. 
plt.figure(figsize = [15, 5]) 

plt.subplot(1, 2, 1)
plt.scatter(data = df_loan, x = 'CreditScoreRangeUpper', y = 'BorrowerAPR', alpha = 0.01)
plt.title('BorrowerAPR vs. CreditScoreRangeUpper')
plt.xlabel('BorrowerAPR')
plt.ylabel('CreditScoreRangeUpper');


plt.subplot(1, 2, 2)
bins_x = np.arange(0, df_loan['BorrowerAPR'].max()+0.05, 0.02)
bins_y = np.arange(500, df_loan['CreditScoreRangeUpper'].max()+100, 20)
plt.hist2d(data = df_loan, x = 'BorrowerAPR', y = 'CreditScoreRangeUpper', bins = [bins_x, bins_y], 
               cmap = 'viridis_r', cmin = 0.5)
plt.colorbar()
plt.title('BorrowerAPR vs. CreditScoreRangeUpper')
plt.xlabel('BorrowerAPR (l)')
plt.ylabel('CreditScoreRangeUpper');


# We can  see the trend that the higher the CreditScore leads to lower APR percentage. The heatmap on the same variables helps to make this point more clear.

# In[ ]:


# Stated MonthlyIncome vs Prosper Rating
plt.figure(figsize = [15, 5])

plt.subplot(1, 2, 1)
sb.boxplot(data=df_loan,x='ProsperScore',y='StatedMonthlyIncome',color=base_color);
plt.xlabel('Prosper Score');
plt.ylabel('Monthly Income');
plt.title('Box plot of monthly income Vs prosper Score');

plt.subplot(1, 2, 2)
plt.scatter(data=df_loan,x='BorrowerAPR',y='StatedMonthlyIncome',color=base_color);
plt.xlabel('BorrowerAPR');
plt.ylabel('Monthly Income');
plt.title('Scatter plot of monthly income Vs Borrower APR');


# There was no clear evidence of any relationship between income with any other variable so lets investigate more here. 
# 
# There is slight decrease in median of monthly income as score get poorer but no relationship between monthly income and APR

# In[ ]:


# Borrower APR vs Status of Loan and  Borrower APR vs Employment status
plt.figure(figsize = [15, 5])

plt.subplot(1, 2, 1)
sb.boxplot(data=df_loan,x='BorrowerAPR',y='LoanStatus',color=base_color);
plt.xlabel('Borrower APR');
plt.ylabel('Loan Status');
plt.title('Box plot of Borrower APR vs Status of Loan');

plt.subplot(1, 2, 2)
sb.boxplot(data=df_loan,x='BorrowerAPR',y='EmploymentStatus',color=base_color);
plt.xlabel('Borrower APR');
plt.ylabel('Employment Status');
plt.title('Box plot of Borrower APR vs Employment Status');


# The median borrower APR of current, completed and Final payment in process are the lowest, with few low outliers of APR rate in charged off. Whereas charged off loans and defaulted are with the highest median of borrower rate. 
# 
# Median borrower APR is lowest for employed and highest for not employed because of high risk attached with unemployed people

# In[ ]:


df_series = df_loan['BorrowerRate'].groupby(df_loan['LoanOriginationQuarter']).mean().reset_index()


# In[ ]:


df_series.LoanOriginationQuarter = pd.Categorical(df_series.LoanOriginationQuarter, sorted(df_series.LoanOriginationQuarter, key=lambda x: x.split(' ')[-1]), ordered = True)
df_series.sort_values('LoanOriginationQuarter', inplace=True)


# In[ ]:


# Mean Borrower rate over time
plt.errorbar(data=df_series,x='LoanOriginationQuarter',y='BorrowerRate');
plt.xticks(rotation = 90);
plt.xlabel('Quaerters');
plt.ylabel('Mean Borrower rate');
plt.title('Quarter and rate trends');


# We can't find any trend here.

# ###  What are the observed relationships in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?
# 
# It was interesting finding to know the reason behind ProsperRating. I was able to eshtablish clear relation between lender yield and borrower rate.
# Borrower rate and Lender Yield have a linear relationship

# ## Multivariate analysis

# In[ ]:


# LenderYield vs Borrower APR  vs ProsperRating
plt.figure(figsize = [10, 5])
plt.scatter(data=df_loan,x='LenderYield',y = 'BorrowerAPR',c='ProsperScore',cmap = 'viridis_r')
plt.colorbar(label = 'ProsperScore');
plt.xlabel('Lender Yield')
plt.ylabel('Borrower APR')
plt.title('LenderYield vs Borrower APR  vs ProsperRating');


# This graphs clearly shows the relationship between all variables. Borrower APR and Lender yield are directly positively correlated as more the interest borrowers pays,more will be yield for lender. For the prosper score, higher the prosper score lower will be the risk attached hence lower will be the APR and that further lowers down the yield.

# In[ ]:


# BorrowerAPR vs. CreditScoreRangeUpper & CreditScoreRangeUpper
plt.figure(figsize = [15, 5]) 
plt.scatter(data = df_loan, x = 'CreditScoreRangeUpper', y = 'BorrowerAPR', c ='ProsperScore', alpha = 0.3)
plt.colorbar(label = 'ProsperScore')
plt.title('BorrowerAPR vs. CreditScoreRangeUpper & CreditScoreRangeUpper')
plt.xlabel('BorrowerAPR')
plt.ylabel('CreditScoreRangeUpper');


# Credit score range upper and prosper score are positively correlated. However the high credit score upper range and borrower APR are negatively correlated.By adding ProsperScore to color encodings, BorrowerAPR decreases as ProsperScore increases. This proves the point that CreditScoreRangeUpper and ProsperScore negatively correlated to BorrowerAPR. 

# In[ ]:


# LoanStatus Vs BorrowerAPR VS EmploymentStatus
plt.figure(figsize=[12,10])
sb.boxplot(x="LoanStatus", y="BorrowerAPR", hue="EmploymentStatus", data=df_loan, palette="RdYlBu");
plt.xticks(rotation = 90);
plt.xlabel('Loan Status');
plt.ylabel('BorrowerAPR');
plt.title('LoanStatus Vs BorrowerAPR VS EmploymentStatus');


# For each category of loan status, the lowest APR is for Employed and Full-time. Whereas highest APR is for Not employed.

# ### What are the observed relationships in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?
# After analyzing the variables across dataset, Borrower APR and Lender yield are directly positively correlated which is obvious as more the APR, more will be the interest and more will be lender's yield. However, CreditScoreRangeUpper and ProsperScore negatively correlated to BorrowerAPR. 
# Scatter plot and Heatmap were also created to find out that ProsperScore and BorrowerAPR were negatively correlated as higher the prosper score lower will be the risk attached hence lower will be the APR and that further lowers down the yield.
# Multiple box plots were analyzed for variables Loan status, APR and Employment Status. We can very well see that for For each category of loan status, the lowest APR is for Employed and Full-time. Whereas highest APR is for Not employed. That means borrower APR is related to Employment status. More secure jobs get low APR and vice versa.

# In[ ]:




