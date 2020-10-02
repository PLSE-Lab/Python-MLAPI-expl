#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook will be used to share my analysis as to whether an SBA loan should be approved or denied based on the provided SBA loan data. I'm currently working as a Data Scientist in the Finance department for the company I work for, and I wanted to use this as a way to apply some of the things I've been learning.
# 
# Because I'm using this for learning, this will be very detailed as I will be explaining my thought process at each step.
# 
# ### Why SBA Loan Approval?
# My first job out of college was at a community bank as a Credit Analyst, where I spent my time underwriting loans for small businesses. I loved being able to dig into the financial statements for each business and see how businesses in different industries operate. This role taught me the importance of small businesses and the role they play in our communities. It also taught me about some of the struggles entreprenuers face when starting a business, including the initial capital necessary to get started.
# 
# Small business owners often seek out SBA (Small Business Association) loans because they guarantee part of the loan. Without going into too much detail, this basically means that the SBA will cover some of the losses should the business default on the loan, which lowers the risk involved for the business owner(s). This increases the risk to the SBA however, which can sometimes make it difficult to get accepted for one of their loan programs. The SBA will play a particularly important role in small business success now with the COVID-19 pandemic which is impacting many small businesses around the world, crippling most of them. I thought it would be interesting to see if I could determine whether or not an SBA loan should be accepted or not given certain characteristics like the industry of the business, the size of the loan, amount of the loan that is guaranteed, etc.

# In[ ]:


# Import packages used for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[ ]:


# Load the SBA loan data and make a copy for exploration
df = pd.read_csv('../input/should-this-loan-be-approved-or-denied/SBAnational.csv')

df_copy = df.copy()


# # Data Exploration
# I begin by taking a look at the data I'll be using for modeling and analysis.
# 
# ### Data Cleaning, Formatting, and Feature Engineering
# Let's see what we're working with.

# In[ ]:


df_copy.head()


# In[ ]:


df_copy.shape


# I begin by checking for null values in the dataset.

# In[ ]:


df_copy.isnull().sum()


# I'm not concerned with ChgOffDate since we're focused on whether or not a loan gets charged off at all rather than when it happens, but the other columns could pose issues. Let's start by removing records from some of these columns with null values.
# 
# I decided to remove the rows entirely rather than imputing because we have a large number of records to work with, and it's hard to know what would be the best imputing method would be given the nature of the information (for example, I don't want to just assume whether the business was new or existing as this has the potential to be a very important feature for us to consider).

# In[ ]:


# Drop null values from specified columns
df_copy.dropna(subset=['Name', 'City', 'State', 'BankState', 'NewExist','RevLineCr', 'LowDoc', 'DisbursementDate', 'MIS_Status'], inplace=True)
df_copy.isnull().sum()


# Next, I want to start making sure each field is the appropriate data type.

# In[ ]:


# Check data types of each feature
df_copy.dtypes


# Looks like we're going to need to make some changes here. I begin with the currency fields that are currently being read as objects rather than floats.
# 
# Let's see how they are being read right now.

# In[ ]:


df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].head()


# It looks like they're coming in as strings because the '$' sign and commas are included. I can't change the type to a float without removing those, so I make those edits here.

# In[ ]:


# Remove '$', commas, and extra spaces from records in columns with dollar values that should be floats
df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = df_copy[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', ''))


# Next, I take a look at ApprovalFY which should be an integer but is coming up as an object type.

# In[ ]:


# Check the number of each data type in the field
df_copy['ApprovalFY'].apply(type).value_counts()


# In[ ]:


df_copy['ApprovalFY'].unique()


# We have a mixture of integers and strings here, with one record including an 'A' as well. I clean these next.

# In[ ]:


# Create a function to apply formatting to the records of str type only
def clean_str(x):
    if isinstance(x, str):
        return x.replace('A', '')
    return x


df_copy['ApprovalFY'] = df_copy['ApprovalFY'].apply(clean_str).astype('int64')


# Now I'll change the type of a few other columns as appropriate.

# In[ ]:


# Change the type of NewExist to an integer, Zip and UrbanRural to str (categorical) and all currency-related fields to float values
df_copy = df_copy.astype({'Zip': 'str', 'NewExist': 'int64', 'UrbanRural': 'str', 'DisbursementGross': 'float', 'BalanceGross': 'float',
                          'ChgOffPrinGr': 'float', 'GrAppv': 'float', 'SBA_Appv': 'float'})

# Check to see dtypes so far
df_copy.dtypes


# I want to address the NAICS codes next. These refer to the industry each business belongs to, where each additional number is like a more granular filter that pinpoints the specific type of business. We're only really concerned with the general industry for this analysis, so we use the first two digits of each business's NAICS code to determine this. Luckily, a list was provided of the industries corresponding to the first two numbers of the NAICS codes, so we can use this as a reference.

# In[ ]:


# Create a new column with the industry the NAICS code represents
# Selects only the first two numbers of the NAICS code
df_copy['Industry'] = df_copy['NAICS'].astype('str').apply(lambda x: x[:2])

# Maps the approprate industry to each record based on the first two digits of the NAICS code
df_copy['Industry'] = df_copy['Industry'].map({
    '11': 'Ag/For/Fish/Hunt',
    '21': 'Min/Quar/Oil_Gas_ext',
    '22': 'Utilities',
    '23': 'Construction',
    '31': 'Manufacturing',
    '32': 'Manufacturing',
    '33': 'Manufacturing',
    '42': 'Wholesale_trade',
    '44': 'Retail_trade',
    '45': 'Retail_trade',
    '48': 'Trans/Ware',
    '49': 'Trans/Ware',
    '51': 'Information',
    '52': 'Finance/Insurance',
    '53': 'RE/Rental/Lease',
    '54': 'Prof/Science/Tech',
    '55': 'Mgmt_comp',
    '56': 'Admin_sup/Waste_Mgmt_Rem',
    '61': 'Educational',
    '62': 'Healthcare/Social_assist',
    '71': 'Arts/Entertain/Rec',
    '72': 'Accom/Food_serv',
    '81': 'Other_no_pub',
    '92': 'Public_Admin'
})

# Remove records where Industry is NaN (NAICS code was a 0)
df_copy.dropna(subset=['Industry'], inplace=True)


# Alright, so far so good. I take a look at FranchiseCode next. The code itself isn't important to me; I care more about whether or not a business is a franchise or not for this analysis. I create a flag field for this.

# In[ ]:


# Create flag column IsFranchise based on FranchiseCode column
df_copy.loc[(df_copy['FranchiseCode'] <= 1), 'IsFranchise'] = 0
df_copy.loc[(df_copy['FranchiseCode'] > 1), 'IsFranchise'] = 1


# Next I look at some of the fields that are considered flags already but aren't necessarily in a useable format right now. These include the NewExist, RevLineCr, LowDoc, and MIS_Status fields.

# In[ ]:


# NewExist
# Makesure NewExist has only 1s and 2s; Remove records where NewExist isn't 1 or 2
df_copy['NewExist'].unique()


# In[ ]:


# Keep records where NewExist == 1 or 2
df_copy = df_copy[(df_copy['NewExist'] == 1) | (df_copy['NewExist'] == 2)]

# Create NewBusiness field where 0 = Existing business and 1 = New business; based on NewExist field
df_copy.loc[(df_copy['NewExist'] == 1), 'NewBusiness'] = 0
df_copy.loc[(df_copy['NewExist'] == 2), 'NewBusiness'] = 1


# In[ ]:


# RevLineCr and LowDoc
# Double check RevLineCr and LowDoc unique values
df_copy['RevLineCr'].unique()


# In[ ]:


df_copy['LowDoc'].unique()


# In[ ]:


# Remove records where RevLineCr != 'Y' or 'N' and LowDoc != 'Y' or 'N'
df_copy = df_copy[(df_copy['RevLineCr'] == 'Y') | (df_copy['RevLineCr'] == 'N')]
df_copy = df_copy[(df_copy['LowDoc'] == 'Y') | (df_copy['LowDoc'] == 'N')]

# RevLineCr and LowDoc: 0 = No, 1 = Yes
df_copy['RevLineCr'] = np.where(df_copy['RevLineCr'] == 'N', 0, 1)
df_copy['LowDoc'] = np.where(df_copy['LowDoc'] == 'N', 0, 1)

# Check that it worked
print(df_copy['RevLineCr'].unique())
print(df_copy['LowDoc'].unique())


# In[ ]:


# MIS_Status
# Make Default target field based on MIS_Status where P I F = 0 and CHGOFF = 1 so we can see what features are prevalant in a defaulted loan
df_copy['Default'] = np.where(df_copy['MIS_Status'] == 'P I F', 0, 1)
df_copy['Default'].value_counts()


# Now that the flag fields have been addressed, let's tackle the date fields.

# In[ ]:


# Convert ApprovalDate and DisbursementDate columns to datetime values
# ChgOffDate not changed to datetime since it is not of value and will be removed later
df_copy[['ApprovalDate', 'DisbursementDate']] = df_copy[['ApprovalDate', 'DisbursementDate']].apply(pd.to_datetime)


# One metric I thought of that would be interesting to see is the number of days it took from the approval of the loan to the actual disbursement of the funds. My hypothesis is that the timing at which the funds were received could have a negative relationship with a business's ability to repay a loan, whereas the longer it took to receive funds, the more difficult it would be to pay off the loan. In my experience as a Credit Analyst, there were a number of businesses that needed loan funding urgently to help the business stay afloat. I'm sure this would vary by industry however.

# In[ ]:


# Create DaysToDisbursement column which calculates the number of days passed between DisbursementDate and ApprovalDate
df_copy['DaysToDisbursement'] = df_copy['DisbursementDate'] - df_copy['ApprovalDate']

# Change DaysToDisbursement from a timedelta64 dtype to an int64 dtype
# Converts series to str, removes all characters after the space before 'd' in days for each record, then changes the dtype to int
df_copy['DaysToDisbursement'] = df_copy['DaysToDisbursement'].astype('str').apply(lambda x: x[:x.index('d') - 1]).astype('int64')


# In[ ]:


# Create DisbursementFY field for time selection criteria later
df_copy['DisbursementFY'] = df_copy['DisbursementDate'].map(lambda x: x.year)


# Another metric I was interested in exploring is whether or not the bank servicing the loan was in the same state that the business was located. My assumption is that it would be more difficult to service a loan for a business in another state and that this could have a negative impact on a business's ability to repay the loan.

# In[ ]:


# Create StateSame flag field which identifies where the business State is the same as the BankState
df_copy['StateSame'] = np.where(df_copy['State'] == df_copy['BankState'], 1, 0)


# The next field I decided to create relates to the amount of the loan the SBA guaranteed. This is a unique feature SBA loans have where the SBA will 'guaranty' a percentage of the loan in the event of a loss. For example if a business took out a 500,000 loan and the SBA guaranteed 50%, if the business was unable to repay 200,000 of the loan the SBA would cover 100,000 of that loss. This makes these loans very attractive to small businesses because it mitigates their risk, but it also increases the risk for the SBA. This is why an analysis like this is important! These loans are typically guaranteed on a percentage basis rather than a specified dollar amount, so I create a field to represent this rather than the guaranteed amount provided in the original dataset.

# In[ ]:


# Create SBA_AppvPct field since the guaranteed amount is based on a percentage of the gross loan amount rather than dollar amount in most situations
df_copy['SBA_AppvPct'] = df_copy['SBA_Appv'] / df_copy['GrAppv']


# I wanted to look at whether the loan amount disbursed was equal to the full amount approved, so I added that feature as well.

# In[ ]:


# Create AppvDisbursed flag field signifying if the loan amount disbursed was equal to the full amount approved
df_copy['AppvDisbursed'] = np.where(df_copy['DisbursementGross'] == df_copy['GrAppv'], 1, 0)


# Now that we've done a lot of formatting to the data, let's make sure the data types are still correct.

# In[ ]:


df_copy.dtypes


# In[ ]:


# Format dtypes where necessary after feature engineering
df_copy = df_copy.astype({'IsFranchise': 'int64', 'NewBusiness': 'int64'})
df_copy.dtypes


# Now that each field we care about appears to have the correct data type, we can remove the fields that won't provide much value to the analysis. These are as follows:
# * LoanNr_ChkDgt and Name - provides no value to the actual analysis
# * City and Zip - each have a large number of unique values, and my assumption is that it is not likely either would have any particularly significant values 
# * Bank - Name of the bank shouldn't matter for analysis, however this could potentially be used when revisiting this analysis to determine the asset size of the bank servicing the loan
# * ChgOffDate - only applies when a loan is charged off and isn't relevant to the analysis
# * NAICS - replaced by Industry
# * NewExist - replaced by NewBusiness flag field
# * FranchiseCode - replaced by IsFranchise flag field
# * ApprovalDate and DisbursementDate - hypothesis that DaysToDisbursement will be more valueable
# * SBA_Appv - guaranteed amount is based on percentage of gross loan amount, not dollar amount typically
# * MIS_Status - Default field replaces this as the target field

# In[ ]:


df_copy.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode',
                      'ChgOffDate', 'DisbursementDate', 'BalanceGross', 'ChgOffPrinGr', 'SBA_Appv', 'MIS_Status'], inplace=True)


# In[ ]:


# Verify all null values are removed from data
df_copy.isnull().sum()


# In[ ]:


# Check the shape of the data as it stands
df_copy.shape


# The last couple of features I wanted to look at are whether a loan was backed by Real Estate, and whether a loan was active during the Great Recession (2007-2009). These were both mentioned in the document which describes the dataset and how it was used for educatoinal purposes, and I think they will be very important features to consider.
# 
# To determine whether a loan was backed by Real Estate, I made a flag that signifies if the loan term is >= 20 years, as real estate-backed loans are typically at least this long since the loan term is usually tied to the useful life of the assets used for collateral. Unfortunately there's no way to know this for sure since it is not included explicitely in the data.
# 
# For loans active during the Great Recession, I created a flag for loans where the Great Recession (2007-2009) between DisbursementFY and DisbursementFY plus the loan term (in years).

# In[ ]:


# Field for loans backed by Real Estate (loans with a term of at least 20 years)
df_copy['RealEstate'] = np.where(df_copy['Term'] >= 240, 1, 0)

# Field for loans active during the Great Recession (2007-2009)
df_copy['GreatRecession'] = np.where(((2007 <= df_copy['DisbursementFY']) & (df_copy['DisbursementFY'] <= 2009)) | 
                                     ((df_copy['DisbursementFY'] < 2007) & (df_copy['DisbursementFY'] + (df_copy['Term']/12) >= 2007)), 1, 0)


# When it comes to the time period for the records used in the analysis and modeling later, I think the document provided with the dataset had good rationale. This rationale is listed in the 3.3 Time Period section of the document, but to summarize, the emphasis was placed on default rates of loans disbursed through 2010. They wanted to account for the Great Recession and restrict the time frame to loans by excluding those disbursed after 2010 since the loan term is typically 5 years or more.
# 
# I wanted to adopt this for my analysis as well, so I set a selection criteria for loans with a disbursement date prior to 2010.

# In[ ]:


# Select only records with a disbursement year through 2010
df_copy = df_copy[df_copy['DisbursementFY'] <= 2010]

# Check how many records remain
df_copy.shape


# ### Data Analysis
# Let's take a break now that we have done some data cleaning and formatting to see what we're dealing with now.

# In[ ]:


df_copy.describe(include=['object', 'float', 'int'])


# This shows some interesting information for the analysis, including:
# * The average loan term is ~94 months with a standard deviation of ~69 months, suggesting the loan terms are pretty spread out; Max loan term of 527 months could suggest some outliers in the data
# * The average number of employees is about 9.8 with 75% of of businesses having 9 or less employees, suggesting NoEmp is very left skewed; Similar situations for created and retained jobs
# * The mean for flag fields essentially shows a percentage, so roughly 42% of loans in the sample are revolving lines of credit and about 6% of loans were a part of the Low Doc program
# * Average gross loan disbursement was ~166,000 with 75% of loans being less than 188,000, suggesting left skewness again
# * About 77.8% of loans in the sample were paid in full
# * Only 3% of businesses were franchised; About 26% of loan applicants were considered new businesses.
# * The average days to loan disbursement was 109; The min was -3,614, suggesting at least one error in the data (since that's ~301 years)
# * Approximately 45.4% of loans were serviced by banks in the same state as the applying business
# * The average percentage of SBA loan guaranteed amount was 65.4%
# * About 11.2% of the loans backed by real estate per my assumptions
# * About 73.4% of the loans in the sample were active at some point during the Great Recession
# 
# After reviewing these details, there are a few more things I want to add to aid in this analysis. One of these things is to create a flag that signifies if the disbursed amount was greater than what was actually approved. I think this would have interesting implications because the disbursement of extra funds could suggest that the business was at greater risk of default. This could be correlated to revolving lines of credit where the business continually draws and pays down a balance however.

# In[ ]:


# Create flag to signify if a larger amount was disbursed than what the Bank had approved
# Likely RevLineCr?
df_copy['DisbursedGreaterAppv'] = np.where(df_copy['DisbursementGross'] > df_copy['GrAppv'], 1, 0)


# I also wanted to remove records with a negative DaysToDisbursement under the assumption that loan funds would not be disbursed until they were approved.

# In[ ]:


# Remove records with loans disbursed prior to being approved
df_copy = df_copy[df_copy['DaysToDisbursement'] >= 0]

# Check how many records are left
df_copy.shape


# In[ ]:


df_copy.describe(include=['object', 'float', 'int'])


# ### Data Visualization
# Although there are likely some potential outliers that could be explored, I want to start looking at the data from a new perspective. Before we go too far however, let's take a look at a correlation matrix to see the relationships among the features.

# In[ ]:


# Correlation Matrix
cor_fig, cor_ax = plt.subplots(figsize=(15, 10))
corr_matrix = df_copy.corr()
cor_ax = sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=30, horizontalalignment='right', fontsize=8)
plt.yticks(fontsize=8)

plt.show()


# Some notable correlations:
# * GrAppv & DisbursementGross, Positive - Makes sense that in most situations, the amount disbursed is close to what was approved
# * DisbursedGreaterAppv & AppvDisbursed, Negative - Also makes sense since when the disbursed amount is greater than approved, the disbursed amount is then not equal to the approved amount
# * RevLineCr & DisbursedGreaterAppv, Positive - Due to the nature of revolving lines of credit (think of it like a credit card for businesses where the business can draw funds with a limit, pays it off when able, and then draw more funds again), this makes sense that over time more funds are used then the limit set for the loan
# * DisbursementFY & ApprovalFY, Positive - More often than not, the funds will be disbursed in the same year they are approved
# * AppvDisbursed & RevLineCr, Negative - Typically, based on my experience underwriting loans as a Credit Analyst, the limit for a line of credit is lower than a term loan on average since the business can continually draw funds from the line of credit when needed after paying off the balance, which would explain the negative relationship.
# * SBA_AppvPct & RevLineCr, Negative - SBA lines of credit can still be eligible for guarantees, however the guarantee percentage is dependant on the size of the loan.  Although this doesn't quite explain the negative relationship between SBA guarantee percentage and a loan being RevLineCr, what could is the type of SBA loan program used for the loan application. The most common used are the SBA 7(a) and SBA Express programs:
# 
#     1. SBA 7(a) program: Provides an 85% guarantee on loan amounts up to 150,000 and a 75% guarantee on loan amounts over 150,000, but require more paperwork from the SBA and the eligibility decision is typically made by the SBA; Turn around time of 5-10 days
#     2. SBA Express: Max loan amount of 350,000 with a guarantee amount of 50%, however the forms required are the lender's own forms and the eligibility decision is made by the lender rather than the SBA; Turn time of about 36 hours
#     
#     Although this dataset unfortunately doesn't contain information on the loan program used for each loan, my assumption is that most businesses elected to use the SBA Express program for revolving lines of credit because it's easier to get loan approval and the turn time is much quicker, meaning the loan guarantee amounts for revolving lines of credit would typically be less.
#     
# Okay, let's look at some graphs of the data. Some ideas that come to mind are:
# * Total/Average disbursed loan amount by industry
# * Average days to disbursement by industry
# * Number of paid in full and defaulted loans by industry
# * Number of paid in full and defaulted loans by ApprovalFY
# * Number of paid in full and defaulted loans by State
# * Percentage of defaulted loans backed by Real Estate
# * Percentage of defaulted loans active during the Great Recession

# In[ ]:


# Total/Average disbursed loan amount by industry
# Create a groupby object on Industry for use in visualization
industry_group = df_copy.groupby(['Industry'])

# Data frames based on groupby by Industry looking at aggregate and average values
df_industrySum = industry_group.sum().sort_values('DisbursementGross', ascending=False)
df_industryAve = industry_group.mean().sort_values('DisbursementGross', ascending=False)

# Establish figure for placing bar charts side-by-side
fig = plt.figure(figsize=(25, 5))

# Add subplots to figure to build 1x2 grid and specify position of each subplot
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# Bar chart 1 = Gross SBA Loan Disbursement by Industry
ax1.bar(df_industrySum.index, df_industrySum['DisbursementGross'] / 1000000000)
ax1.set_xticklabels(df_industrySum.index, rotation=30, horizontalalignment='right', fontsize=10)

ax1.set_title('Gross SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax1.set_xlabel('Industry')
ax1.set_ylabel('Gross Loan Disbursement (Billions)')

# Bar chart 2 = Average SBA Loan Disbursement by Industry
ax2.bar(df_industryAve.index, df_industryAve['DisbursementGross'])
ax2.set_xticklabels(df_industryAve.index, rotation=30, horizontalalignment='right', fontsize=10)

ax2.set_title('Average SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax2.set_xlabel('Industry')
ax2.set_ylabel('Average Loan Disbursement')

plt.show()


# Some notes:
# * Retail trade and Manufacturing industries had significantly more loan funds distributed to them during the sample period compared to other industries
# * Although the Agriculture, forestry, fishing and hunting, Mining, quarrying, and oil and gas extraction, and Management of companies and enterprises industries had a small amount of total loan funds distributed to them during this time relative to most other industries, they had the highest average loan amount compared to other industries; This suggests they had a small number of large loans

# In[ ]:


# Average days to disbursement by industry
fig2, ax = plt.subplots(figsize=(15, 5))

ax.bar(df_industryAve.index, df_industryAve['DaysToDisbursement'].sort_values(ascending=False))
ax.set_xticklabels(df_industryAve['DaysToDisbursement'].sort_values(ascending=False).index, rotation=35,
                   horizontalalignment='right', fontsize=10)

ax.set_title('Average Days to SBA Loan Disbursement by Industry from 1984-2010', fontsize=15)
ax.set_xlabel('Industry')
ax.set_ylabel('Average Days to Disbursement')

plt.show()


# Notes:
# * Interestingly, some of the industries with the highest average loan amount also had the highest number of days to disbursement of funds, including the Agriculture, forestry, fishing and hunting, and Management of companies and enterprises industries

# In[ ]:


# Paid in full and defaulted loans
fig3 = plt.figure(figsize=(15, 10))

ax1a = plt.subplot(2, 1, 1)
ax2a = plt.subplot(2, 1, 2)

# Function for creating stacked bar charts grouped by desired column
# df = original data frame, col = x-axis grouping, stack_col = column to show stacked values
# Essentially acts as a stacked histogram when stack_col is a flag variable
def stacked_setup(df, col, axes, stack_col='Default'):
    data = df.groupby([col, stack_col])[col].count().unstack(stack_col)
    data.fillna(0)

    axes.bar(data.index, data[1], label='Default')
    axes.bar(data.index, data[0], bottom=data[1], label='Paid in full')


# Number of Paid in full and defaulted loans by industry
stacked_setup(df=df_copy, col='Industry', axes=ax1a)
ax1a.set_xticklabels(df_copy.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default').index,
                     rotation=35, horizontalalignment='right', fontsize=10)

ax1a.set_title('Number of PIF/Defaulted Loans by Industry from 1984-2010', fontsize=15)
ax1a.set_xlabel('Industry')
ax1a.set_ylabel('Number of PIF/Defaulted Loans')
ax1a.legend()

# Number of Paid in full and defaulted loans by State
stacked_setup(df=df_copy, col='State', axes=ax2a)

ax2a.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
ax2a.set_xlabel('State')
ax2a.set_ylabel('Number of PIF/Defaulted Loans')
ax2a.legend()

plt.tight_layout()
plt.show()


# In[ ]:


# Check Default percentage by Industry
def_ind = df_copy.groupby(['Industry', 'Default'])['Industry'].count().unstack('Default')
def_ind['Def_Percent'] = def_ind[1]/(def_ind[1] + def_ind[0])
def_ind


# In[ ]:


# Check Default percentage by State
def_state = df_copy.groupby(['State', 'Default'])['State'].count().unstack('Default')
def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])
def_state


# Notes:
# * Industries with the highest number of loans during sample period: Retail trade (78,554), Professional, scientific and technical services (47,081) and Construction (47,047)
# * Industries with the highest Default percentage: Finance and Insurance (34.4%), Real Estate and rental leasing (33.8%) and Transportation and warehousing (30.7%)
# * States with the highest number of loans during sample period: California (59,121), New York (33,059) and Texas (28,941)
# * State with the highest Default percentage: Florida (33.8%), Arizona (32.6%) and Nevada (31.6%)

# In[ ]:


# Paid in full and Defaulted loans by DisbursementFY
# Decided to use a stacked area chart here since it's time series data
fig4, ax4 = plt.subplots(figsize=(15, 5))

stack_data = df_copy.groupby(['DisbursementFY', 'Default'])['DisbursementFY'].count().unstack('Default')
x = stack_data.index
y = [stack_data[1], stack_data[0]]

ax4.stackplot(x, y, labels=['Default', 'Paid in full'])
ax4.set_title('Number of PIF/Defaulted Loans by State from 1984-2010', fontsize=15)
ax4.set_xlabel('Disbursement Year')
ax4.set_ylabel('Number of PIF/Defaulted Loans')
ax4.legend(loc='upper left')

plt.show()


# There is a clear increase in loan volume leading up to the peak of the Great Recession, with a subsequent drop in loan volume immediately following that time. Looking at the graph, it appears the default rate of loans increased during that time as well.

# In[ ]:


# Paid in full and defaulted loans backed by Real Estate
fig5 = plt.figure(figsize=(20, 10))

ax1b = fig5.add_subplot(1, 2, 1)
ax2b = fig5.add_subplot(1, 2, 2)

stacked_setup(df=df_copy, col='RealEstate', axes=ax1b)
ax1b.set_xticks(df_copy.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default').index)
ax1b.set_xticklabels(labels=['No', 'Yes'])

ax1b.set_title('Number of PIF/Defaulted Loans backed by Real Estate from 1984-2010', fontsize=15)
ax1b.set_xlabel('Loan Backed by Real Estate')
ax1b.set_ylabel('Number of Loans')
ax1b.legend()

# Paid in full and defaulted loans active during the Great Recession
stacked_setup(df=df_copy, col='GreatRecession', axes=ax2b)
ax2b.set_xticks(df_copy.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default').index)
ax2b.set_xticklabels(labels=['No', 'Yes'])

ax2b.set_title('Number of PIF/Defaulted Loans Active during the Great Recession from 1984-2010', fontsize=15)
ax2b.set_xlabel('Loan Active during Great Recession')
ax2b.set_ylabel('Number of Loans')
ax2b.legend()

plt.show()


# In[ ]:


# Check Default percentage for loans backed by Real Estate
def_re = df_copy.groupby(['RealEstate', 'Default'])['RealEstate'].count().unstack('Default')
def_re['Def_Percent'] = def_re[1]/(def_re[1] + def_re[0])
def_re


# In[ ]:


# Check Default percentage for loans active during the Great Recession
def_gr = df_copy.groupby(['GreatRecession', 'Default'])['GreatRecession'].count().unstack('Default')
def_gr['Def_Percent'] = def_gr[1]/(def_gr[1] + def_gr[0])
def_gr


# The volume of loans backed by real estate was much less than those not backed by real estate (which makes sense that most people aren't willing to take on that much risk), however the default rate is also much less for loans backed by real estate. This is likely because the people and businesses who have their loans backed by real estate have much more skin in the game so they're more willing to do what it takes to pay the debt.
# 
# I thought loans active during the Great Recession would have a noticeably higher default rate than those not active during that time. My assumption is this difference would be more apparent if the focus was on loans disbursed in the few years leading up to the Great Recession, perhaps beginning in 2004 or 2005. 

# In[ ]:


df_copy.dtypes


# # Modeling
# Now that I've done some more exploring of the data, I think it's about time to experiment with some modeling. I know I want to try a Logistic Regression model, and I'd like to try an XGBoost model as well as I know this is currently a popular model to use in Kaggle competitions and generally performs pretty well.
# 
# Something I want to keep in mind when evaluating model performance is that a good accuracy doesn't necessarily mean the model performed well. We need to consider metrics like the precision, recall, and F1-score to ensure we are evaluating model performance based on the 'cost' of the outcomes. For example in this situation, it's better if we predict a loan will default and it doesn't than if we predict a loan will be paid in full and ends up in default. In other words, we want a model that minimizes the number of false negatives (since a 1 in the Default field signifies a 'positive' value in this case). Essentially we want a model that predicts the correct outcome most of the time, but when it gets it wrong it's not as bad.
# 
# But which metric is best for evaluating the model then? Borrowing definitions of precision and recall from Google's Machine Learning Crash Course (found at https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall), precision attempts to answer what proportion of positive identifications was actually correct, whereas recall attempts to answer what proportion of actual positives was identified correctly. The F1-score is a weighted average of precision and recall. General accuracy therefore is better if the outcomes have similar costs, however F1-score should be relied on more heavily if the outcomes have different costs as is the case in this scenario.

# In[ ]:


# One-hot encode categorical data
df_copy = pd.get_dummies(df_copy)

df_copy.head()


# In[ ]:


# Establish target and feature fields
y = df_copy['Default']
X = df_copy.drop('Default', axis=1)

# Scale the feature values prior to modeling
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25)


# In[ ]:


# Initialize model
log_reg = LogisticRegression(random_state=2)

# Train the model and make predictions
log_reg.fit(X_train, y_train)
y_logpred = log_reg.predict(X_val)

# Print the results
print(classification_report(y_val, y_logpred, digits=3))


# We can see here that with the Logistic Regression model, we have a decent accuracy at 87.5%, however the F1-score of 68.4% for defaulted loans does not seem very promising. The precision suggests that the model is correct 78.2% of the time when the loan defaults, and the recall suggests that the model identifies 60.8% of defaulted loans correctly. That means that 39.2% of loans that defaulted were incorrectly classified as loans that would be paid in full, which is NOT very good.
# 
# Let's see if the XGBoost model can do any better.

# In[ ]:


xgboost = XGBClassifier(random_state=2)

xgboost.fit(X_train, y_train)
y_xgbpred = xgboost.predict(X_val)

# Print the results
print(classification_report(y_val, y_xgbpred, digits=3))


# This is MUCH better across the board! Not only do we have a general accuracy of 95.6%, but also the precision, recall and F1-score are all improved by quite a bit. I might go back and try tweaking some of the hyperparameters later, but for now I'm satisfied with this. Let's take a look at some of the most important features.

# In[ ]:


# List the importance of each feature
for name, importance in sorted(zip(X.columns, xgboost.feature_importances_)):
    print(name, "=", importance)


# Some interesting notes from the list of feature importances:
# * The top five features by level of importance are Term, StateSame, ApprovalFY, UrbanRural_1 (Urban), and BankState_NC in that order
# * Industry feature importance - Highest: Healthcare/Social_assist, Lowest: Mgmt_comp, Public_Admin and Utilities
# * A loan being backed by Real Estate surprisingly had zero feature importance; I had also anticipated a loan being active during the Great Recession being in the top five most important features
# * Whether or not a business applying for an SBA loan is a new business did not seem to be very important when determining if the loan would default
# 
# There's plenty more I could dig into in regard to the feature importances, but these are a few of the things I noticed right away, some of which I didn't expect. One thing I wanted to try was seeing if reducing the number of features used to the most important ones would have a positive impact on the model performance, since the current model has a high level of dimensionality.

# In[ ]:


# Build pipeling for feature selection and modeling; SelectKBest defaults to top 10 features
xgb_featimp = XGBClassifier(random_state=2)

pipe = Pipeline(steps=[
    ('feature_selection', SelectKBest()),
    ('model', xgb_featimp)
])

pipe.fit(X_train, y_train)
y_featimppred = pipe.predict(X_val)

print(classification_report(y_val, y_featimppred, digits=3))


# It looks like reducing the number of features, and thereby dimensionality of the data, didn't affect the results too drastically. In fact, this model would likely perform better in a real world test because it is far more generalized. Let's take a look at what features were actually selected then.

# In[ ]:


# List the importance of each feature
for name, importance in sorted(zip(X.columns, xgb_featimp.feature_importances_)):
    print(name, "=", importance)


# Looks like the Term is still the most important by far, but the top five features have certainly changed! Job creation played a much larger role in this model, which is interesting to see. Perhaps onboarding new employees is more expensive, which puts a strain on the business's bottom line and therefore ability to repay the loan. I also find it interesting that the loans which were part of the Low Doc program was selected as a most important feature, however it apparently shows zero importance here.

# # Conclusion and next steps
# According to this analysis, the factor that contributes the most to whether or not a loan goes into default is the length of the term of the loan, where the longer the term is the higher the chance that the loan will go into default. Further analysis could be done in this area, and I may revisit this at a later date and see if binning the loan term length will prove more valuable in modeling. The term of the loan is typically tied to the size of the loan, which was also selected as one of the most important features in determining whether or not a loan will be paid in full. There are a number of other factors that I was unable to consider for this analysis that I believe would be beneficial, such as which of the SBA loan programs each loan fell under. I could certainly go back and address some of the existing outliers and skewness in some of the features, and would also consider looking at other types of models and will be interested to play with the hyperparameters of the models when I look at this again in the future.
# 
# Despite all of the data provided here, there is something else that isn't captured in this data that is arguably the most important and relevant factor in determining the ability of a business to repay the loan: the business owner(s) and the business operations themselves! Although the industry does have some weight in this aspect, the data doesn't include the cash flow of each business, working capital, existing debt they had prior to applying for the SBA loan, etc. The data also can't capture the personality, attitude and drive of a business owner to make the business successful. One of the most important factors we used when underwriting loan applications during my time as a Credit Analyst was the character of the business owner(s). A business can be very successful, but at the end of the day if the owner doesn't want to pay the loan they won't. It's a sad truth, but it happens more often than you'd think.
# 
# If you've read this far, thanks for taking the time to look at my analysis! As mentioned, I will likely revisit this to dig deeper as I sharpen my skillset. I certainly learned a lot working through this, and I hope you're able to take something away from this as well.

# In[ ]:




