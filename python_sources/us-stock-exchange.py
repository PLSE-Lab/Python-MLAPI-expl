#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# **Objective of this sheet is to analyse data of US Stock Exchange **
# 1. Fundamental as well as Stock Market Behaviour of Companies of US from year 2012 to 2015. 
# 2. We also do KDE clustering by deviding the data to test and train data.
# 3. Plot graph of sum of the squared distance between centroid and each member of the cluster.
# 4. Done silhouette analysis on KMeans clustering.
# 5. Doing Fundamental Analysis of US stock Exchange.

# In[ ]:


import pandas as pd

import numpy as np
# 1.2 OS related package
import os
# 1.3 Modeling librray
# 1.3.1 Scale data
from sklearn.preprocessing import StandardScaler
# 1.3.2 Split dataset
from sklearn.model_selection import train_test_split
# 1.3.3 Class to develop kmeans model
from sklearn.cluster import KMeans
# 1.4 Plotting library
import matplotlib.pyplot as plt
import seaborn as sns
# 1.5 How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer

import warnings
warnings.filterwarnings('ignore')


# 2.0 Set your working folder to where data is
os.chdir("../input")
os.listdir()

# 2.1 Read csv file
fundamentals = pd.read_csv("fundamentals.csv")
prices = pd.read_csv("prices.csv")
split = pd.read_csv("prices-split-adjusted.csv")
securities= pd.read_csv("securities.csv")

fundamentals.info()

#2.2 Change the column names and remove special characters
fundamentals_col = {
'Ticker Symbol'						:	'Ticker_Symbol',
'Period Ending'						:	'Period_Ending',
'Accounts Payable'	               			:	'Accounts_Payable',
'Accounts Receivable'	                		:	'Accounts_Receivable',
'After Tax ROE'	                        		:	'After_Tax_ROE',
'Capital Expenditures'          			:	'Capital_Expenditures',
'Capital Surplus'					:	'Capital_Surplus',
'Cash Ratio'						:	'Cash_Ratio',
'Cash and Cash Equivalents'				:	'Cash_and_Cash_Equivalents',
'Changes in Inventories'				:	'Changes_in_Inventories',
'Common Stocks'						:	'Common_Stocks',
'Cost of Revenue'					:	'Cost_of_Revenue',
'Current Ratio'						:	'Current_Ratio',
'Deferred Asset Charges'				:	'Deferred_Asset_Charges',
'Deferred Liability Charges'				:	'Deferred_Liability_Charges',
'Depreciation'						:	'Depreciation',
'Earnings Before Interest and Earnings Before Tax'	:	'Earnings_Before_Interest_and_Earnings_Before_Tax',
'Effect of Exchange Rate'				:	'Effect_of_Exchange_Rate',
'Equity Earnings/Loss Unconsolidated Subsidiary'				:	'Equity_Earnings_Loss_Unconsol_Subsidiary',
'Fixed Assets'						:	'Fixed_Assets',
'Goodwill'						:	'Goodwill',
'Gross Margin'						:	'Gross_Margin',
'Gross Profit'						:	'Gross_Profit',
'Income Tax'						:	'Income_Tax',
'Intangible Assets'					:	'Intangible_Assets',
'Interest Expense'					:	'Interest_Expense',
'Inventory'						:	'Inventory',
'Investments'						:	'Investments',
'Liabilities'						:	'Liabilities',
'Long-Term Debt'					:	'Long-Term_Debt',
'Long-Term Investments'					:	'Long-Term_Investments',
'Minority Interest'					:	'Minority_Interest',
'Misc. Stocks'						:	'Misc._Stocks',
'Net Borrowings'					:	'Net_Borrowings',
'Net Cash Flow'						:	'Net_Cash_Flow',
'Net Cash Flow-Operating'				:	'Net_Cash_Flow-Operating',
'Net Cash Flows-Financing'				:	'Net_Cash_Flows-Financing',
'Net Cash Flows-Investing'				:	'Net_Cash_Flows-Investing',
'Net Income'						:	'Net_Income',
'Net Income Adjustments'				:	'Net_Income_Adjustments',
'Net Income Applicable to Common Shareholders'				:	'Net_Income_Applicable_to_Common_Shareholders',
'Net Income-Cont. Operations'				:	'Net_Income-Cont_Operations',
'Net Receivables'					:	'Net_Receivables',
'Non-Recurring Items'					:	'Non-Recurring_Items',
'Operating Income'					:	'Operating_Income',
'Operating Margin'					:	'Operating_Margin',
'Other Assets'						:	'Other_Assets',
'Other Current Assets'					:	'Other_Current_Assets',
'Other Current Liabilities'				:	'Other_Current_Liabilities',
'Other Equity'						:	'Other_Equity',
'Other Financing Activities'				:	'Other_Financing_Activities',
'Other Investing Activities'				:	'Other_Investing_Activities',
'Other Liabilities'					:	'Other_Liabilities',
'Other Operating Activities'				:	'Other_Operating_Activities',
'Other Operating Items'					:	'Other_Operating_Items',
'Pre-Tax Margin'					:	'Pre-Tax_Margin',
'Pre-Tax ROE'						:	'Pre-Tax_ROE',
'Profit Margin'						:	'Profit_Margin',
'Quick Ratio'						:	'Quick_Ratio',
'Research and Development'				:	'Research_and_Development',
'Retained Earnings'					:	'Retained_Earnings',
'Sale and Purchase of Stock'				:	'Sale_and_Purchase_of_Stock',
'Sales, General and Admin.'				:	'Sales_General_and_Admin',
'Short-Term Debt / Current Portion of Long-Term Debt'				:	'Short-Term_Debt_Current_Portion_of_Long-Term_Debt',
'Short-Term Investments'				:	'Short-Term_Investments',
'Total Assets'						:	'Total_Assets',
'Total Current Assets'					:	'Total_Current_Assets',
'Total Current Liabilities'				:	'Total_Current_Liabilities',
'Total Equity'						:	'Total_Equity',
'Total Liabilities'					:	'Total_Liabilities',
'Total Liabilities & Equity'				:	'Total_Liabilities_Equity',
'Total Revenue'						:	'Total_Revenue',
'Treasury Stock'					:	'Treasury_Stock',
'For Year'						:	'For_Year',
'Earnings Per Share'					:	'Earnings_Per_Share',
'Estimated Shares Outstanding'				:	'Estimated_Shares_Outstanding'
}

fundamentals.rename(columns = fundamentals_col,inplace = True)


# In[ ]:


#2.3 Group data by Ticker Symbols and take a mean of all numeric variables.

grpfundamentals = fundamentals.groupby(['Ticker_Symbol'],as_index = False)
grpfundamentals.groups
gpfundamentals = grpfundamentals.agg({
'Accounts_Payable'	:np.mean,
'Accounts_Receivable'	:np.mean,
'After_Tax_ROE'	:np.mean,
'Capital_Expenditures'	:np.mean,
'Capital_Surplus'	:np.mean,
'Cash_Ratio'	:np.mean,
'Cash_and_Cash_Equivalents'	:np.mean,
'Changes_in_Inventories'	:np.mean,
'Common_Stocks'	:np.mean,
'Cost_of_Revenue'	:np.mean,
'Current_Ratio'	:np.mean,
'Deferred_Asset_Charges'	:np.mean,
'Deferred_Liability_Charges'	:np.mean,
'Depreciation'	:np.mean,
'Effect_of_Exchange_Rate'	:np.mean,
'Equity_Earnings_Loss_Unconsol_Subsidiary'	:np.mean,
'Fixed_Assets'	:np.mean,
'Goodwill'	:np.mean,
'Gross_Margin'	:np.mean,
'Gross_Profit'	:np.mean,
'Income_Tax'	:np.mean,
'Intangible_Assets'	:np.mean,
'Interest_Expense'	:np.mean,
'Inventory'	:np.mean,
'Investments'	:np.mean,
'Liabilities'	:np.mean,
'Long-Term_Debt'	:np.mean,
'Long-Term_Investments'	:np.mean,
'Minority_Interest'	:np.mean,
'Misc._Stocks'	:np.mean,
'Net_Borrowings'	:np.mean,
'Net_Cash_Flow'	:np.mean,
'Net_Cash_Flow-Operating'	:np.mean,
'Net_Cash_Flows-Financing'	:np.mean,
'Net_Cash_Flows-Investing'	:np.mean,
'Net_Income'	:np.mean,
'Net_Income_Adjustments'	:np.mean,
'Net_Income_Applicable_to_Common_Shareholders'	:np.mean,
'Net_Income-Cont_Operations' :np.mean,
'Net_Receivables'	:np.mean,
'Non-Recurring_Items':np.mean,
'Operating_Income':np.mean,
'Operating_Margin':np.mean,
'Other_Assets':np.mean,
'Other_Current_Assets':np.mean,
'Other_Current_Liabilities':np.mean,
'Other_Equity':np.mean,
'Other_Financing_Activities':np.mean,
'Other_Investing_Activities':np.mean,
'Other_Liabilities':np.mean,
'Other_Operating_Activities':np.mean,
'Other_Operating_Items':np.mean,
'Pre-Tax_Margin':np.mean,
'Pre-Tax_ROE':np.mean,
'Profit_Margin':np.mean,
'Quick_Ratio':np.mean,
'Research_and_Development':np.mean,
'Retained_Earnings':np.mean,
'Sale_and_Purchase_of_Stock':np.mean,
'Sales_General_and_Admin':np.mean,
'Short-Term_Debt_Current_Portion_of_Long-Term_Debt':np.mean,
'Short-Term_Investments':np.mean,
'Total_Assets':np.mean,
'Total_Current_Assets':np.mean,
'Total_Current_Liabilities':np.mean,
'Total_Equity':np.mean,
'Total_Liabilities':np.mean,
'Total_Liabilities_Equity':np.mean,
'Total_Revenue':np.mean,
'Treasury_Stock':np.mean,
'Earnings_Per_Share':np.mean,
'Estimated_Shares_Outstanding':np.mean,
})


# In[ ]:


#Drop all rows having null values

gpfundamentals.dropna(inplace=True)

y = gpfundamentals['Ticker_Symbol'].values
gpfundamentals.drop(columns = ['Ticker_Symbol'], inplace = True)


# In[ ]:


# Scale Data using Standard Scalar
ss = StandardScaler()     # Create an instance of class
ss.fit(gpfundamentals)                # Train object on the data
X = ss.transform(gpfundamentals)      # Transform data
X[:5, :]                  # See first 5 rows

#Split Dataset into Train and Test

X_train,X_test,_,y_test =train_test_split(X,y,test_size = .25)

#Examine the results getting shape of X_Train and X_test

X_train.shape
X_test.shape

# Examine the Clusters using K Means Clustering
clf = KMeans(n_clusters = 2)

#Train the Object Over Data
clf.fit(X_train)

# 5.3 So what are our clusters?
clf.cluster_centers_
clf.cluster_centers_.shape         
clf.labels_                        
clf.labels_.size                   
clf.inertia_ 

# Now Interpret using silhoutte score
silhouette_score(X_train, clf.labels_) 

# Now make predictions over Test Data and Check Accuracy
y_pred = clf.predict(X_test)
y_pred

# How Good is Your Prediction

np.sum(y_pred == y_test)/y_test.size


#  Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # 7.1.1 How many clusters?
    n_clusters = i+1
    # 7.1.2 Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    # 7.1.3 Train the kmeans object over data
    clf1.fit(X_train)
    # 7.1.4 Store the value of inertia in sse
    sse.append(clf1.inertia_ )


# **Scatter Plot to check the behaviour of Cluster whether clusters are distinguishable.**

# In[ ]:


# Are Clusters Distinguishable

dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)


# **Scatter Plot Analysis **- Scatter Plot analysis indicates one cluster is totally dense and other one is totally distributed.**

# **SSE Line Plot of US Fundamental Data**

# In[ ]:


#  Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # 7.1.1 How many clusters?
    n_clusters = i+1
    # 7.1.2 Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    # 7.1.3 Train the kmeans object over data
    clf1.fit(X_train)
    # 7.1.4 Store the value of inertia in sse
    sse.append(clf1.inertia_ )
#  Plot the line now
sns.lineplot(range(1, 11), sse)


# **SSE Plot Analysis -** As we seen that SSE Clustering plot elbow located at 2 means 2 clusters are optimum choice for this dataset.

# **Silhoutte plot of US Companies Fundamentals**

# In[ ]:


# Silhoutte plot of this Data

visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()   


# **Silhauette Plot analysis** displays cluster 1 is having a negative Silhoutte score indicates these points assigned to wrong cluster
# whereas cluster zero having a excellent Silhoutte score.

# **Fundamental Analysis of US Stock Exchange**

# In[ ]:


# Merging of Tables Fundamentals,Security and in order to do fundamental analysis.

#Rename Columns of Security Table to merge with fundamentals
securities_col = {
'Ticker symbol':'Ticker_Symbol',
'SEC filings':'SEC_filings',
'GICS Sector' : 'GICS_Sector',
'GICS Sub Industry' : 'GICS_Sub_Industry'
}

securities.rename(columns = securities_col,inplace = True)

funda_analysis  = securities.merge(fundamentals,on=['Ticker_Symbol'],how = 'left')

#Dropping of some unuseful columns from funda_analysis table
funda_analysis.dtypes

funda_analysis.drop(['SEC_filings','Address of Headquarters','Date first added','CIK'], axis = 1)

#Grouping Data Sector Wise for analysis
grpfunda_analysis = funda_analysis.groupby(['GICS_Sector'],as_index = False)
grpfunda_analysis.groups
gpfunda_analysis = grpfunda_analysis.agg({
'Accounts_Payable'	:np.mean,
'Accounts_Receivable'	:np.mean,
'After_Tax_ROE'	:np.mean,
'Capital_Expenditures'	:np.mean,
'Capital_Surplus'	:np.mean,
'Cash_Ratio'	:np.mean,
'Cash_and_Cash_Equivalents'	:np.mean,
'Changes_in_Inventories'	:np.mean,
'Common_Stocks'	:np.mean,
'Cost_of_Revenue'	:np.mean,
'Current_Ratio'	:np.mean,
'Deferred_Asset_Charges'	:np.mean,
'Deferred_Liability_Charges'	:np.mean,
'Depreciation'	:np.mean,
'Effect_of_Exchange_Rate'	:np.mean,
'Equity_Earnings_Loss_Unconsol_Subsidiary'	:np.mean,
'Fixed_Assets'	:np.mean,
'Goodwill'	:np.mean,
'Gross_Margin'	:np.mean,
'Gross_Profit'	:np.mean,
'Income_Tax'	:np.mean,
'Intangible_Assets'	:np.mean,
'Interest_Expense'	:np.mean,
'Inventory'	:np.mean,
'Investments'	:np.mean,
'Liabilities'	:np.mean,
'Long-Term_Debt'	:np.mean,
'Long-Term_Investments'	:np.mean,
'Minority_Interest'	:np.mean,
'Misc._Stocks'	:np.mean,
'Net_Borrowings'	:np.mean,
'Net_Cash_Flow'	:np.mean,
'Net_Cash_Flow-Operating'	:np.mean,
'Net_Cash_Flows-Financing'	:np.mean,
'Net_Cash_Flows-Investing'	:np.mean,
'Net_Income'	:np.mean,
'Net_Income_Adjustments'	:np.mean,
'Net_Income_Applicable_to_Common_Shareholders'	:np.mean,
'Net_Income-Cont_Operations' :np.mean,
'Net_Receivables'	:np.mean,
'Non-Recurring_Items':np.mean,
'Operating_Income':np.mean,
'Operating_Margin':np.mean,
'Other_Assets':np.mean,
'Other_Current_Assets':np.mean,
'Other_Current_Liabilities':np.mean,
'Other_Equity':np.mean,
'Other_Financing_Activities':np.mean,
'Other_Investing_Activities':np.mean,
'Other_Liabilities':np.mean,
'Other_Operating_Activities':np.mean,
'Other_Operating_Items':np.mean,
'Pre-Tax_Margin':np.mean,
'Pre-Tax_ROE':np.mean,
'Profit_Margin':np.mean,
'Quick_Ratio':np.mean,
'Research_and_Development':np.mean,
'Retained_Earnings':np.mean,
'Sale_and_Purchase_of_Stock':np.mean,
'Sales_General_and_Admin':np.mean,
'Short-Term_Debt_Current_Portion_of_Long-Term_Debt':np.mean,
'Short-Term_Investments':np.mean,
'Total_Assets':np.mean,
'Total_Current_Assets':np.mean,
'Total_Current_Liabilities':np.mean,
'Total_Equity':np.mean,
'Total_Liabilities':np.mean,
'Total_Liabilities_Equity':np.mean,
'Total_Revenue':np.mean,
'Treasury_Stock':np.mean,
'Earnings_Per_Share':np.mean,
'Estimated_Shares_Outstanding':np.mean,
})

# We are doing fundamental Analysis of Various Sectors of US Companies

columns = ["Current_Ratio","Goodwill","Operating_Income","Net_Income","Net_Cash_Flow","Profit_Margin","Pre-Tax_ROE","Earnings_Per_Share","Research_and_Development",]


fig =plt.figure(figsize= (20,20))
for i in range(len(columns)):
    plt.subplot(3,3,i+1)
    chart =sns.barplot(x="GICS_Sector",y=gpfunda_analysis[columns[i]],data = gpfunda_analysis)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
  
    
    


# **current ratio** is a liquidity ratio that measures a company's ability to pay short-term obligations 
# or those due within one year. 
# Best Performer - Information Technology and Health care sector is very sound to meet current obligations.
# Worst Performer - Utilities Sector is worst performer.
# 
# **Goodwill** - Goodwill is an intangible asset it indicates intrinsic value of company. Best performer in this
# parameter is Telecommunication services.
# 
# **Operating income**- is an accounting figure that measures the amount of profit realized from a business's 
# operations, after deducting operating expenses such as wages, depreciation, and cost of goods sold (COGS).
# Leader again is Telecommunication Services.
# 
# **Net income** is the amount of accounting profit a company has left over after paying off all its expenses. 
# Leader again in this Telecommunication Services.
# 
# **Net cash flow** is nothing but the difference between cash inflows and outflows of a business. Companies with 
# positive cash inflows are cash rich companies. Leader in this analysis is Information Technology but surprisingly
# Telecommunication sector is laggard in this having negaive cash flows.
# 
# 
# **Profit Margin** - Indicates how sustainable a business is means how much profit generated it is calculated
#  ratio of a company's profit (sales minus all expenses) divided by its revenue. Energy and Real State sector
# having a very good profit margin sector whereas telecom is laggard in this.
# 
# **Pre Tax ROE** - As per warran Buffet Return on Equity is a best indicator to analyse company fundamentals 
# calculated by dividing net income by shareholders' equity. In this leader is Consumer Staples is a leader
# and Real estate Telecommunication and Utilities are laggards.
# 
# **Earning Per Share** - Earnings per share (EPS) is a company's net profit divided by the number of 
# common shares it has outstanding. EPS indicates how much money a company makes for each share. Leaders in
# this sector is Materials and Consumer Discritionary.
# 
# **Research and Development** - R&D is a process by which companies acquire new knowledge means sectors having 
# investment in more R&D are futuristic sectors. Leaders in this Information Technology and Health Care.
# 
# 
# 

# **Final Conclusion** - In this notesheet we analysed cluster analysis of US stock market data using Silhoutte plot 
# analysis our observation in this SSE arm elbow at 2 means optimum cluster to analyse this data is 2. 
# Cluster 1 having negative Silhoutte score indicates these points assigned to wrong cluster whereas cluster
# zero having excellent data.
# 
# During Fundamental analysis we observe sectors on various parameters our findings are Telecommunication, Information Technology, Health Care, Consumer Staples
# are the best sectors during year 2012 to 2016 for US stock market as per fundamental analysis.
# 
# Shortly we also analyse the Technical Analysis i.e. price behaviour of this analysis of US stock market and
# analyse the price behaviour with reference to fundamental analysis. So be connected with this sheet and find
# what's next ......
# 
# **If you like my effort please UPVOTE me that encourages me to do better.**
# 
# 
# 
