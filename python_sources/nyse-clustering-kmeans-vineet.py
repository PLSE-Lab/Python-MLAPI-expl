#!/usr/bin/env python
# coding: utf-8

# # New York Stock Exchange : Data Vizualisation, Clustering, KMeans..by Vineet Vivek

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.io as pio;
pio.renderers.default='notebook'
import os
from sklearn.preprocessing import StandardScaler
# Split dataset
from sklearn.model_selection import train_test_split
# Develop kmeans model
from sklearn.cluster import KMeans
# How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ### Accessing Files

# In[ ]:


os.chdir('/kaggle/input/nyse/')


# In[ ]:


df=pd.read_csv("fundamentals.csv")


# In[ ]:


os.listdir()


# ### Data..

# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


new_col_names={'Ticker Symbol':'Ticker_Symbol',
               'Period Ending':'Period_Ending',
               'Accounts Payable':'Accounts_Payable',
               'Accounts Receivable':'Accounts_Receivable',
               'Add\'l income/expense items':'Addnl_IncomeExpense_Items',
               'After Tax ROE':'After_Tax_ROE',
               'Capital Expenditures': 'Capital_Expenditures',
               'Capital Surplus':'Capital_Surplus',
               'Cash Ratio':'Cash_Ratio',
               'Cash and Cash Equivalents':'Cash_and_Cash_Equivalents',
               'Changes in Inventories':'Changes_in_Inventories',
               'Common Stocks':'Common_Stocks',
               'Cost of Revenue':'Cost_of_Revenue',
               'Current Ratio':'Current_Ratio',
               'Deferred Asset Charges':'Deferred_Asset_Charges',
               'Deferred Liability Charges':'Deferred_Liability_Charges',
               'Depreciation':'Depreciation',
               'Earnings Before Interest and Tax':'Earnings_Before_Interest_and_Tax',
               'Earnings Before Tax':'Earnings_Before_Tax',
               'Effect of Exchange Rate':'Effect_Of_Exchange_Rate',
               'Equity Earnings/Loss Unconsolidated Subsidiary':'Equity_EarningsnLoss_Unconsolidated_Subsidiary',
               'Fixed Assets':'Fixed_Assets',
               'Goodwill':'Goodwill',
               'Gross Margin':'Gross_Margin',
               'Gross Profit':'Gross_Profit',
               'Income Tax':'Income_Tax',
               'Intangible Assets':'Intangible_Assets',
               'Interest Expense':'Interest_Expense',
                'Inventory':'Inventory',
               'Investments':'Investments',
               'Liabilities':'Liabilities', 
               'Long-Term Debt':'LongTerm_Debt', 
               'Long-Term Investments':'LongTerm_Investments',
               'Minority Interest':'Minority_Interest', 
               'Misc. Stocks':'Misc_Stocks', 
               'Net Borrowings':'Net_Borrowings', 
               'Net Cash Flow':'Net_Cash_Flow',
                'Net Cash Flow-Operating':'Net_Cash_FlowOperating',
               'Net Cash Flows-Financing':'Net_Cash_FlowsFinancing',
               'Net Cash Flows-Investing':'Net_Cash_FlowsInvesting',
               'Net Income':'Net_Income', 
               'Net Income Adjustments':'Net_Income_Adjustments',
               'Net Income Applicable to Common Shareholders':'Net_Income_Applicable_To_Common_Shareholders',
               'Net Income-Cont. Operations':'Net_IncomeCont_Operations',
               'Net Receivables':'Net_Receivables',
               'Non-Recurring Items':'NonRecurring_Items',
               'Operating Income':'Operating_Income', 
               'Operating Margin':'Operating_Margin',
               'Other Assets':'Other_Assets',
               'Other Current Assets':'Other_Current_Assets',
               'Other Current Liabilities':'Other_Current_Liabilities',
               'Other Equity':'Other_Equity',
               'Other Financing Activities':'Other_Financing_Activities',
               'Other Investing Activities':'Other_Investing_Activities',
               'Other Liabilities':'Other_Liabilities',
               'Other Operating Activities':'Other_Operating_Activities',
               'Other Operating Items':'Other_Operating_Items',
               'Pre-Tax Margin':'PreTax_Margin',
               'Pre-Tax ROE':'PreTax_ROE',
                'Profit Margin':'Profit_Margin',
               'Quick Ratio':'Quick_Ratio',
               'Research and Development':'Research_and_Development',
               'Retained Earnings':'Retained_Earnings', 
               'Sale and Purchase of Stock':'Sale_and_Purchase_of_Stock',
                'Sales, General and Admin.':'Sales_General_and_Admin',
               'Short-Term Debt / Current Portion of Long-Term Debt':'ShortTerm_Debt_Current_Portion_of_LongTerm_Debt',
               'Short-Term Investments':'ShortTerm_Investments',
               'Total Assets':'Total_Assets',
               'Total Current Assets':'Total_Current_Assets',
               'Total Current Liabilities':'Total_Current_Liabilities', 
               'Total Equity':'Total_Equity', 
               'Total Liabilities':'Total_Liabilities',
               'Total Liabilities & Equity':'Total_Liabilities_Equity',
               'Total Revenue':'Total_Revenue',
               'Treasury Stock':'Treasury_Stock',
               'For Year':'For_Year',
               'Earnings Per Share':'Earnings_Per_Share', 
               'Estimated Shares Outstanding':'Estimated_Shares_Outstanding'
              }


# In[ ]:


df.rename(new_col_names,inplace=True,axis=1)


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


grouped = df.groupby('Ticker_Symbol')


# In[ ]:


grouped.describe()


# In[ ]:


df.mean()


# ### Data Visualization

# In[ ]:


plt.figure(figsize=(10,10))
sns.distplot(df.Gross_Profit)


# In[ ]:


sns.distplot(df.Earnings_Per_Share)


# In[ ]:


sns.distplot(df.Current_Ratio)


# In[ ]:


sns.boxplot(x='Operating_Income',
           y='Gross_Profit',
           data=df)


# In[ ]:


sns.jointplot(df.Liabilities,df.Goodwill,kind="reg")


# In[ ]:


sns.catplot(x='Ticker_Symbol',y='Earnings_Per_Share', kind='bar',data=df)


# In[ ]:


sns.catplot(x='Sale_and_Purchase_of_Stock',y='Gross_Profit', kind='bar',data=df)


# In[ ]:


sns.relplot(x='Total_Liabilities', y='Total_Revenue',kind='scatter',data=df)


# ### Clustering..

# In[ ]:


df.shape


# In[ ]:


df=df.dropna()


# In[ ]:


df.shape


# In[ ]:


num_columns=df.select_dtypes(include=['int64','float64']).copy()


# In[ ]:


num_columns.describe()


# In[ ]:


ss = StandardScaler()


# In[ ]:


ss.fit(num_columns)


# In[ ]:


X=ss.transform(num_columns)


# In[ ]:


np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )


# In[ ]:


X[:5,:]


# In[ ]:


y=num_columns['Earnings_Per_Share'].values


# ### Data Split

# In[ ]:


X_train,X_test,_,y_test=train_test_split(X,y,test_size=0.25)


# In[ ]:


clf=KMeans(n_clusters=2)


# In[ ]:


clf.fit(X_train)


# In[ ]:


clf.cluster_centers_


# In[ ]:


clf.cluster_centers_.shape


# In[ ]:


clf.labels_


# In[ ]:


clf.labels_.size


# In[ ]:


clf.inertia_


# In[ ]:


silhouette_score(X_train,clf.labels_)


# In[ ]:


y_pred=clf.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


np.sum(y_pred==y_test)/y_test.size


# In[ ]:


dx=pd.Series(X_test[:,0])
dy=pd.Series(X_test[:,1])
sns.scatterplot(dx,dy,hue=y_pred)


# In[ ]:


sse=[]
for i,j in enumerate(range(10)):
    n_clusters = i+1
    clf1 = KMeans(n_clusters = n_clusters)
    clf1.fit(X_train)
    sse.append(clf1.inertia_ )


# In[ ]:


sns.lineplot(range(1, 11), sse)


# In[ ]:


visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()              # Finalize and render the figure


# In[ ]:


from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()


# ## Thanks for giving ur valuable time/feedback

# ## Have a great day!!
