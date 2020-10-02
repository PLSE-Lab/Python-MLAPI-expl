#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Variables:
# 
# * RCONHT71: Total loans held for trading
# * RCON2200: Total Deposit
# * RCFD2170: Total Asset
# * RCON1766: C&I Loans
# * RCFD2948: Total Liabilities
# * RIAD4340: Net Income
# * RCFAP742: Tier 1 Capital
# * RCON3210: Total Equity Capital

# ## TODO-1:

# Density plots/histograms/top n shares, of bank total assets. 
# 
# Questions: 
# * How concentrated is the US banking market?
# * What's the distribution of total assets across banks? (log total assets might be better behaved. What's the log SD?)
# * What are the top n share? i.e. sort banks by total assets, take cumsum(total assets) / sum(total assets), for top 10 banks.

# In[ ]:


pd.set_option('display.max_columns',1000)
path = '../input/summer-research'
df_rc=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RC 03312020.txt',delimiter="\t")


# In[ ]:


df_rc=df_rc.dropna(how='all',axis=1)
df_rc.head()


# In[ ]:


df_rc_totalasset=df_rc[['IDRSSD','RCFD2170','RCFD2948']]
df_rc_totalasset=df_rc_totalasset.dropna(how='all',thresh=3)
df_rc_totalasset=df_rc_totalasset.astype(float)
df_rc_totalasset.tail()


# ### Density Plot

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.figsize": (8, 6)}); 
sns.set(style="white",palette='deep',color_codes=False)


# > #### Density Plot

# In[ ]:


df_rc_plot=df_rc_totalasset
df_rc_plot_log=df_rc_plot['RCFD2170'].astype(float).apply(np.log)  # Log asset
#sns.kdeplot(data=df_rc_plot_log,label="Bank Total Asset" ,shade=True)


fig = plt.figure(figsize=(16,6))
ax1= fig.add_subplot(1,2,1)
ax2= fig.add_subplot(1,2,2)
sns.kdeplot(df_rc_plot_log,shade=True,label='Log Bank Total Asset',ax=ax1)
sns.kdeplot(df_rc_plot['RCFD2170'],label="Bank Total Asset" ,shade=True,ax=ax2)
sns.despine()


# ### Histograms

# In[ ]:


import plotly.express as px
#fig = px.histogram(cp, x="degree_p", y="salary", color="gender")
ax1=px.histogram(df_rc_plot_log,width=600,height=450,title="Log Bank Total Asset",histnorm='density')
#fig.update_layout(
 #   autosize=False)#,paper_bgcolor="LightSteelBlue")
#fig.update_yaxes(automargin=True)
ax2=px.histogram(df_rc_plot['RCFD2170'],width=600,height=450,title="Bank Total Asset",histnorm='density')
ax1.show()
ax2.show()


# In[ ]:


df_rc_plot_log=pd.DataFrame(df_rc_plot_log)
df_rc_plot_log['IDRSSD']=df_rc_plot['IDRSSD']
fig=px.scatter_polar(df_rc_plot_log, r="RCFD2170", theta="IDRSSD",color="IDRSSD")
#fig.update_layout(font_size=6)
fig.show()


# ### Log SD(Standard Deviation)

# In[ ]:


df_rc_plot['RCFD2170'].astype(float).describe()
log_sd_plus=1.851376e+08+4.525690e+08
log_sd_minus=1.851376e+08-4.525690e+08
right=sum(np.where(df_rc_plot['RCFD2170'].astype(float)>log_sd_plus,1,0))
left=sum(np.where(df_rc_plot['RCFD2170'].astype(float)<log_sd_minus,1,0))
(80-right-left)/80


# #### Log :

# In[ ]:


df_rc_plot_log=df_rc_plot['RCFD2170'].astype(float).apply(np.log)
df_rc_plot_log.describe()
log_sd_plus=16.887587+2.541675
log_sd_minus=16.887587-2.541675
right=sum(np.where(df_rc_plot_log>log_sd_plus,1,0))
left=sum(np.where(df_rc_plot_log<log_sd_minus,1,0))
(80-right-left)/80


# ### Top n shares

# In[ ]:


df_rc_plot_log_sort=sorted(df_rc_plot_log,reverse=True)
Top_n_shares=np.cumsum(df_rc_plot_log_sort)/sum(df_rc_plot_log_sort)
fig = px.bar(Top_n_shares[0:10],title='Top 10 Share for Log Total Asset',width=600,height=400)
fig.show()


# In[ ]:


df_rc_plot_sort=sorted(df_rc_plot['RCFD2170'].astype(float),reverse=True)
Top_n_shares=np.cumsum(df_rc_plot_sort)/sum(df_rc_plot_sort)
fig = px.bar(Top_n_shares[0:10],title='Top 10 Share for Total Asset',width=600,height=400)
fig.show()


# ## TODO-2:

# How well are different ``size'' metrics correlated?
# 
# Questions:
# * What are the size metrics? (deposits? loans? etc.)
# * Scatterplot these against each other (deposit vs total assets, loans vs total assets, etc.). Maybe do logs, as this will be better behaved

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# #### Merge Total Asset and Total Deposit:

# In[ ]:


df_rce_totaldeposit=df_rc[['IDRSSD','RCON2200']]
df_rce_totaldeposit=df_rce_totaldeposit.drop(index=0).astype(float)
#df_rce_totaldeposit=np.where(df_rce_totaldeposit==0,np.nan,df_rce_totaldeposit)
#df_rce_totaldeposit=pd.DataFrame(df_rce_totaldeposit)
df_rce_totaldeposit_new=df_rce_totaldeposit.dropna(how='all',thresh=2)
df_rce_totaldeposit_new.columns=['IDRSSD','RCON2200']
df_asset_deposit=pd.merge(df_rc_totalasset,df_rce_totaldeposit_new,on='IDRSSD')
df_asset_deposit.tail()


# In[ ]:


fig_1=px.histogram(df_asset_deposit['RCON2200'].apply(np.log),width=600,height=450,title="Log Bank Total Deposit",histnorm='density')
fig_2=px.histogram(df_asset_deposit['RCON2200'],width=600,height=450,title="Bank Total Deposit",histnorm='density')
fig_1.show()
fig_2.show()


# 

# In[ ]:


df_loans=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCH 03312020.txt',delimiter="\t")
df_totalloans=df_loans[['IDRSSD','RCONHT71']].dropna(how='all',thresh=2)
df_totalloans=df_totalloans.astype(float)
df_asset_deposit_loan=pd.merge(df_asset_deposit,df_totalloans,on='IDRSSD')
df_asset_deposit_loan.columns=['IDRSSD','Total Asset','Total Liabilities','Total Deposit','Total loans held for trading']
df_asset_deposit_loan.tail()


# In[ ]:


df_asset_deposit_loan['Total loans held for trading']=np.where(df_asset_deposit_loan['Total loans held for trading']>0,1,0)
df_asset_deposit_loan.tail()


# In[ ]:


px.scatter(df_asset_deposit_loan,x='Total Asset',y='Total Deposit',size='Total Liabilities',color='Total loans held for trading',width=800,trendline="ols")


# After Log Transformation:

# In[ ]:


df_asset_deposit_loan_log=df_asset_deposit_loan.copy()
df_asset_deposit_loan_log[['Total Asset','Total Liabilities','Total Deposit']]=df_asset_deposit_loan_log[['Total Asset','Total Liabilities','Total Deposit']].apply(np.log)
df_asset_deposit_loan_log.tail()
px.scatter(df_asset_deposit_loan_log,x='Total Asset',y='Total Deposit',size='Total Liabilities',color='Total loans held for trading',width=800,trendline="ols")


# #### C&I Loans:

# It's quite strange to see loads of zero in loans. Since I do not find the variable called "Total Loans". Therefore, I use "C&I Loans" instead and in most previous studies they also use C&I loans. But it is so strange that I do not find a bank both has total asset and C&I loans.
# 
# If I do not consider the "Loan" variable, there are also 80 banks included in the data sample.

# In[ ]:


df_ci_loans=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCCI 03312020.txt',delimiter="\t")
df_ci_loans_new=df_ci_loans[['IDRSSD','RCON1766']]
df_ci_loans_new=df_ci_loans_new.dropna(how='all',thresh=2)


# In[ ]:


df_deposit_loans=pd.merge(df_asset_deposit,df_ci_loans_new,on='IDRSSD',how='left')
df_deposit_loans.columns=['IDRSSD','Total Asset','Total Liabilities','Total Deposit','C&I Loans']
df_deposit_loans['IDRSSD']=df_deposit_loans['IDRSSD'].astype(float)
df_deposit_loans.tail()
#px.scatter(df_deposit_loans,x='Total Deposit',y='C&I Loans',width=800)
df_deposit_loans.dropna(how='all',thresh=5)


# In[ ]:


df_rc_totalasset.tail()


# In[ ]:


pd.merge(df_rc_totalasset,df_ci_loans_new,on='IDRSSD',how='left')


# In[ ]:


df_ci_loans_new[df_ci_loans_new['IDRSSD']==12311]


# It seems these 80 banks(selected according to the variable total asset) do not contain the data of C&I Loans.

# Then, I select other commonly used variables, such as: 
# 
# Net Income, Regulatory Capital(Tier 1 Capital), Equity Capital

# In[ ]:


import seaborn as sns
df_RI=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RI 03312020.txt',delimiter="\t")
df_netincome=df_RI[['IDRSSD','RIAD4340']]
df_netincome.dropna(how='all',thresh=2,inplace=True)


# In[ ]:


df_asset_deposit_loan_income_log=pd.merge(df_asset_deposit_loan_log,df_netincome,on='IDRSSD')
df_asset_deposit_loan_income_log.rename(columns={'RIAD4340':'Net Income'},inplace=True)
df_asset_deposit_loan_income_log['Net Income']=df_asset_deposit_loan_income_log['Net Income'].astype(float).apply(np.log)
df_asset_deposit_loan_income_log.tail()


# In[ ]:


df_RCR=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCRI 03312020.txt',delimiter="\t")
df_tier1=df_RCR[['IDRSSD','RCFAP742']].dropna(how='all',thresh=2)


# In[ ]:


df_asset_deposit_loan_income_regulatory_log=pd.merge(df_asset_deposit_loan_income_log,df_tier1,on='IDRSSD')
df_asset_deposit_loan_income_regulatory_log.rename(columns={'RCFAP742':'Tier 1 Capital'},inplace=True)
df_asset_deposit_loan_income_regulatory_log['Tier 1 Capital']=df_asset_deposit_loan_income_regulatory_log['Tier 1 Capital'].astype(float).apply(np.log)
df_asset_deposit_loan_income_regulatory_log.tail()


# In[ ]:


df_capital=df_rc[['IDRSSD','RCON3210']].dropna(how='all',thresh=2)
df_capital['RCON3210']=df_capital['RCON3210'].astype(float).apply(np.log)
df_capital.rename(columns={'RCON3210':'Total Equity Capital'},inplace=True)
pd.merge(df_asset_deposit_loan_income_regulatory_log,df_capital,on='IDRSSD')


# But I do not find the common part of Total Equity Capital with the existing dataset. 

# The most confusing part of the dataset is the "loans". I can not find the suitable variable to represent "loans".
# If I do not consider loans, the dataset can include 80 banks but here is only 56 banks.

# In[ ]:


sns.set(style="darkgrid",palette='deep',color_codes=False)
df_plot=df_asset_deposit_loan_income_regulatory_log[['Total Asset','Total Liabilities','Total Deposit','Total loans held for trading','Net Income','Tier 1 Capital']]
sns.pairplot(df_plot, hue="Total loans held for trading", size=2, diag_kind="kde")


# ### Suggestions from these two questions:

# For question 1:
# 
# * If we do not Log the bank total asset, we can see the US banking market is concentrated, since the total asset value of top 10 banks is 0.72. But the distribution of US banking market is dispersed. The density plot shows the distribution is right skewed and most of banks are below 5e8 total asset.
# * 80 banks have the value of total asset, meaning that our dataset has to be confined with these 80 banks.
# 
# For question 2:
# 
# * From the scatter plot, I can see the positive correlation nearly in all the variables.
# * But there is a problem -- the limitation of "Loans", I can not find the suitable data considering "Loans". The bank call report does not include "Total Loans" and "C&I Loans" does not have the common part with other variables(e.g., If bank A has the data of total asset, then it does not have the data of C&I Loans). 

# ## TODO-3:

# ### Foregin vs Domestic:

# Variables:
# 
# * RCON2215: TOTAL TRANSACTIONS ACCOUNTS
# * RCFN2200: TOTAL DEPOSITS

# In[ ]:


df_Deposit_Liabilities_foreign=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCEII 03312020.txt',delimiter="\t")
df_Deposit_Liabilities_domestic=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCEI 03312020.txt',delimiter="\t")


# In[ ]:


df_totaldomestic_deposit=df_Deposit_Liabilities_domestic[['IDRSSD','RCON2215']].dropna(thresh=2)
print(df_Deposit_Liabilities_foreign['RCFN2200'].dropna())
df_totalforeign_deposit=df_Deposit_Liabilities_foreign[['IDRSSD','RCFN2200']].dropna()
df_totalforeign_deposit.tail()


# In[ ]:


df_deposit_fd=pd.merge(df_totalforeign_deposit,df_totaldomestic_deposit,on='IDRSSD')
df_deposit_fd=df_deposit_fd.astype(float)
print(df_deposit_fd)
print("\n")
print("The number of zero in Foreign Deposit:\n")
print(len(df_deposit_fd[df_deposit_fd['RCFN2200']==0]))


# We can see here the number of banks that have the data of foreign deposit is 72. This number will decrease when we intersect foreign deposit and total asset(80 numbers).

# In[ ]:


deposit_ftod_ratio=df_deposit_fd['RCFN2200']/df_deposit_fd['RCON2215']
deposit_ftod_ratio=pd.DataFrame(deposit_ftod_ratio)
deposit_ftod_ratio['IDRSSD']=df_deposit_fd['IDRSSD']
deposit_ftod_ratio.columns=['Ratio','IDRSSD']
deposit_ftod_ratio.tail()


# In[ ]:


#deposit_ftod_ratio=df_deposit_fd['RCFN2200']/df_deposit_fd['RCON2215']
px.bar(deposit_ftod_ratio['Ratio'],title='Foreign/Domestic',width=600,height=400)


# In[ ]:


fig=px.scatter_polar(deposit_ftod_ratio, r="Ratio", theta="IDRSSD",color="IDRSSD")
fig.show()


# ### Mortage vs C&I:

# Variables:
# 
# * RCFDG300+G304+G308:  Mortgage-backed securities (MBS)
# * RCON1766: C&I Loans

# In[ ]:


df_ci_loans_new.tail()


# In[ ]:


df_rcd=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCD 03312020.txt',delimiter="\t")
df_mbs=df_rcd[['IDRSSD','RCFDG379','RCFDG380','RCFDG381','RCFDK197','RCFDK198']]
df_mbs.dropna(thresh=6).tail()


# In[ ]:


df_rcd=pd.read_csv(f'{path}/FFIEC CDR Call Schedule RCD 03312020.txt',delimiter="\t")
df_rcd=df_rcd.dropna(how='all',axis=1)
df_mbs=df_rcd['RCFDG379']+df_rcd['RCFDG380']+df_rcd['RCFDG381']+df_rcd['RCFDK197']+df_rcd['RCFDK198']
#df_mbs['IDRSSD']=df_rcd['IDRSSD']
#df_mbs.dropna(how='all',thresh=2)

