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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.simplefilter('ignore')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_excel('/kaggle/input/japan-bank/japan_bank.xls')


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df = df.drop(['id', 'year'], 1)
df['securities_interest_dividend_per_1m'] = df['securities_interest_dividend_per_1m'].replace('-',np.nan)
df['securities_interest_dividend_per_1m'] = df['securities_interest_dividend_per_1m'].astype(np.float)
df.index = df['bank name']


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


# In[ ]:


df['asset_per_1m'].sort_values(ascending=False).plot.bar(figsize=(20,7))


# In[ ]:


def mega_bank(num):
    if num > 0.25*1e8:
        return 1
    
    else:
        return 0


# In[ ]:


df['mega_bank'] = df['asset_per_1m'].map(mega_bank)


# In[ ]:


df_ratio = pd.DataFrame()
df_ratio['cash_asset_ratio%'] = (df['cash_deposit_per_1m'] / df['asset_per_1m']) * 100
df_ratio['loan_asset_ratio%'] = (df['loan_per_1m'] / df['asset_per_1m']) * 100
df_ratio['stock_asset_ratio%'] = (df['stocks_an_ bonds_per_1m'] / df['asset_per_1m']) * 100
df_ratio['ROA%'] = (df['ordinary_profit_per_1m'] / df['asset_per_1m']) * 100
df_ratio['capital_adequacy_ratio%'] = (df['equity_per_1m'] / df['asset_per_1m']) * 100
df_ratio['loan-deposit_ratio%'] = (df['loan_per_1m'] / df['deposit_per_1m']) * 100
df_ratio['income_profit_ratio%'] = (df['ordinary_profit_per_1m'] / df['ordinary_income_per_1m']) * 100
df_ratio['financial_leverage'] = df['asset_per_1m'] / df['equity_per_1m']
df_ratio['loan_interest_ratio%'] = (df['loan_interest_per_1m'] / df['loan_per_1m']) * 100
df_ratio['depoit_interest_raio%'] = (df['deposit_interest_per_1m'] / df['deposit_per_1m']) * 100
df_ratio['asset_income_ratio%'] = (df['asset_management_income_per_1m'] / df['asset_per_1m']) * 100
df_ratio['service_revenue_ratio%'] = (df['revenue_from_service_per_1m'] / df['service_costs_per_1m']) * 100
df_ratio['stock_manegement_ratio%'] = (df['securities_interest_dividend_per_1m'] / df['stocks_an_ bonds_per_1m']) * 100
df_ratio['mega_bank'] = df['mega_bank']


# In[ ]:


df_ratio.head()


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(round(df_ratio.corr(), 3), annot=True, cmap='jet')


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA()
df_ratio = df_ratio.dropna()
x_pca = pca.fit_transform(df_ratio.loc[:, :'stock_manegement_ratio%'])
df_pca = pd.DataFrame(x_pca)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


df_ratio = df_ratio.reset_index()
df_ratio['1_st'] = df_pca[0]
df_ratio['2_nd'] = df_pca[1]


# In[ ]:


sns.pairplot(df_ratio, hue='mega_bank')

