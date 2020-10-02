#!/usr/bin/env python
# coding: utf-8

# # Clustering of Credit Card Data using Gaussian Mixture Method (GMM)

# ####  Introduction
# 
# ### ***Gaussian Mixture Method*** : It is a  probabilistic model which is used to represent normally distributed subpopulations within an overall population.In Mixture mode subpopulation assigment is unknown.It allows the model to learn the subpopulation automatically,so we can say it is unsupervised learning.Clusters are elliptical-shaped
# 
# ### ***About Data Set*** : Dataset has  one notebook which consist credit card usage behavior of customers  in the last 6 months.Customer segmentation to define marketing strategy.
# 
# #### Contents of this workbook:
#  * **Data Loading:** Load/Read Data from file
#  * **Cleansing of Data:** Rename columns,check Na values,fill/drop na values
#  * **Data Processing:** Standardization  & Normalization
#  * **Plotting of Data:** Graphing of data
#  * **Clustering and Interpretation using Gaussian Mixture Method :** How many GMM clusters using '*bic' and 'aic'*.Perform clustering
#  * **TSNE Check :**  Check if your clusters do not overlap
#  * **Anomalous clients vs Un-anomalous clients :** Anomalous clients where density less than 4%
#  

# ### Call libraries
# 

# In[ ]:



get_ipython().run_line_magic('reset', '-f')
from sklearn.cluster import KMeans
#For creating elliptical-shaped clusters
from sklearn.datasets import make_blobs
#OS related
import os
#Data manipulation
import pandas as pd
import numpy as np

#for math functions
import math

# Data processing 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

#Graphing
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly.express as px
from matplotlib.colors import LogNorm
import seaborn as sns
#TSNE
from sklearn.manifold import TSNE

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


from sklearn.mixture import GaussianMixture

import scipy


# ### Set Directory

# In[ ]:


os.chdir("/kaggle/input/ccdata")
os.listdir()            # List all files in the folder


# ### Load Data

# In[ ]:


dfcc=pd.read_csv('CC GENERAL.csv')
dfcc.head()
print("No of Customers:",dfcc.shape[0])
print("No of Columns:",dfcc.shape[1])


# * ***Dataset summarizes the usage of 8950 Customers based on 18 behavioral variables***

# In[ ]:


dfcc.columns


# * ***Data Dictionary for Credit Card dataset***
#  * ***CUSTID :*** Identification of Credit Card holder (Categorical)
#  * ***BALANCE :***  Balance amount left in their account to make purchases 
#  * ***BALANCEFREQUENCY :***  How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
#  * ***PURCHASES :*** Amount of purchases made from account
#  * ***ONEOFFPURCHASES :*** Maximum purchase amount done in one-go
#  * ***INSTALLMENTSPURCHASES :*** Amount of purchase done in installment
#  * ***CASHADVANCE :*** Cash in advance given by the user
#  * ***PURCHASESFREQUENCY :*** How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
#  * ***ONEOFFPURCHASESFREQUENCY :*** How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
#  * ***PURCHASESINSTALLMENTSFREQUENCY :*** How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
#  * ***CASHADVANCEFREQUENCY :*** How frequently the cash in advance being paid
#  * ***CASHADVANCETRX :*** Number of Transactions made with "Cash in Advanced"
#  * ***PURCHASESTRX :*** Numbe of purchase transactions made
#  * ***CREDITLIMIT :*** Limit of Credit Card for user
#  * ***PAYMENTS :*** Amount of Payment done by user
#  * ***MINIMUM_PAYMENTS :*** Minimum amount of payments made by user
#  * ***PRCFULLPAYMENT :*** Percent of full payment paid by user
#  * ***TENURE :*** Tenure of credit card service for user
#  * ****Anomalous clients vs Un-anomalous client :**** Anomalous clients ie with density less than 4%

# ### Cleansing of Data

# In[ ]:


#rename column names
dfcc.columns = [i.lower() for i in dfcc.columns]
dfcc.columns


# In[ ]:


#Drop cust_id..No use of This column
dfcc.drop(columns = ['cust_id'], inplace = True)


# In[ ]:


#Check null value

#null_columns=dfcc.columns[dfcc.isnull().any()]
dfcc.columns[dfcc.isnull().any()]
#Two columns having Null values
#Check how many Null Values
print("Number of Null Values:\n",dfcc[dfcc.columns[dfcc.isnull().any()]].isnull().sum())


# ##### *We will see graphical behaviour of these two columns*

# In[ ]:


#sns.distplot(dfcc.credit_limit)
#sns.distplot(dfcc.minimum_payments)
sns.kdeplot(dfcc.credit_limit, shade=True)
sns.kdeplot(dfcc.minimum_payments, shade=True)


# *We will fill na values with median

# ##### *Fill NA values*

# In[ ]:


values = {
              'minimum_payments' :   dfcc['minimum_payments'].median(),
                 'credit_limit'               :     dfcc['credit_limit'].median()
               }

dfcc.fillna(value = values, inplace = True)


# ### *Data Processing:* Standardization  & Normalization
#  * Standardization : StandardScaler() is for column-wise standardization.
#  * Normalization:  normalize() is to normalize each observation(row-wise) 
# 

# In[ ]:


ss =  StandardScaler()
cc_ss= ss.fit_transform(dfcc)
cc_ss = normalize(cc_ss)
df_out= pd.DataFrame(cc_ss, columns = dfcc.columns.values)
df_out


# ### Plotting of Data : Graphical representation  to find the relation between data

# In[ ]:



dfSummary=dfcc.describe()
dfSummary
dfSummary=dfSummary.T
dfSummary.plot(kind='bar',figsize = (20,8))
#check details where purchases is max
dfcc[dfcc.purchases == 49039.57]


# * Average Purchase is 1003, max purchase is 49039.57 .
# * 50% is 361.2800 .
# * Only one card holder has max value for each column.so we can due to that customer avg is high

# In[ ]:


(1.00*dfcc['tenure'].value_counts().sort_index()/len(dfcc)).plot(kind='barh')
plt.title('Tenure Distribution')
plt.xlabel('Distribution % ');


# * Mostly card holders has tenure 12

# In[ ]:


fig = px.box(dfcc,
                y="balance",
                x="tenure",
                
               title='Tenure Vs Balance',
                hover_data=dfcc.columns
               )
fig.show()
#Outlier values(Q3+1.5IQR) are more where Tenure is 12


# *Outlier values(Q3+1.5IQR) are more where Tenure is 12
# *Card holders having tenure of 11 & 12 has more balance

# In[ ]:


px.histogram(data_frame =dfcc,
                      x = 'tenure',
                      y = 'purchases',
                      marginal = 'violin',
                      title='Tenure vs Purchases',
                      histfunc = 'avg'
                
             )
#Observation : Card holders having tenure 12 has done max purchase


# In[ ]:


px.histogram(data_frame =dfcc,
                      x = 'tenure',
                      y = 'credit_limit',
                      marginal = 'violin',
                      title='Tenure vs credit_limit',
                      histfunc = 'avg'
                
             )
#Observation : Highest Avg credit limit : where tenure is 12


# * Card holders having tenure 12 has highest Avg credit limit

# 

# 

# In[ ]:


#Dist plot for all coulmns
plt.figure(figsize=(15,18))
noofrows= math.ceil(df_out.shape[1]/3)
noofrows
df_out_columns=df_out.columns.values

for i in range(df_out.shape[1]):
  plt.subplot(noofrows,3,i+1)
  sns.distplot(df_out[df_out_columns[i]])
 

plt.tight_layout()

#observation :  right skewed curve:ONEOFFPURCHASES,Installment purchase,Cash advance,payments,minimum_payments,balance


# In[ ]:





# In[ ]:


#Correlation Map for all features
df_corr=df_out.corr()
plt.figure(figsize = (15, 9))
sns.heatmap(df_corr, linecolor = 'black', linewidth = 1, annot = True)
plt.title('Correlation of credit card data\'s features \n Co Relation >0  means  poistive  co linear realtion \n < 0 means opposite Relation ')
plt.show()


# ##### #Purchases and oneoff_purchases has highest co relation i.e .87. We will draw a graph to show 

# In[ ]:


sns.jointplot(df_out.purchases, df_out.oneoff_purchases, kind = 'reg') 


# In[ ]:


frequency_cols = [col for col in df_out.columns if 'frequency' in col]
for i,col in enumerate(frequency_cols):
    g=sns.jointplot(df_out.credit_limit, df_out[col], kind = 'reg')
    s = scipy.stats.linregress(x = df_out['credit_limit'],y = df_out[col])
    g.fig.suptitle("Correlation coefficient : credit_limit Vs " + col + " : " + str(s[2]) )
    
#Observation  correlation coefficient is < 0 : Both Variable in opposite Direction
#Observation  correlation coefficient is > 0 : Both Variable in same Direction


# In[ ]:


df_outnozeropurchase=df_out[df_out.purchases != 0]
sns.kdeplot(df_outnozeropurchase['purchases'], shade=True)
sns.kdeplot(df_outnozeropurchase['installments_purchases'], shade=True)
sns.kdeplot(df_outnozeropurchase['oneoff_purchases'], shade=True)
sns.kdeplot(df_outnozeropurchase['credit_limit'], shade=True)
plt.title('Density Estimation Plot')
#observation : Installments_purchase and oneoff_purchase has equal ratio where purchases are high
#Credit Card limit does not have significant relation with purchases, installments_purchases,oneoff_purchases


# ### Clustering and Interpretation using Gaussian Mixture Method 

# #### Find out how many GMM clusters using 'bic' and 'aic'.

# In[ ]:


#Array for aic & bic
bic = []
aic = []
for i in range(16):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(df_out)
    bic.append(gm.bic(df_out))
    aic.append(gm.aic(df_out))
fig = plt.figure()


# In[ ]:


#Draw aic ,bic on plot to understand
plt.plot(range(1,len(aic)+1), aic,marker="o",label="aic")
plt.plot(range(1,len(bic)+1), bic,marker="o",label="bic")
plt.legend()
plt.show()


# ##### *from above graphical representation.we will fix 2 clusters*

# # TSNE Check

# In[ ]:


#Gussian Mixture
gm = GaussianMixture(
                     n_components = 2,
                     n_init = 10,
                     max_iter = 100)
gm.fit(df_out)
 
#Find tsne
tsne = TSNE(n_components = 2,perplexity=40.0)
tsne_out = tsne.fit_transform(df_out)
tsne_out


# In[ ]:


#draw TSNE
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=10,              # marker size
            linewidths=20,      # linewidth of marker edges
            c=gm.predict(df_out)   # Colour as per gm
            )


# # Anomalous clients vs Un-anomalous clients

# In[ ]:


densities = gm.score_samples(df_out)
density_threshold = np.percentile(densities,4)
anomalies      =     df_out[densities < density_threshold]      # Data of anomalous customers
# Unanomalous data
unanomalous =  df_out[densities >= density_threshold]      # Data of unanomalous customers
df_anomaly     =  pd.DataFrame(anomalies, columns = df_out.columns.values)
df_unanomaly = pd.DataFrame(unanomalous, columns = df_out.columns.values)
df_anomaly.shape
df_unanomaly.shape


# In[ ]:


#@author : Ashok sir 
#few changes made by me
def densityplots(df1,df2, label1 = "Anomalous",label2 = "Normal"):
    # df1 and df2 are two dataframes
    # As number of features are 17, we have 20 axes
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,15))
    ax = axes.flatten()
    fig.tight_layout()
    # Do not display 18th, 19th and 20th axes
    #because we need only 17 axes
    axes[3,3].set_axis_off()
    axes[3,2].set_axis_off()
    axes[3,4].set_axis_off()
    for i,col in enumerate(df1.columns):
        # https://seaborn.pydata.org/generated/seaborn.distplot.html
        # For every i, draw two overlapping density plots in different colors
        sns.distplot(df1[col],
                     ax = ax[i],
                     kde_kws={"color": "k", "lw": 3, "label": label1},   # Density plot features
                     hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "g"}) # Histogram features
        sns.distplot(df2[col],
                     ax = ax[i],
                     kde_kws={"color": "red", "lw": 3, "label": label2},
                     hist_kws={"histtype": "step", "linewidth": 2,"alpha": 1, "color": "b"})
densityplots(df_anomaly, df_unanomaly, label2 = "Unanomalous")


# 
# * balance : Un-anamolous has slightly more balance
# * balance_frequency : Un-anamolous has more balance frequency
# * Purchases : Un-anamolous make more purchases
# * oneoff_purchases : Un-anamolous make more one-off purchases
# * installments_purchases : Un-anamolous slightly more installment purchases
# * cash advance : Un-anamolous cash advance is more
# * purchase frequency : Un-anamolous is bi-modal curve. Abnormal has significantly more purchase frequency
# * on-off purchase frequency : Un-anamolous has more on-off purchase frequency
# * installments frequency : Un-anamolous has bimodal curve and have more purchase installments frequency for lower frequencies. 
# * For higher purchase installments frequency, abnormal occurrence is more
# * cash advance frequency : anamolous has more cash advance frequency
# * payments : Un-anamolous has signitficantly more payments
# * minimum payments : Un-anamolous has signitficantly more minimum payments
# * Tenure : Un-anamolous has slightly more tenure
# * cash_advanc_trx,purchase_trx,credit_limit,prc_full_payment : No significant difference

# In[ ]:




