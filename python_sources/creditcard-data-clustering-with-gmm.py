#!/usr/bin/env python
# coding: utf-8

# # Credit Card Data Clustering
# 
# Here we have a sample of around 9000 creditcard holders for a span of 6 months depicting their purchase habits and the same can be used to segment the customers for marketting purpose to maximize sales as well customer satifaction.
# 
# This kernel focuses on **Gaussian Mixture Model** for clustering and will be devided in following sections:-
# * [Introduction to Gaussian Mixture model](#1)
# * [Importing Required Packages](#2)
# * [Preprocessing](#3)
# * [Data Visualization](#4)
# * [Selection of k for GMM](#5)
# * [Clustering using GMM](#6)
# * [Interpretation of Clusters](#7)
# * [Anomaly Detection](#8)

# # <a id="1">Gaussian Mixture Model</a>
# 
# Gaussian Mixture Models (GMMs) are based on Gaussian Distributions and are flexible building blocks for other machine learning algorithms. They are great approximations for general probability distributions but also because they remain somewhat interpretable even when the dataset gets very complex. Mixture Models do not require to know about data and the subpopulation to which it belongs but learn about the same later on by finding the distribution(s) for its each feature.
# 
# In our dataset we have 17 features, _viz_ , 
#  - **CUSTID** : Identification of Credit Card holder (Categorical)
#  - **BALANCE** : Balance amount left in their account to make purchases
#  - **BALANCEFREQUENCY** : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
#  - **PURCHASES** : Amount of purchases made from account
#  - **ONEOFFPURCHASES** : Maximum purchase amount done in one-go
#  - **INSTALLMENTSPURCHASES** : Amount of purchase done in installment
#  - **CASHADVANCE** : Cash in advance given by the user
#  - **PURCHASESFREQUENCY** : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
#  - **ONEOFFPURCHASESFREQUENCY** : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
#  - **PURCHASESINSTALLMENTSFREQUENCY** : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
#  - **CASHADVANCEFREQUENCY** : How frequently the cash in advance being paid
#  - **CASHADVANCETRX** : Number of Transactions made with "Cash in Advanced"
#  - **PURCHASESTRX** : Numbe of purchase transactions made
#  - **CREDITLIMIT** : Limit of Credit Card for user
#  - **PAYMENTS** : Amount of Payment done by user
#  - **MINIMUM_PAYMENTS** : Minimum amount of payments made by user
#  - **PRCFULLPAYMENT** : Percent of full payment paid by user
#  - **TENURE** : Tenure of credit card service for user
# 
# Leaving the first feature each may have its own Gaussian distribution that can easily depict the customer behaviour. Now we start with importing the required packages.

# # <a id = "2">Importing Required Packages</a>
# We have imported various packages here for purpose as below:-
#  - **Numpy** : *for array manipulation*
#  - **Pandas** : *for reading dataset and manipulating its data*
#  - **matplotlib and Seaborn** : *for Data Visualization*
#  - **sklearn packages** : *for clustering*
#  - **IPython classes** : *for formatting output in kernel*

# In[ ]:


#STEP 1: Get right arrows in quiver

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler,normalize
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture as GMM
from sklearn.manifold import TSNE

import warnings #To hide warnings

#1.1: Set the stage
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import display, HTML

InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings("ignore")


# ## Reading and Pre-viewing Data

# In[ ]:


cc = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')
cc.info()
cc[cc.columns[cc.isna().any()]].isna().sum().to_frame().T
cc.sample(5)
cc.describe()


# In[ ]:


cc.quantile([0.75,0.8,.85,.9,.95,1])


# ### OBSERVATIONS
# - Balance is updated frequently(**25% percentile is 1**) in most cases.
# - Most of the customers have CC tenure 12 (**1<sup>st</sup> quantile is 12**).
# - most of the customers avoid cash advance, but a section (below 5% of them all,above 75% use no cash advances) uses cash advances and these advances are paid even 50% faster than regular advances[**CASHADVANCEFREQUENCY max value is 1.5 instead of general frequency max 1.0**]
# - Only 5% of customers make most of purachases.

# In[ ]:


display(HTML('<h4>There are '+str(np.sum(cc.BALANCE>cc.CREDIT_LIMIT))
             +' customers in the list who have more balance than the credit limit assigned. '
             +'It may be due to more payament than usage and/or continuous pre-payment.</h4>'))


# # <a id="3">Pre Pocessing</a>
# The dataset has 8 features and 8950 observations. The dataset has **NaN** values in 2 features, viz, **CREDIT_LIMIT and	MINIMUM_PAYMENTS**. To remove these NaN values we can either remove these rows/columns or can **impute the features with mean/median/minimum or maximum** (since data being numerical, forefill or backfill not considered).Again, as the columns are of importance thus min/max imputation is out of question. Now to find out correct imputation we check their distribution.

# In[ ]:


cc.rename(columns = {col:col.lower() for col in cc.columns.values},inplace=True)
sns.jointplot(cc.credit_limit,cc.minimum_payments,kind = 'kde', dropna=True)


# As both the plots are skewed (to right), mean imputation can affect the model thus we go with **median imputation**. To do so we can either use sklearn **SimpleImputer** class or can simply use **fillna method from pandas to replace all NaN values with median of the feature** as shown below.
# 
# After that,the cust_id feature does not affect our model(being unique) we can drop the same.
# Futhermore, The dataset has to be **Scaled and normalized** to avoid effect of size of value-set premise. 

# In[ ]:


cc.fillna(cc.median(),inplace=True) #More outliers thus median in both cases
cust = cc.cust_id
cc.drop(columns = ['cust_id'],inplace=True)


# In[ ]:


ss = StandardScaler()
X= normalize(ss.fit_transform(cc.copy()))
X = pd.DataFrame(X,columns=cc.columns.values)


# # <a id="4">Data Visualization and Interpretation</a>
# After scaling and normalization, we can verify the distribution of the features and their mutual- dependence. *First*, we draw a box chart to see the percentiles as below.

# In[ ]:


fig, axs = plt.subplots(6,3, figsize=(20, 20))
for i in range(17):
        p = sns.distplot(cc[cc.columns[i]], ax=axs[i//3,i%3],kde_kws = {'bw':2})
        p = sns.despine()
plt.show()


# ### OBSERVATIONS
# 
# - In some cases **Balance** is unutilised.
# - There are **two** clear segments for purchases, one avoiding purchases at all and another purchasing much frequenly(just 5% making almost all)
# - Most of the cutomers avail credit limit below 10k
# - Payments are highly skewed and most payments are below 10k

# In[ ]:


X.boxplot(figsize = (30,25),grid=True,fontsize=25,rot=90)


# As the chart shows, there is a number of outliers in almost all the features. While Balance frequency and tenure have outliers down almost all other features have outliers up the whisker. We can assume that there we will find quiet a many anomalous observations, that opens a scope of finding frauds also.
# <br>
# Now we can check the correlations of features.

# In[ ]:


plt.figure(figsize=(16,12))
p = sns.heatmap(cc.corr(),annot=True,cmap='jet').set_title("Correlation of credit card data\'s features",fontsize=20)
plt.show()


# The heatmap shows the features are corellated. Here we can either drop the columns by capping variance or can use all of them for clustering. We gao with the second option.

# # <a id ="5">Selection of k for Gaussian Mixture Model</a>
# We can evaluate the likelihood of the data under the model, using cross-validation to avoid over-fitting. Another means of correcting for over-fitting is to adjust the model likelihoods using some analytic criterion such as the Akaike information criterion (AIC) or the Bayesian information criterion (BIC).
# Here, we are checking for both AIC as well BIC for estimating value of k

# In[ ]:


#Selecting correct number of components for GMM
models = [GMM(n,random_state=0).fit(X) for n in range(1,12)]
d = pd.DataFrame({'BIC Score':[m.bic(X) for m in models],
                  'AIC Score': [m.aic(X) for m in models]},index=np.arange(1,12))
d.plot(use_index=True,title='AIC and BIC Scores for GMM wrt n_Compnents',figsize = (10,5),fontsize=12)


# In[ ]:


from sklearn.base import ClusterMixin
from yellowbrick.cluster import KElbow

class GMClusters(GMM, ClusterMixin):

    def __init__(self, n_clusters=1, **kwargs):
        kwargs["n_components"] = n_clusters
        kwargs['covariance_type'] = 'full'
        super(GMClusters, self).__init__(**kwargs)

    def fit(self, X):
        super(GMClusters, self).fit(X)
        self.labels_ = self.predict(X)
        return self 

oz = KElbow(GMClusters(), k=(2,12), force_model=True)
oz.fit(X)
oz.show()


# Here we have limited the models to "full- covariance".<br>
# The chart here shows that the value of AIC as well BIC are continuously decreasing. That may be due to so much noise or small size of sample. For covenience, we take model[6] as our model.

# In[ ]:


model= models[6]
model.n_init = 10
model


# # <a id="6">Clustering using GMM (k = 7)</a>
# Now we actually cluster the data with k =7. For better results we can change n_init to 10.

# In[ ]:


clusters = model.fit_predict(X)
display(HTML('<b>The model has converged :</b>'+str(model.converged_)))
display(HTML('<b>The model has taken iterations :</b>'+str(model.n_iter_)))


# # <a id="7">Interpretation of Clusters</a>

# In[ ]:


sns.countplot(clusters).set_title('Cluster sizes',fontsize=20)


# In[ ]:


cc1 = cc.copy()
cc1['cluster']=clusters
for c in cc1:
    if c != 'cluster':
        grid= sns.FacetGrid(cc1, col='cluster',sharex=False,sharey=False)
        p = grid.map(sns.distplot, c,kde_kws = {'bw':2})
plt.show()


# In[ ]:


#cc1.groupby('cluster').agg({np.min,np.max,np.mean}).T
for i in range(7):
    display(HTML('<h2>Cluster'+str(i)+'</h2>'))
    cc1[cc1.cluster == i].describe()


# ### OBSERVATIONS
# 
# <h4>CLUSTER 0</h4>
# 
#  - 1367 cutomers fall in cluster0.
#  - Low to moderate purcahses.
#  - Balances are updated frequently.
#  - **CASH ADVANCES AVOIDED**
#  - All have full tenure(12 years)
#  - More than 75% do not pay in full.
#  - Oneoff Purchase and purchase installments are again biased on both extremes.
#  
# <h4>CLUSTER 1</h4>
# 
#  - Most populous cluster
#  - No purchases at all of either kind
#  - Low Balance is kept (about Rs. 2000 against max of balances i.e., Rs. 30000)
#  - Only Cash Advances are availed.
#  - Low credit limit (average Rs. 4000)
#  - Full payments **are avoided**
#  
# <h4>CLUSTER 2</h4>
# 
#  - **NO CASH ADVANCES**
#  - All have tenure of 12 years
#  - Rather low level of balance is maintained
#  - **NO FULL PAYMENTS**
#  - Balance updation is frequent.
#  
# <h4>CLUSTER 3</h4>
# 
#  - **NO ONE-OFF PURCHASES**
#  - Cash advances are hugely avaoided
#  - **Frequent but low  volume INSTALLMENT purchases**
#  - Varying but low majorly low tenure facilities availed.
# 
# <h4>CLUSTER 4</h4>
# 
#  - Minimum payments are rather present and are multi-fold of credit limit. **This means minimum dues are regularly paid**
#  - Tenure of 12 years
#  - Balances are updated frequently.
#  - Purchases, cash advances and payments all are low volumne and frequent. **It deduces MOSTLY lower-middle income group**
#  
# <h4>CLUSTER 5</h4>
# 
#  - **NO CASH ADVANCES--NO ONE-OFF PURCHASES**
#  - More utilisation of limit as balances are kept low.
#  - Balance updation is frequent.
#  - Mean installment purchases are low volume.
#  - Low number of purchase transaction (mean 11)
#  
# <h4>CLUSTER 6</h4>
# 
#  - Varying credit limits
#  - Varying tenures
#  - Low balances maintained and updated frequently
#  - Over-all mixed up behaviour

# In[ ]:


tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(X.copy())

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker=10,
            s=10,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=clusters   # Colour as per gmm
            )


# # <a id="8">Anomaly Detection</a>
# Anomaly points can be considered as points which do occur in outliers or rather say in areas of low density. We can identify them by setting some density threshold.
# Here we are taking density threshold as 4%.

# In[ ]:


density = model.score_samples(X)
density_threshold = np.percentile(density,4)
cc1['cluster']=clusters
cc1['Anamoly'] = density<density_threshold
cc1


# In[ ]:


df = cc1.melt(['Anamoly'], var_name='cols',  value_name='vals')

g = sns.FacetGrid(df, row='cols', hue="Anamoly", palette="Set1",sharey=False,sharex=False,aspect=3)
g = (g.map(sns.distplot, "vals", hist=True, rug=True,kde_kws = {'bw':2}).add_legend())


# In[ ]:


unanomaly = X[density>=density_threshold]
c = clusters[density>=density_threshold]
tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(unanomaly)
plt.figure(figsize=(15,10))
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],marker='x',s=10, linewidths=5, c=c)

