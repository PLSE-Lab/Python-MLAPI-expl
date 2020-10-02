#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture 


# In[ ]:


cc = pd.read_csv("../input/cc.csv")


# In[ ]:


cc.columns = [i.lower() for i in cc.columns]


# In[ ]:


cc.drop(columns = ['cust_id'], inplace = True)


# In[ ]:


cc.fillna(cc.mean(), inplace=True)


# In[ ]:


cc.isnull().sum()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[ ]:


test=cc
ss = StandardScaler()     
ss.fit(test)                
x = ss.transform(test)      
x[:5, :]                  


# In[ ]:


df_out = preprocessing.normalize(x)


# In[ ]:


plt.scatter(df_out[:,0],df_out[:,1],alpha=0.2)
plt.show()


# In[ ]:


test_out = pd.DataFrame(data=df_out,  columns=['balance', 'balance_frequency', 'purchases', 'oneoff_purchases', 'installments_purchases', 'cash_advance', 'purchases_frequency',       'oneoff_purchases_frequency', 'purchases_installments_frequency',       'cash_advance_frequency', 'cash_advance_trx', 'purchases_trx',       'credit_limit', 'payments', 'minimum_payments', 'prc_full_payment',       'tenure'])


# In[ ]:


test_out.head()


# ## BALANCE AND BALANCE FREQUENCY

# In[ ]:


X = df_out[:,[0,1]] 
X


# In[ ]:


d = pd.DataFrame(X)


# In[ ]:


plt.scatter(d[0], d[1])
plt.show()


# In[ ]:


gmm = GaussianMixture(n_components = 3)


# In[ ]:


gmm.fit(d)


# In[ ]:


labels = gmm.predict(d)


# In[ ]:


d['labels']= labels


# In[ ]:


d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 


# In[ ]:


plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 


# ## PURCHASES AND INSTALLMENT PURCHASES

# In[ ]:


X = df_out[:,[2,4]] 
X
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])
plt.show()


# In[ ]:


gmm = GaussianMixture(n_components = 3)
gmm.fit(d)
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 


# In[ ]:


plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
plt.show()


# ## ONEOFF & INSTALLMENT PURCHASES

# In[ ]:


X = df_out[:,[3,4]] 
X
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])
plt.show()


# In[ ]:


gmm = GaussianMixture(n_components = 3)
gmm.fit(d)
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 


# In[ ]:


plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='m') 
plt.show()


# ## PURCHASE AND PURCHASE FREQUENCY

# In[ ]:


X = df_out[:,[2,6]] 
X
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])
plt.show()


# In[ ]:


gmm = GaussianMixture(n_components = 3)
gmm.fit(d)
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 
plt.scatter(d0[0], d0[1], c='m') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 


# ## LIMIT  AND PAYMEMTS

# In[ ]:


X = df_out[:,[12,13]] 
X
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])
plt.show()


# In[ ]:


gmm = GaussianMixture(n_components = 3)
gmm.fit(d)
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 


# In[ ]:


plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
plt.show()


# In[ ]:


from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_,  gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


# In[ ]:


plt.figure(figsize=(12,9))
gmm = GaussianMixture(n_components=3, )
plot_gmm(gmm, X)
plt.show()


# ## BALANCE AND % OF FULL PAYMENTS

# In[ ]:


X = df_out[:,[2,13]] 
X
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])


# In[ ]:


gmm = GaussianMixture(n_components = 3)
gmm.fit(d)
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 


# In[ ]:


plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
plt.show()


# In[ ]:


plt.figure(figsize=(12,9))
gmm = GaussianMixture(n_components=3, random_state=42)
plot_gmm(gmm, X)
plt.show()


# ## PURCHASE FREQUENCY AND ONEOFF PURCHASE FREQUENCY

# In[ ]:


X = df_out[:,[6,7]] 
X
d = pd.DataFrame(X)
plt.scatter(d[0], d[1])
plt.show()


# In[ ]:


gmm = GaussianMixture(n_components = 3)
gmm.fit(d)
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0] 
d1 = d[d['labels']== 1] 
d2 = d[d['labels']== 2] 


# In[ ]:


plt.scatter(d0[0], d0[1], c ='r') 
plt.scatter(d1[0], d1[1], c ='yellow') 
plt.scatter(d2[0], d2[1], c ='g') 
plt.show()


# In[ ]:


plt.figure(figsize=(12,9))
gmm = GaussianMixture(n_components=3, random_state=42)
plot_gmm(gmm, X)


# ## Densityplots to distinguish anomalous from normal:

# In[ ]:


X = df_out[:,:17] 
X
d = pd.DataFrame(X)
gmm = GaussianMixture(n_components = 3)
gmm.fit(d)

densities = gmm.score_samples(X)


# In[ ]:


density_threshold = np.percentile(densities,4)


# In[ ]:


anomalies =X[densities < density_threshold] 
unanomalous = X[densities >= density_threshold]


# In[ ]:


df_anomaly  = pd.DataFrame(anomalies, columns = cc.columns.values)
df_unanomaly  = pd.DataFrame(unanomalous, columns =cc.columns.values)


# In[ ]:


def densityplots(df1,df2, label1 = "Anomalous",label2 = "Normal"):    
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,15))
    ax = axes.flatten()
    fig.tight_layout()
    
    axes[3,3].set_axis_off()
    axes[3,2].set_axis_off()
    axes[3,4].set_axis_off()
    
    for i,j in enumerate(cc.columns):        
        sns.distplot(df1.iloc[:,i],
                     ax = ax[i],
                     kde_kws={"color": "k", "lw": 3, "label": label1},   # Density plot features
                     hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"}) # Histogram features
        sns.distplot(df2.iloc[:,i],
                     ax = ax[i],
                     kde_kws={"color": "red", "lw": 3, "label": label2},
                     hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "b"})


densityplots(df_anomaly, df_unanomaly, label2 = "Unanomalous")


# In[ ]:




