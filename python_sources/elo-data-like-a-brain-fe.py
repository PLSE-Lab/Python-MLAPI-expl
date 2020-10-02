#!/usr/bin/env python
# coding: utf-8

# **TSNE analisys**

# 1. import train and test df with calculated features. You can try your data
# 2. preprocess data
# 3. calculate TSNE (30-40 mins)
# 4. visualize outliers and targets
# 
# later:
# smart imputer (not mean but train from dataset)
# tsne import pretrained 2D,3D
# calculate knn for outliers
# merchants.csv additional features
# multitask - classification+regression in one model

# In[ ]:


get_ipython().system('pip install MulticoreTSNE')


# In[ ]:


from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np


# In[ ]:


import os
os.listdir('../input/')
path='../input/features-for-elo-merchants-competition/'


# In[ ]:


#import data:
train_df = pd.read_csv(path+'full_train_df.csv', index_col='card_id')
test_df = pd.read_csv(path+'full_test_df.csv', index_col='card_id')
df = train_df.append(test_df)
train_len = train_df.shape[0]

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']
cols = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

data = df[cols]
outliers = train_df.outliers
target = train_df.target

#nan mean imputer:
from sklearn.preprocessing import Imputer
#from sklearn.impute import SimpleImputer
imp = Imputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)
#scale:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data=scaler.fit_transform(data)


# In[ ]:


print(train_df.shape, test_df.shape, data.shape)


# In[ ]:


#TSNE train
tsne_model = TSNE(n_jobs=-1, verbose=1, random_state=42)#, n_iter=100) #default n_iter=1000
embeddings=tsne_model.fit_transform(data)
#only train Iteration 1000: error is 4.964520
#full Iteration 1000: error is 5.587587


# In[ ]:


tsne_model.kl_divergence_


# if you want to export:
# pd.DataFrame(embeddings).to_csv('full_tsne_2d_embeddings.csv', index=True)

# In[ ]:


#TSNE visualization

c = np.array([10 if i>=train_len else outliers[i]*18 for i in range(df.shape[0])])
print('color map:', c.shape)
print('counts:', pd.value_counts(c))
print('embeddings shape:',embeddings.shape)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

plt.figure(1, figsize=(20, 10))
plt.scatter(vis_x, vis_y, c=c,cmap=plt.cm.get_cmap("jet", 20), marker='.', alpha=0.5)
plt.colorbar(ticks=range(20))
plt.clim(1, 20)
plt.title('full dataset TSNE-2D 18 - outliers, 0 - train, 10 - test')
plt.tight_layout()
plt.savefig('full_tsne_2d_embeddings_outliers.png')
plt.show()


# In[ ]:


plt.figure(1, figsize=(20, 10))

#c2 = np.array([10 if i>=train_len else target[i]+10 for i in range(df.shape[0])])
#plt.scatter(vis_x, vis_y, c=target+10,cmap=plt.cm.get_cmap("jet", 20), marker='.')

c2 = np.array([target[i]+8 if target[i]<30 else 18 for i in range(train_len)])
plt.scatter(vis_x[:train_len], vis_y[:train_len], c=c2[:train_len],cmap=plt.cm.get_cmap("jet", 20), marker='.', alpha=0.5)
plt.colorbar(ticks=range(20))
plt.clim(1, 20)
plt.title('full dataset TSNE-2D target (19 - train)')
plt.tight_layout()
plt.savefig('full_tsne_2d_embeddings_targets.png')
plt.show()


# 
