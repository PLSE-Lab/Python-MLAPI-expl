#!/usr/bin/env python
# coding: utf-8

# # Mercari Price Challenge -Ridge Regression,  LightGBM

# ![](https://lh3.googleusercontent.com/EfS3_Du-Af957T9VKhKAsazAOf_jo6urlXPUviz4rZ2KGa-69gHvDRMjgNJNPMLvkF-WFJWikJlkAllsUKMI1c0Y5mUT8ghcRvtB13R0ODMheXW1mrdrRgYjetXo6jTUtiL6wrwm5YRNoKFmaUcvToECEE_LYn-TC5C8wJtpqTQad1Q6Nhcg-0qJ40bYs8SdkO8soFgzTBilw6V-nK1a55Yv5WavggpT2TaTF7bbHRsIzHpk2pcXI1l92-bYTPOvyWsnWqlopuHPokC0yGDghwvciOT92nMmg-WeVcyqufRBTwhvogZ5FteFWTFDrrpkl4wi3_WTUAoJy0l5Ci72-3dp_01pSiHacbSfBPJO6cJJVJKNQ1TA_7O-lTn5CMZyUyfUuvk0D76WmxUAo4sSkgUNJw1lV3bsOZm-IGZ_UpyIZ5GSOLFXQChTQ1cNSjeEl-heUR7mT7-W8DCfAs-GCVQuOF-kdA1i9Dk5Cc9kYIg3Q1XiggPEB3lRN2sK9AxWH7L1on8-gUa5srx5xupKd9t4-y1PkCOZ1qicCRwlHLGErO3tmn-D9UTWxrn2VGQ3DroDiZwK45yiOHhSi1UhTQQC7aAmYvs-5x7gU3mibA=w1200-h630-no)

# I'm beginner Kaggler. This Kernel is based on following Prateek's kernel and Tilli's kernel.
# https://www.kaggle.com/iamprateek/submission-to-mercari-price-suggestion-challenge/notebook   
# https://www.kaggle.com/tilii7/cross-validation-weighted-linear-blending-errors  
# 
# Original Data supplied by Mercari is too large to analyze in comfort, so I use only 10,000 data.

# Firstly, I check data overview.

# In[1]:


import pandas as pd
train = pd.read_csv("../input/train.tsv", delimiter = '\t', nrows=10000)
print ("train.shape: " + str(train.shape))
train.head()


# ![](https://lh3.googleusercontent.com/uiPP_5ezjrLyqLCgoAVonxzpWFyom1aEZcTRLoKxf3nY-GpWAncB_1XKtHzXCQtkrF-HcEZPRI6DLhOk8hoCfCvxz5t5kg3ikioefRbhG89YNhdZjgPtOQVlmulYR-btvXcJhgRpcmBhu-0aww_d1vGD5EgFB7xHRSpQzUu-PDUIsNvQ8yzbKo5EcA0YacHijLTgJv2SI2FqZRMGIP5PXiY_FlvfbWWPBUHT31_3-TAGwqEV3_lxtmcj1O4OpHdfFLGayAC3dapYCtPehBYAU0Ekp4zUWXlSv2SkeK-D1tNNd43OBIjrwMoQ81yPZdUATrqB5cKu0t1-wNczWkYZLjAPoGzxVE0Fw9d_eik9i7L8yjgHqozCh_iZN0sJn_zEYhjbFuKo-MkrjObY-GZXOnZ4KMmnyj5Dv5EkppFB4WUdg25aH6ULE69OGZO6Hh6zhuaLzKyjtoOSXC3Wscm1oZScUx7k9Brft9L5jaXoSJ-rexi9RLWqgM13PNs9lgxTlLnRd8zk5Nt66SQNRAyRkCySXs1qZQvUQ2RrAkLdoX5YaOvbWx5yS2t32oI47i26SyHX5pWAvK-byizDwiMAXvw-qKCnWR8sv6ys2a8=w600-h461-no)

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10, 6))
plt.hist(train["price"])
plt.xlabel("price[$]")
plt.ylabel("count")
plt.title("Price Histogram")
plt.show()


# Most goods are traded within $200.  Check a detail ditribution of price within $200.

# In[3]:


plt.figure(figsize=(10, 6))
plt.hist(train["price"], bins=500)
plt.xlim(0, 200)
plt.xlabel("price[$]")
plt.ylabel("count")
plt.title("Price Histogram")
plt.show()


# It shows a price between $10 and $20 has the high frequency.

# Next, make price ditribution of each item category visuallize by using seaborn plot.

# In[4]:


import time
import seaborn as sns
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train["general_cat"], train["sub_cat1"], train["sub_cat2"] =     zip(*train['category_name'].apply(lambda x: split_cat(x)))
train.drop("category_name", axis=1)
start_time = time.time()
plt.figure(figsize=(16, 8))
ax = sns.violinplot(x="general_cat", y="price", data=train, inner=None)
ax = sns.swarmplot(x="general_cat", y="price", data=train, edgecolor="gray", hue="sub_cat1")
plt.xticks(rotation=30)
plt.ylim(0, 200)
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.25), ncol=5)
plt.show()
print('Showing graph took {} secs.'.format(time.time() - start_time))


# It took over 10 minutes to show the plot.  
# According to graph, it looks like Kids and Beauty category goods have a violin plot which shows relatively high density in low price. And Electronics category have high variation between low price and high price. All category has a step distribution of $10, $25, $40 and so on, not a continuous distribution.

# By the way, what is the most high price good in 10,000 goods?? 

# In[5]:


train.sort_values(by="price", ascending=False).head(1)


# Chanel Classic Flag Bag medium Caviar L $1,506
# ![](https://lh3.googleusercontent.com/iwuZkNjZ4MK3-dJVAGZVnxrl4Ax5PghfplkHaRGWRPRY-OJ9XA0-hkc3i_njySmXf3RXRW1bQIMq9RQCYaPL01nv_eZMDzNA4W_HAfloiLEkbcNRI6z9hf3DkmQm_9xPgN14OdPgIPs7EyGxWOzior19-SHUD_SjzaVvQdLf7VfL6wDop0LeBt3nCDlFtoGKj6aOWHFByKpVAH2Lr6mD1FUt4vldvrHidpqCayqFmLwIiGOMZABTxpkLOT0jgC4LBqWR9ewTqFoGmvTc1dku7_g5uRJJ0xyJs7Z-qnFp-DpNHSzG9Y-XZ-lZLH_NHNzXzNeNIzI1BVq2M4nPxq6uH9kbyJSE9DFMNdJLQ19FdHGk2bl5wsEm-9WK6AktV6Gm7XPAdRDran__1_4HMcUvJ7GBDecgjyoLKdvjlIEKrObnbZYB3IdYYJ4Kw45kfuQVqCZVsWRwdp3nIDWcqnFp32796hDThmpTh9O8U42b8z0IAQMnN4_tBpiilwkF8MFfL4RXeKNv9Xa7o8PPLGXu-LDTVLiL3d6NWCE29_4OQoWatT-vldrj_VlSO9EL-ArxVhrKmUbPOP3VmGnyV3QTKaOVijsqIQn5g-GYEcJyKQ=w577-h620-no)  

# Checking a data overview done. From here, I'll prepare for machine learning tool.  
# Firstly, complete NaN columns of category and brand name and item description with "missing". 

# In[6]:


def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['sub_cat1'].fillna(value='missing', inplace=True)
    dataset['sub_cat2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)
handle_missing_inplace(train)
train['brand_name'].value_counts().head()


# The brand name has over 40% missing columns.

# In[7]:


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:750]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:450]
    pop_category2 = dataset['sub_cat1'].value_counts().loc[lambda x: x.index != 'missing'].index[:450]
    pop_category3 = dataset['sub_cat2'].value_counts().loc[lambda x: x.index != 'missing'].index[:450]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['sub_cat1'].isin(pop_category2), 'sub_cat1'] = 'missing'
    dataset.loc[~dataset['sub_cat2'].isin(pop_category3), 'sub_cat2'] = 'missing'
cutting(train)
train['brand_name'].value_counts().head()


# Next, change the data type of category and item description to categorical data for extraction text data. 

# In[8]:


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['sub_cat1'] = dataset['sub_cat1'].astype('category')
    dataset['sub_cat2'] = dataset['sub_cat2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
to_categorical(train)
train.dtypes


# From here, analyze by using machine learning tool. 
# CountVectorizer class in scikit learn library to get the frequency of words in item name.

# In[9]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=5)
X_name = cv.fit_transform(train['name'])
print (X_name.shape)
occ = np.asarray(X_name.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
counts_cv.sort_values(by='occurrences', ascending=False).head(10)


# It shows data has 1,508 unique words.  
# Most frequent word is "pink". "lularoe" is the 5th frequent word. 
# ![](https://lh3.googleusercontent.com/8CDnwWJdPzECpe_txjAXrRB5e0xRnyHN23uUAcxTVpBGDQxo5-dnEzBZHuoyPeLY0kqAnh9gUgGe0jjk2LZjF85WGLn9OCYJGpWN-EmebxCUBWaLO7N8Sd_NxPvFAkOL0mEJX9j0D9dB9cdZySHwAq26h3r_s2fL_jOSCz2g4CXMEF_fX8jRk1hOGwIF-DTRik8wK6Ilq9gg7_5ZswE0BmNJYo-r92atRmfdz8r8zaEHo_LywHHWVQG8-WgWaLkqluwN9EL-3ZhBXmz6mhd4JcQSNO5ARAKnFq0eGRaUo1-x4fObf2pgHfDcZ1nL1X0j56pahj2eF3OC83J4MHV_ZZZOtDNFkO4q5LXu8ohtXmMv6c_MS1epbMZvt5EA3Ph4HWvDDLb--a--aQUXk-Kv4_lNfY4IYBb2c6GVLIeho0jXE4K7dsAItZlMyxyifw1iUPPrNhyIguAM5H-FFm4S2eHUR81uH-hfWSgIHWOYlVglhKheCfl98Ld0tYXJyw_Zi0qOtHemTHef94a0jsrjEwjEr3cTdSyfEswkUiQyOybgiAkO4RfeQTvfx97EjZ6OO2b171cJC_U9Y9MLySincgzPMbIKquvWydlGP--0UQ=w744-h400-no)

# Take same processing to category name.

# In[10]:


cv = CountVectorizer(min_df=5)
combine_category = [train["general_cat"], train["sub_cat1"], train["sub_cat2"]]
X_category1 = cv.fit_transform(train['general_cat'])
print ("----general_cat----")
print (X_category1.shape)
occ = np.asarray(X_category1.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
print (counts_cv.sort_values(by='occurrences', ascending=False).head())
X_category2 = cv.fit_transform(train['sub_cat1'])
print ("----sub_cat1----")
print (X_category2.shape)
occ = np.asarray(X_category2.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
print (counts_cv.sort_values(by='occurrences', ascending=False).head())
X_category3 = cv.fit_transform(train['sub_cat2'])
print ("----sub_cat2----")
print (X_category3.shape)
occ = np.asarray(X_category3.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': cv.get_feature_names(), 'occurrences': occ})
print (counts_cv.sort_values(by='occurrences', ascending=False).head())


# TfidfVectorizer class in scikit learn library to get the tf-idf value of words in item description.

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=1000,
                         ngram_range=(1, 3),
                         stop_words='english')
X_description = tv.fit_transform(train['item_description'])
weights = np.asarray(X_description.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': tv.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(10)


# Best tf-idf word is "description". It maybe contained in "Not description yet"

# LabelBinarizer class in scikit learn library to get the sparse matrix for a brand name.

# In[12]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(train['brand_name'])
occ = np.asarray(X_brand.sum(axis=0)).ravel().tolist()
counts_cv = pd.DataFrame({'term': lb.classes_, 'occurrences': occ})
counts_cv.sort_values(by='occurrences', ascending=False).head(10)


# csr_matrix class in scipy library to get the dummy value for a item condition id and a shipping id.

# In[13]:


from scipy.sparse import csr_matrix
X_dummies = csr_matrix(pd.get_dummies(train[['item_condition_id', 'shipping']],
                                          sparse=True).values)
X_dummies


# hstack class in scipy library to get the sparse matrix for above matrixes.

# In[14]:


from scipy.sparse import hstack
sparse_merge = hstack([X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name]).tocsr()
sparse_merge.shape


# From here, predict and verify about the price result of Ridge regression model and LightGBM.

# In[15]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = sparse_merge
y = np.log1p(train["price"])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 144) 

modelR = Ridge(alpha=.5, copy_X=True, fit_intercept=True, max_iter=100,
      normalize=False, random_state=101, solver='auto', tol=0.01)
modelR.fit(train_X, train_y)
predsR = modelR.predict(test_X)

def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

rmsleR = rmsle(predsR, test_y)
print ("Ridge Regression RMSLE = " + str(rmsleR))


# In[16]:


import lightgbm as lgb

train_XL1, valid_XL1, train_yL1, valid_yL1 = train_test_split(train_X, train_y, test_size = 0.1, random_state = 144) 
d_trainL1 = lgb.Dataset(train_XL1, label=train_yL1, max_bin=8192)
d_validL1 = lgb.Dataset(valid_XL1, label=valid_yL1, max_bin=8192)
watchlistL1 = [d_trainL1, d_validL1]
paramsL1 = {
        'learning_rate': 0.65,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 60,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.5,
        'nthread': 4
    }
modelL1 = lgb.train(paramsL1, train_set=d_trainL1, num_boost_round=8000, valid_sets=watchlistL1, early_stopping_rounds=5000, verbose_eval=500) 
predsL1 = modelL1.predict(test_X)
rmsleL1 = rmsle(predsL1, test_y)
print ("LightGBM1 RMSLE = " + str(rmsleL1))


# In[17]:


train_XL2, valid_XL2, train_yL2, valid_yL2 = train_test_split(train_X, train_y, test_size = 0.1, random_state = 101) 
d_trainL2 = lgb.Dataset(train_XL2, label=train_yL2, max_bin=8192)
d_validL2 = lgb.Dataset(valid_XL2, label=valid_yL2, max_bin=8192)
watchlistL2 = [d_trainL2, d_validL2]
paramsL2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
    }
modelL2 = lgb.train(paramsL2, train_set=d_trainL2, num_boost_round=5500, valid_sets=watchlistL2, early_stopping_rounds=5000, verbose_eval=500) 
predsL2 = modelL2.predict(test_X)
rmsleL2 = rmsle(predsL2, test_y)
print ("LightGBM2 RMSLE = " + str(rmsleL2))


# The rmsle scores are Ridge: 0.1504, LightGBM1:  0.1524, LightGBM2: 0.1587.
# Composite these predicted results. 

# In[18]:


preds = predsR*0.3 + predsL1*0.35 + predsL2*0.35
rmsle = rmsle(preds, test_y)
print ("Total RMSLE = " + str(rmsle))


# By compositing 3 prediction, it resulted 0.1435 score.  
# Finally, make the scatterplot of actual price and predicted price.

# In[19]:


actual_price = np.expm1(test_y)
preds_price = np.expm1(preds)

plt.figure(figsize=(12,10))
cm = plt.cm.get_cmap('winter')
x_diff = np.clip(100 * ((preds_price - actual_price) / actual_price), -75, 75)
plt.scatter(x=actual_price, y=preds_price, c=x_diff, s=10, cmap=cm)
plt.colorbar()
plt.plot([0, 100], [0, 100], 'k--', lw=1)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices [$]')
plt.ylabel('Predicted Prices [$]')
plt.show()


# That's all. Thanks.
