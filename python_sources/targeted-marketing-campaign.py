#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_excel("/kaggle/input/arketing-campaign/marketing_campaign.xlsx")
data.head()


# In[ ]:


data.describe()


# ## **EDA**

# In[ ]:


# Missing data
def missing_rep(df):
    miss = df.isna().sum()
    miss = miss[miss>0]
    miss_p = miss/df.shape[0]
    miss_t = miss_p>0.03

    return pd.DataFrame({"Missings" : miss, "Proportion of Missings" : miss_p, "Higher than 3%" : miss_t})


missing_rep(data)


# In[ ]:


def highlight_above_threshold(val):
    if val < .15:
        color = 'red'
    else:
        color = 'black'
    #color = 'red' if val < .15 else 'black'
    return 'color: %s' % color


# In[ ]:


feat_c = ["Education", "Marital_Status", "Kidhome", "Teenhome", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4",
         "AcceptedCmp5", "Complain"]

# Categorical features analysis
def cat_feat_describe(df, fc, target, n, thresh):

    fl = []
    if (type(fc)==list):
    
        for feature in fc:
            fl.append(df.groupby([feature]).agg({target : ["count", "mean"]}))    

            fm = pd.concat(fl, keys=fc)

            fm = pd.DataFrame({"Number of observations" : fm.iloc[:,0], "Discrimination ability" : fm.iloc[:,1],
                                 "More than n observations" : fm.iloc[:,0]>n})
    else:
        fm = (df.groupby(fc).agg({target : ["count", "mean"]}))
        
        fm = pd.DataFrame({"Number of observations" : fm.iloc[:,0], "Discrimination ability" : fm.iloc[:,1],
                                 "More than n observations" : fm.iloc[:,0]>n})
        
    return fm


feat_sum = cat_feat_describe(data, feat_c, "Response", 40, 0.15)
feat_sum.style.applymap(highlight_above_threshold)


# In[ ]:


data_ = data.copy()

low_discriminability_cat = ["Absurd", "Alone", "YOLO", "Married", "Together"]
data_['Marital_Status'].loc[data_['Marital_Status'].isin(low_discriminability_cat)] = 'Other'
data_.groupby("Marital_Status").count().index


# In[ ]:


low_discriminability_cat = ["Graduation", "2n Cycle", "Basic"]
data_['Education'].loc[data_['Education'].isin(low_discriminability_cat)] = 'Other'
data_.groupby("Education").count().index


# In[ ]:


data_['NumberOff'] = data_['Kidhome'] + data_['Teenhome']
feat_c.append("NumberOff")


# In[ ]:


def cat_feat_plot(df, fc, target, thresh):

    sns.set_style("whitegrid")    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.4)
    i=1
    
    for feat in fc:
        plot_df = cat_feat_describe(df, feat, target, 50, thresh).iloc[:,1]
        plot_df = plot_df.sort_values(ascending=False)
        plot_df = pd.DataFrame(plot_df)
        plot_df = pd.DataFrame(plot_df.reset_index())
        ax =sns.barplot(plot_df[feat], plot_df['Discrimination ability'], palette='magma')
        ax.set_ylabel("", size = 10)
        ax.axhline(y=thresh, color="black", ls='--')
        ax.set_xticklabels(plot_df.index, size=10)
        
        if i<11:
            plt.subplot(2, 5, i)
            i+=1
        
        fig.suptitle('Discrimination ability of categories', ha='center',
                     va='center', fontsize=15, y=0.92, fontweight='bold')
        fig.yaxis_title='Discrimination ability'
        fig.text(0.07, 0.5,'Discrimination ability', ha='center', va='center',
                 rotation='vertical', fontsize=15)
        fig.set_figheight(9)
        fig.set_figwidth(16)
        
cat_feat_plot(data_, feat_c, "Response", 0.15)


# #### Transforms date format in days

# In[ ]:


def days_since(dates_series, date_format):
    n = len(dates_series)
    result = [0] * n

    for i in range(n):
        result[i] = (datetime.today()-datetime.strptime(dates_series[i], date_format)).days
    
    return result

data_["Days_Customer"] = days_since(list(data_.Dt_Customer), "%Y-%m-%d")
data_ = data_.drop(columns="Dt_Customer")
data_["Days_Customer"].head()


# #### Describes numerical attributes

# In[ ]:


feat_n = list(data_.columns)
feat_n = list(filter(lambda x: x not in feat_c, feat_n))

feat_n.remove("ID") # Removing ID column

data_[feat_n].describe() # Describing only Numerical Variables


# #### Drops constant variables

# In[ ]:


std = data_[feat_n].describe().iloc[2,:]
const_lab = [std[std<0.05].index[0], std[std<0.05].index[1]]
std[std<0.05]


# In[ ]:


data_.drop(labels=const_lab, axis=1, inplace=True)
feat_n = list(filter(lambda x: x not in const_lab, feat_n))


# #### Removes inconsistant age

# In[ ]:


data_[(2020 - data_["Year_Birth"])>90]


# In[ ]:


data_ = data_[(2020 - data_["Year_Birth"])<=90]


# #### Drops rows with less missing *income* (about 1% of the data)

# In[ ]:


data_ = data_[~data_['Income'].isna()]
data_.shape


# #### Correlation matrix

# In[ ]:


# The function to "zoom" in the correlation matrix.
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

def corr_matrix(df):
    # Compute the correlation matrix
    corr = df.corr()
    cmap = sns.diverging_palette(5, 250, as_cmap=True)
    
    vis = corr.style.background_gradient(cmap, axis=1)            .set_properties(**{'max-width': '80px', 'font-size': '10pt'})            .set_caption("Hover to magify")            .set_precision(2)            .set_table_styles(magnify())

    return vis

feat_n_ = feat_n.copy()
feat_n.remove("Response") # Removing the Targer variable from the list of numerical features to be analyzed by correlation.
corr_matrix(data_[feat_n])


# In[ ]:


corr = data_[feat_n].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

plt.figure(figsize=(11,9))

cmap = sns.diverging_palette(5, 250, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.6, vmin=-.6, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Numerical features correlation matrix", fontsize=16)
plt.show()


# #### Distribution according to response

# In[ ]:


def num_feat_plot(df, feat_nlist, target, feat_clist = None):
    
    sns.set_style("darkgrid")    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.4)
    i=1
    
    if(target in feat_nlist):
        feat_nl = feat_nlist.copy()
        feat_nl.remove(target)

    
    for feat in feat_nl:
        sns.violinplot(data = df, y = feat, x = target)
        ax=sns.violinplot(data = df, y = feat, x = target, palette="Reds")
        ax.set_xlabel("", size = 10)
        
        if i<15:
            plt.subplot(3, 5, i)
            i+=1
        
        fig.suptitle('Distribution by response', ha='center',
                     va='center', fontsize=15, y=0.92, fontweight='bold')
        fig.text(0.5, 0.06, 'Response', ha='center', va='center', fontsize=14)
        fig.set_figheight(10)
        fig.set_figwidth(17)

    return

num_feat_plot(data_, feat_n_, "Response")


# #### Outlier detection

# In[ ]:


from sklearn.ensemble import IsolationForest

def anom_plot(df, num_feat_list, l, c):
    sns.set_style("whitegrid") 
    fig, axs = plt.subplots(l, c, figsize=(22, 12), facecolor='w', edgecolor='k')
    axs = axs.ravel()

    for i, column in enumerate(num_feat_list):
        isolation_forest = IsolationForest(n_estimators=500, contamination="auto")
        isolation_forest.fit(df[column].values.reshape(-1,1))

        xx = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1,1)
        anomaly_score = isolation_forest.decision_function(xx)
        outlier = isolation_forest.predict(xx)
    
        axs[i].plot(xx, anomaly_score, label='anomaly score')
        axs[i].fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                     where=outlier==-1, color='r', 
                     alpha=.4, label='outlier region')
        axs[i].legend()
        axs[i].set_title(column)
        
    fig.suptitle('Anomaly detection', ha='center',
                     va='center', fontsize=20, y=0.92, fontweight='bold')
        
    return
    
anom_plot(data_, feat_n, 3, 5)


# The following cell defines two utility functions to semi-automatically identify outliers, through univariate perspective, in a pandas.Series using standard deviation from the mean (filter_by_std) and interquartile range (filter_by_iqr).

# In[ ]:


def filter_by_iqr(series_, k=1.5, return_thresholds=False):
    q25, q75 = np.percentile(series_, 25), np.percentile(series_, 75)
    iqr = q75-q25
    
    cutoff = iqr*k
    lower_bound, upper_bound = q25-cutoff, q75+cutoff
    
    if return_thresholds:
        return lower_bound, upper_bound
    else:
        return [True if i < lower_bound or i > upper_bound else False for i in series_]


# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer

def bivariate_outlier_id_plot(df, list_num_features, target, n_bins=20):
    sns.set_style("darkgrid") 
    fig = plt.figure(figsize=(22, 12))
    color = "floralwhite"
    i=1
    for feature in list_num_features:
        if feature == "Income":
            ser = df[feature].copy()
            ser.dropna(inplace=True)
        else:
            ser = df[feature]
          
        # box plots
        thresholds = filter_by_iqr(ser, 1.5, True)
        outliers = df[[feature, target]][df[feature]>thresholds[1]]

        ax = fig.add_subplot(5, 3, i)

        box = ax.boxplot(ser, flierprops=dict(markerfacecolor='r', marker='s'), 
                         vert=False, patch_artist=True, sym="w")                                                                  
        ax.plot(outliers.iloc[:, 0][outliers.iloc[:, 1]==0], np.ones(sum(outliers.iloc[:, 1]==0)), color="black", marker = "o", markersize=4)
        ax.plot(outliers.iloc[:, 0][outliers.iloc[:, 1]==1], np.ones(sum(outliers.iloc[:, 1]==1)), color="red", marker = "D", markersize=6)
        ax.set_title(feature)
        box['boxes'][0].set_facecolor(color)

        i+=1
        
    fig.suptitle('Variable Distribution',  ha='center',
                     va='center', fontsize=20, y=1.03, fontweight='bold')

    plt.tight_layout()
    plt.show()

bivariate_outlier_id_plot(data_, feat_n, "Response", n_bins=20)


# ## **Split dataset**

# In[ ]:


seeds = [4, 56, 92, 105, 400]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_,
                                                    data_["Response"],
                                                    test_size=0.3,
                                                    random_state=seeds[0])


# #### Multivariate outlier detection

# In[ ]:


# Simple function to check if the matrix is positive definite 
#(for example, it will return False if the matrix contains NaN).
def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False 

# The function to calculate the Mahalanobis Distance. Returns a list of distances.
def MahalanobisDist(data):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            vars_mean = []
            for i in range(data.shape[0]):
                vars_mean.append(list(data.mean(axis=0)))
            diff = data - vars_mean
            md = []
            for i in range(len(diff)):
                md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
            return md
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")
        
def MD_detectOutliers(data, extreme=False):
    MD = MahalanobisDist(data)

    std = np.std(MD)
    k = 3. * std if extreme else 2. * std
    m = np.mean(MD)
    up_t = m + k
    low_t = m - k
    outliers = []
    for i in range(len(MD)):
        if (MD[i] >= up_t) or (MD[i] <= low_t):
            outliers.append(i)  # index of the outlier
    return np.array(outliers)


# In[ ]:


data_aux = X_train[feat_n_]
outliers_i = MD_detectOutliers(np.array(data_aux))
len(outliers_i)


# In[ ]:


outliers = pd.DataFrame()
for i in outliers_i:
    outliers = outliers.append(data_aux.iloc[i,:])
    
outliers.head()


# Drops the outliers

# In[ ]:


X_train = X_train[~X_train.index.isin(outliers.index)]
y_train = y_train[~y_train.index.isin(outliers.index)]


# ## **Feature Engineering**

# Business-oriented features

# In[ ]:


# Percentage of Monetary Units spent on gold products out of the total spent
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = X_train["MntGoldProds"].iloc[i]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
    
X_train["PrpGoldProds"] = aux
X_train["PrpGoldProds"].head()


# In[ ]:


aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = X_test["MntGoldProds"].iloc[i]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
    
X_test["PrpGoldProds"] = aux
X_test["PrpGoldProds"].head()


# In[ ]:


# Number of Accepted Campaigns out of the last 5 Campaigns
aux = [0]* X_train.shape[0]


for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])
    
    
X_train["NmbAccCmps"] = aux
X_train["NmbAccCmps"].head()


# In[ ]:


# Number of Accepted Campaigns out of the last 5 Campaigns
aux = [0]* X_test.shape[0]


for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])
    
    
X_test["NmbAccCmps"] = aux
X_test["NmbAccCmps"].head()


# In[ ]:


# Proportion of Accepted Campaigns out of the last 5 Campaigns
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])/5
    
X_train["PrpAccCmps"] = aux
X_train["PrpAccCmps"].head()


# In[ ]:


# Proportion of Accepted Campaigns out of the last 5 Campaigns
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']].iloc[i,:])/5
    
X_test["PrpAccCmps"] = aux
X_test["PrpAccCmps"].head()


# In[ ]:


# Proportion of Monetary Units spent on Wine out of the total spent
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntWines"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PrpWines"] = aux
X_train["PrpWines"].head()


# In[ ]:


# Proportion of Monetary Units spent on Wine out of the total spent
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntWines"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PrpWines"] = aux
X_test["PrpWines"].head()


# In[ ]:


# Proportion of Monetary Units spent on Fruits out of the total spent
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntFruits"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PrpFruits"] = aux
X_train["PrpFruits"].head()


# In[ ]:


# Proportion of Monetary Units spent on Fruits out of the total spent
aux = [0]* X_test.shape[0]


for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntFruits"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PrpFruits"] = aux
X_test["PrpFruits"].head()


# In[ ]:


# Proportion of Monetary Units spent on Meat out of the total spent
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntMeatProducts"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PrpMeat"] = aux
X_train["PrpMeat"].head()


# In[ ]:


# Proportion of Monetary Units spent on Meat out of the total spent
aux = [0]* X_test.shape[0]


for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntMeatProducts"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PrpMeat"] = aux
X_test["PrpMeat"].head()


# In[ ]:


# Proportion of Monetary Units spent on Fish out of the total spent
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(X_train[["MntFishProducts"]].iloc[i,:]/sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_train["PrpFish"] = aux
X_train["PrpFish"].head()


# In[ ]:


# Proportion of Monetary Units spent on Fish out of the total spent
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(X_test[["MntFishProducts"]].iloc[i,:]/sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:]))
    
X_test["PrpFish"] = aux
X_test["PrpFish"].head()


# In[ ]:


# Monetary
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
X_train["Mnt"] = aux
X_train["Mnt"].head()


# In[ ]:


# Monetary
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])
    
X_test["Mnt"] = aux
X_test["Mnt"].head()


# In[ ]:


# Buy Potential
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = float(sum(X_train[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])/((X_train[["Income"]].iloc[i,:])*2))   
    
X_train["BuyPot"] = aux
X_train["BuyPot"].head()


# In[ ]:


# Buy Potential
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = float(sum(X_test[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']].iloc[i,:])/((X_test[["Income"]].iloc[i,:])*2))
    
X_test["BuyPot"] = aux
X_test["BuyPot"].head()


# In[ ]:


# Frequency
aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    aux[i] = sum(X_train[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].iloc[i,:])
    
X_train["Freq"] = aux
X_train["Freq"].head()


# In[ ]:


# Frequency
aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    aux[i] = sum(X_test[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].iloc[i,:])
    
X_test["Freq"] = aux
X_test["Freq"].head()


# In[ ]:


# Creating RFM feature using Recency, Freq and Mnt:
feature_list, n_bins = ["Recency", "Freq", "Mnt"], 5
rfb_dict = {}
for feature in feature_list:
    bindisc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile")
    feature_bin = bindisc.fit_transform(X_train[feature].values[:, np.newaxis])
    feature_bin = pd.Series(feature_bin[:, 0], index=X_train.index)
    feature_bin += 1
    
    if feature == "Recency":
        feature_bin = feature_bin.sub(5).abs() + 1
    rfb_dict[feature+"_bin"] = feature_bin.astype(int).astype(str)

X_train["RFM"] = (rfb_dict['Recency_bin'] + rfb_dict['Freq_bin'] + rfb_dict['Mnt_bin']).astype(int)
X_train.head()


# In[ ]:


# Creating RFM feature using Recency, Freq and Mnt:
feature_list, n_bins = ["Recency", "Freq", "Mnt"], 5
rfb_dict = {}
for feature in feature_list:
    bindisc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile")
    feature_bin = bindisc.fit_transform(X_test[feature].values[:, np.newaxis])
    feature_bin = pd.Series(feature_bin[:, 0], index=X_test.index)
    feature_bin += 1
    
    if feature == "Recency":
        feature_bin = feature_bin.sub(5).abs() + 1
    rfb_dict[feature+"_bin"] = feature_bin.astype(int).astype(str)

X_test["RFM"] = (rfb_dict['Recency_bin'] + rfb_dict['Freq_bin'] + rfb_dict['Mnt_bin']).astype(int)
X_test.head()


# Scales the features

# In[ ]:


feat_n.extend(('PrpGoldProds',
       'NmbAccCmps', 'PrpAccCmps', 'PrpWines', 'PrpFruits', 'PrpMeat',
       'PrpFish', 'Mnt', 'BuyPot', 'Freq', 'RFM'))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

suffix = "_t"

data_scaler = X_train[feat_n]
data_scaler_test = X_test[feat_n]

fscaler = MinMaxScaler()
scaled_d = fscaler.fit_transform(data_scaler.values)
scaled_d_test = fscaler.transform(data_scaler_test.values)

colnames = [s + suffix for s in data_scaler.columns]

X_train = pd.concat([X_train, pd.DataFrame(scaled_d, index=data_scaler.index, columns=colnames)], axis=1)
X_test = pd.concat([X_test, pd.DataFrame(scaled_d_test, index=data_scaler_test.index, columns=colnames)], axis=1)

X_train.head()


# Box-Cox transformation

# In[ ]:


from scipy import stats
# Receives a dataframe consisting only of scaled features and the target, and the name of the target feature.
# Returns both the dataframe with the features already transformed to the best transformation and a dictionary
# containing the name of each feature with its best transformation name.
def power_transf(df, target_feat):

    # define a set of transformations
    trans_dict = {"x": lambda x: x, "log": np.log, "sqrt": np.sqrt, 
                  "exp": np.exp, "**1/4": lambda x: np.power(x, 0.25), 
                  "**2": lambda x: np.power(x, 2), "**4": lambda x: np.power(x, 4)}

    target = target_feat
    best_power_dict = {}
    for feature in df.columns[:-1]:
        max_test_value, max_trans, best_power_trans = 0, "", None
        for trans_key, trans_value in trans_dict.items():
            # apply transformation
            feature_trans = trans_value(df[feature])
            if trans_key == "log":
                feature_trans.loc[np.isfinite(feature_trans)==False] = -50.

            # bin feature
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            feature_bin = bindisc.fit_transform(feature_trans.values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=df.index)

            # obtain contingency table
            df_ = pd.DataFrame(data={feature: feature_bin, target: df[target]})
            cont_tab = pd.crosstab(df_[feature], df_[target], margins = False)        

            # compute Chi-Squared
            chi_test_value = stats.chi2_contingency(cont_tab)[0]
            if chi_test_value > max_test_value:
                max_test_value, max_trans, best_power_trans = chi_test_value, trans_key, feature_trans      

        best_power_dict[feature] = (max_test_value, max_trans, best_power_trans)
        df[feature] = best_power_trans
        
    return df, best_power_dict


# In[ ]:


def power_transf(X_train, X_test, target_feat):

    # define a set of transformations
    trans_dict = {"x": lambda x: x, "log": np.log, "sqrt": np.sqrt, 
                  "exp": np.exp, "**1/4": lambda x: np.power(x, 0.25), 
                  "**2": lambda x: np.power(x, 2), "**4": lambda x: np.power(x, 4)}

    target = target_feat
    best_power_dict = {}
    for feature in X_train.columns[:-1]:
        max_test_value, max_trans, best_power_trans = 0, "", None
        for trans_key, trans_value in trans_dict.items():
            # apply transformation
            feature_trans = trans_value(X_train[feature])
            feature_trans_test = trans_value(X_test[feature])
            if trans_key == "log":
                feature_trans.loc[np.isfinite(feature_trans)==False] = -50.
                feature_trans_test.loc[np.isfinite(feature_trans_test)==False] = -50.

            # bin feature
            bindisc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy="uniform")
            feature_bin = bindisc.fit_transform(feature_trans.values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=X_train.index)

            # obtain contingency table
            df_ = pd.DataFrame(data={feature: feature_bin, target: X_train[target]})
            cont_tab = pd.crosstab(df_[feature], df_[target], margins = False)        

            # compute Chi-Squared
            chi_test_value = stats.chi2_contingency(cont_tab)[0]
            if chi_test_value > max_test_value:
                max_test_value, max_trans, best_power_trans = chi_test_value, trans_key, feature_trans      

        best_power_dict[feature] = (max_test_value, max_trans, best_power_trans)
        X_train[feature] = best_power_trans
        
    return X_train, X_test, best_power_dict


# In[ ]:


# We need to create a dataframe containing only the scaled features with the Response.
df_pt = X_train.iloc[:,-15:]
df_pt_test = X_test.iloc[:,-15:]

df_pt["Response"] = X_train["Response"]

data_pt, data_pt_test, best_pt = power_transf(df_pt, df_pt_test, "Response")

print("Best Power Transformation for each feature:")
for key in best_pt:
    print("\t->", key, best_pt[key][1])


# In[ ]:


# Replacing the old columns of scaled features with the features transformed according to the best transformation
coln = data_pt.columns[:-1]

X_train.drop(columns=coln, inplace=True)
X_train[coln] = data_pt[coln]

X_test.drop(columns=coln, inplace=True)
X_test[coln] = data_pt_test[coln]

X_train.head()


# Merges categories with low discrimination ability

# In[ ]:


# Merging Categories
# in Marital_Status:  "Single" as 3, "Widow" as 2, "Divorced" as 1 and ["Married", "Together"] as 0
X_train["Marital_Status_bin"] = X_train['Marital_Status'].apply(lambda x: 3 if x == "Single" else
                                                            (2 if x == "Widow" else
                                                             (1 if x == "Divorced" else 0))).astype(int)

X_test["Marital_Status_bin"] = X_test['Marital_Status'].apply(lambda x: 3 if x == "Single" else
                                                            (2 if x == "Widow" else
                                                             (1 if x == "Divorced" else 0))).astype(int)


# In[ ]:


# in Education: "Phd" as 2, "Master" as 1 and ['Graduation', 'Basic', '2n Cycle'] as 0
X_train["Education_bin"] = X_train['Education'].apply(lambda x: 2 if x == "PhD" else (1 if x == "Master" else 0)).astype(int)

X_test["Education_bin"] = X_test['Education'].apply(lambda x: 2 if x == "PhD" else (1 if x == "Master" else 0)).astype(int)


# In[ ]:


X_train["Kidhome"] = X_train['Kidhome'].astype(int)
X_train["Teenhome"] = X_train['Teenhome'].astype(int)
X_train["NumberOff"] = X_train['NumberOff'].astype(int)

X_test["Kidhome"] = X_test['Kidhome'].astype(int)
X_test["Teenhome"] = X_test['Teenhome'].astype(int)
X_test["NumberOff"] = X_test['NumberOff'].astype(int)


# In[ ]:


aux = [0]* X_train.shape[0]

for i in range(X_train.shape[0]):
    if(int(X_train[["Kidhome"]].iloc[i,:])+int(X_train[["Teenhome"]].iloc[i,:])>0):
        aux[i] = 1
    else:
        aux[i] = 0
    
X_train["HasOffspring"] = aux
X_train["HasOffspring"].head()


# In[ ]:


aux = [0]* X_test.shape[0]

for i in range(X_test.shape[0]):
    if(int(X_test[["Kidhome"]].iloc[i,:])+int(X_test[["Teenhome"]].iloc[i,:])>0):
        aux[i] = 1
    else:
        aux[i] = 0
    
X_test["HasOffspring"] = aux
X_test["HasOffspring"].head()


# In[ ]:


# Adding these new categorical features to the list:
feat_c.append("Marital_Status_bin")
feat_c.append("Education_bin")
feat_c.append("HasOffspring")

# Our data now:
X_train.head()


# PCA

# In[ ]:


X_train.drop(["ID",'Education','Marital_Status','Year_Birth','Year_Birth',
            'Income', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'Days_Customer', 'PrpGoldProds','NmbAccCmps', 'PrpAccCmps', 'PrpWines',
            'PrpFruits', 'PrpMeat','PrpFish', 'Mnt', 'BuyPot', 'Freq', 'RFM'], 
           axis=1, inplace=True)

X_test.drop(["ID",'Education','Marital_Status','Year_Birth','Year_Birth',
            'Income', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'Days_Customer', 'PrpGoldProds','NmbAccCmps', 'PrpAccCmps', 'PrpWines',
            'PrpFruits', 'PrpMeat','PrpFish', 'Mnt', 'BuyPot', 'Freq', 'RFM'], 
           axis=1, inplace=True)


# In[ ]:


from sklearn.decomposition import PCA
columns = X_train.columns
columns = columns.drop(['Kidhome','Teenhome','NumberOff','AcceptedCmp3', 'AcceptedCmp4',
                        'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response',
                        'Marital_Status_bin', 'Education_bin', 'HasOffspring'])


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train[columns])
principalComponents_test = pca.transform(X_test[columns])

X_train["pc1"] = principalComponents[:,0]
X_train["pc2"] = principalComponents[:,1]
X_test["pc1"] = principalComponents_test[:,0]
X_test["pc2"] = principalComponents_test[:,1]

X_train.head()


# In[ ]:


feat_n =['Year_Birth_t', 'Income_t', 'Recency_t', 'MntWines_t', 'MntFruits_t',
       'MntMeatProducts_t', 'MntFishProducts_t', 'MntSweetProducts_t',
       'MntGoldProds_t', 'NumDealsPurchases_t', 'NumWebPurchases_t',
       'NumCatalogPurchases_t', 'NumStorePurchases_t', 'NumWebVisitsMonth_t',
       'Days_Customer_t', 'PrpGoldProds_t', 'NmbAccCmps_t', 'PrpAccCmps_t',
       'PrpWines_t', 'PrpFruits_t', 'PrpMeat_t', 'PrpFish_t', 'Mnt_t',
       'BuyPot_t', 'Freq_t', 'RFM_t', 'pc1', 'pc2']


# In[ ]:


feat_c.remove('Education')
feat_c.remove('Marital_Status')


# #### Gets dummies for categories

# In[ ]:


cat_columns = ['Kidhome', 'Teenhome', 'NumberOff','Marital_Status_bin','Education_bin']
X_train = pd.get_dummies(X_train, prefix_sep="_",
                              columns=cat_columns)
X_test = pd.get_dummies(X_test, prefix_sep="_",
                              columns=cat_columns)
X_train.head()


# In[ ]:


feat_c.remove('Kidhome')
feat_c.remove('Teenhome')
feat_c.remove('NumberOff')
feat_c.remove('Marital_Status_bin')
feat_c.remove('Education_bin')

feat_c.extend(('Kidhome_0', 'Kidhome_1',
       'Kidhome_2', 'Teenhome_0', 'Teenhome_1', 'Teenhome_2', 'NumberOff_0',
       'NumberOff_1', 'NumberOff_2', 'NumberOff_3', 'Marital_Status_bin_0',
       'Marital_Status_bin_1', 'Marital_Status_bin_2', 'Marital_Status_bin_3',
       'Education_bin_0', 'Education_bin_1', 'Education_bin_2'))


# ## **Feature Selection**

# In[ ]:


# Is given as input a dataframe, a list of continuous features names, a list of categorical features names,
# the name of the target feature and returns a dataframe with the discrimination ability of each feature and if
# its p-value is lower than 0.05.
# 10 is the default number of bins and uniform is the strategy used in the binning of continuous features.
def chisq_ranker(df, continuous_flist, categorical_flist, target, n_bins=10, binning_strategy="uniform"):
    chisq_dict = {}
    if  continuous_flist:
        bindisc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', 
                               strategy=binning_strategy)
        for feature in continuous_flist:            
            feature_bin = bindisc.fit_transform(df[feature].values[:, np.newaxis])
            feature_bin = pd.Series(feature_bin[:, 0], index=df.index)
            cont_tab = pd.crosstab(feature_bin, df[target], margins = False)
            chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2] 
    if  categorical_flist:
        for feature in categorical_flist:  
            cont_tab = pd.crosstab(df[feature], df[target], margins = False)          
            chisq_dict[feature] = stats.chi2_contingency(cont_tab.values)[0:2]       
    
    df_chi = pd.DataFrame(chisq_dict, index=["Chi-Squared", "p-value"]).transpose()
    df_chi.sort_values("Chi-Squared", ascending=False, inplace=True)
    df_chi["valid"]=df_chi["p-value"]<=0.05
    
    
    return df_chi


# In[ ]:


df_chisq_rank = chisq_ranker(X_train, feat_n, feat_c, "Response")
df_chisq_rank.head(15)


# In[ ]:


sns.set_style('whitegrid') 

plt.subplots(figsize=(13,12))
pal = sns.color_palette("RdBu_r", len(df_chisq_rank))
rank = df_chisq_rank['Chi-Squared'].argsort().argsort()  

sns.barplot(y=df_chisq_rank.index,x=df_chisq_rank['Chi-Squared'], palette=np.array(pal[::-1])[rank])
plt.title("Features' worth by Chi-Squared statistic test", fontsize=18)
plt.ylabel("Input feature", fontsize=14)
plt.xlabel("Chi-square", fontsize=14)

plt.show()


# ## **Balance training set**

# In[ ]:


y_train.value_counts().plot(kind='bar', title='Count (target)');


# #### Oversampling

# In[ ]:


from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

y_train.value_counts().plot(kind='bar', title='Count (target)');


# ## Models

# In[ ]:


scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.values))
X_test = scaler.transform(X_test)


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LR = LogisticRegression()
LR.fit(X_train, y_train)


# In[ ]:


y_pred = LR.predict(X_test)
print('ROC score: {}'.format(roc_auc_score(y_test, y_pred)))


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# Naive Bayes

# In[ ]:


NB = GaussianNB()
NB.fit(X_train, y_train)


# In[ ]:


y_pred = LR.predict(X_test)
print('ROC score: {}'.format(roc_auc_score(y_test, y_pred)))


# DNN

# In[ ]:


from keras import models
from keras import layers
import tensorflow as tf

model = models.Sequential()
model.add(layers.Dense(5, activation='relu', input_dim=53))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])


# In[ ]:


history = model.fit(X_train, y_train, epochs=30)


# In[ ]:


cnn_model = model
metrics = cnn_model.evaluate(X_test, y_test)
print("{}: {}".format(cnn_model.metrics_names[1], metrics[1]))


# ## Results

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

