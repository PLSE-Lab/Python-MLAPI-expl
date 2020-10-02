#!/usr/bin/env python
# coding: utf-8

# Introduction to **Inter-American Development Bank** and **Costa Rican Household Poverty Level Prediction
# **
# The Inter-American Development Bank (IADB or IDB or BID) is the largest source of development financing for Latin America and the Caribbean.[1] Established in 1959, the IDB supports Latin American and Caribbean economic development, social development and regional integration by lending to governments and government agencies, including State corporations.
# 
# Poverty headcount ratio at national poverty line (% of population) in Costa Rica was reported at 20.5 % in 2016, according to the World Bank collection of development indicators, compiled from officially[ recognized sources](https://tradingeconomics.com/costa-rica/poverty-headcount-ratio-at-national-poverty-line-percent-of-population-wb-data.html).
# 
# <a href="https://ibb.co/h2k1ry"><img src="https://preview.ibb.co/ckOXyd/Screenshot_from_2018_07_20_08_50_31.png" alt="Screenshot_from_2018_07_20_08_50_31" border="0"></a>
# 
# The data science is so amazingly universal now that we can help find cancerous nuclie, to predit stocks, recommendations, save lives and with this competition we get to help people find their **SWEET HOME**.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from scipy.special import boxcox
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#PLOTLY
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.graph_objs import Scatter, Figure, Layout
cf.set_config_file(offline=True)


# In[ ]:


print(">> Loading Data...")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Quick Look on shapes ...

# In[ ]:


print("Train shape {}".format(train.shape))
print("Test shape {}".format(test.shape))


# In[ ]:


target = train['Target'].astype('int')


# ## Target Distributions

# In[ ]:


data = [go.Histogram(x=target)]
layout = go.Layout(title = "Target Histogram")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Lets have a look at unique value counts of the target variable

# ## Target Value Counts

# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(train.Target)
plt.title("Value Counts of Target Variable")


# ## Numer of Missing values in datasets

# In[ ]:


print(f"Numer of Missing values in train: ", train.isnull().sum().sum())
print(f"Number of Missing values in train: ", test.isnull().sum().sum())


# ## Correlations with target variable

# In[ ]:


from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

labels = []
values = []
for col in train.columns:
    if col not in ["Id", "Target"]:
        labels.append(col)
        values.append(spearmanr(train[col].values, train["Target"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='red')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# ## Correlation Heatmap of Top 50 correlated features with target

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train[corr_df.col_labels[:50]].corr())


# ## Heatmap of top 10 correlated features

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train[corr_df.col_labels[:10]].corr(), annot=True)


# ## Finding Feature Importance with Light GBM

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# Remember to Label, OneHot and Dummy Encode your features. As most of them are categorical here ...

# In[ ]:


train.head()


# In[ ]:


train.drop(['Id','Target'], axis=1, inplace=True)


# In[ ]:


obj_columns = [f_ for f_ in train.columns if train[f_].dtype == 'object']
for col in tqdm(obj_columns):
    le = LabelEncoder()
    le.fit(train[col].astype(str))
    train[col] = le.transform(train[col].astype(str))


# In[ ]:


lgbm = LGBMClassifier()
xgbm = XGBClassifier()
train = train.astype('float32') # For faster computation
lgbm.fit(train, target , verbose=False)
xgbm.fit(train, target ,verbose=False)


# ## Feature Importance by Duo
# 
# I always go with LGBM and XGBoost Model Importances. Reason being they are quick and always catch the most relevant features and interactions. Also, at this point I would like to caution the use case of regular feature importance by Random Forest. 
# 
# **The scikit-learn Random Forest feature importance and R's default Random Forest feature importance strategies are biased. To get reliable results in Python, use permutation importance, provided here and in  rfpimp package (via pip). For R, use importance=T in the Random Forest constructor then type=1 in R's importance() function. In addition, your feature importance measures will only be reliable if your model is trained with suitable hyper-parameters.**

# In[ ]:


LGBM_FEAT_IMP = pd.DataFrame({'Features':train.columns, "IMP": lgbm.feature_importances_}).sort_values(by='IMP', ascending=False)

XGBM_FEAT_IMP = pd.DataFrame({'Features':train.columns, "IMP": xgbm.feature_importances_}
                            ).sort_values(
                              by='IMP', ascending=False)


# Top 10 features as seen by LightGBM model and XGBM model

# In[ ]:


LGBM_FEAT_IMP.head(10).transpose()


# In[ ]:


XGBM_FEAT_IMP.head(10).transpose()


# In[ ]:


data = [go.Bar(
            x= LGBM_FEAT_IMP.head(50).Features,
            y= LGBM_FEAT_IMP.head(50).IMP, 
            marker=dict(color='green',))
       ]
layout = go.Layout(title = "LGBM Top 50 Feature Importances")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


data = [go.Bar(
            x= XGBM_FEAT_IMP.head(50).Features,
            y= XGBM_FEAT_IMP.head(50).IMP, 
            marker=dict(color='blue',))
       ]
layout = go.Layout(title = "XGBM Top 50 Feature Importances")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## Difference of Describe of Train - Test

# Taking Mutual Importance features by two models

# In[ ]:


cols_imp = list(set(LGBM_FEAT_IMP[LGBM_FEAT_IMP.IMP > 0 ].Features.values) & set(
    XGBM_FEAT_IMP[XGBM_FEAT_IMP.IMP > 0 ].Features.values))
MUTUAL_50 = cols_imp[:50]
DIFF_DESCRIBE = train[MUTUAL_50].describe().transpose() - test[MUTUAL_50].describe().transpose()
DIFF_DESCRIBE.style.format("{:.2f}").bar(align='mid', color=['#d65f5f', '#5fba7d'])


# ## Unsupervised Clustering

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist


# **Let's find if the unique value of the target match the optimal K cluster through Unsupervised Learning K Means Algorithm**
# 
# **How many number of clusters? How to decide ?**
# 
# As unsupervised models dont have a metric as true labels are absent. We often wonder what should be the actual k values.
# 
# There are two methods to determine the optimum k clusters:
# 
# **1. Elbow Method**
# 
# This method looks at the percentage of variance explained as a function of the number of clusters: One should choose a number of clusters so that adding another cluster doesn't give much better modeling of the data. More precisely, if one plots the percentage of variance explained by the clusters against the number of clusters, the first clusters will add much information (explain a lot of variance), but at some point the marginal gain will drop, giving an angle in the graph. The number of clusters is chosen at this point, hence the "elbow criterion".
# 
# 
# **Silhouette Scores**
# a
# The Silhouette Coefficient is a metric to estimate the optimum number of clusters. It uses average intra-cluster distance and average nearest-cluster distance for each sample. Higher the value of the score, the better the estimate. Typically the silhoutte scores go high and then fall peaking at an optimum cluster number. The values lie between -1.0 and 1.0.
# 
# Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.

# In[ ]:


X = train[cols_imp].dropna()
distortions = []
for k in tqdm(range(1,8)):
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])


# In[ ]:


# Create a trace
trace = go.Line(
    x = [1,2,3,4,5,6,7,8],
    y = distortions,
    line = dict(
    color = 'red',
    width = 2),
    mode = 'lines+markers',
    name = 'lines+markers'
)
data = [trace]
layout = go.Layout(title = "Elbow Method Optimal Clusters - 3 (From Graph)")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


from sklearn.metrics import silhouette_score
k_clusters = []
sil_coeffecients = []

for n_cluster in range(2,6):
    kmeans = KMeans(n_clusters = n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label)
    print("For n_clusters={}, Silhouette Coefficient = {}".format(n_cluster, sil_coeff))
    sil_coeffecients.append(sil_coeff)
    k_clusters.append(n_cluster)


# In[ ]:


# Create a trace
trace = go.Line(
    x = [1,2,3,4,5,6],
    y = sil_coeffecients,
    line = dict(
    color = 'orange',
    width = 2),
    mode = 'lines+markers',
    name = 'lines+markers'
)
data = [trace]
layout = go.Layout(title = "Silhouette Optimal Clusters - 3 (From Graph)")
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ## Whats next ? 
# 
# 1.  Baseline scores using XGBoost, CatBoost and LGBM
# 2. Recursive Feature Engineering
# 3. Categorical Feature Interaction
# 4. In dept feature engineering tutorials ahead...
