#!/usr/bin/env python
# coding: utf-8

# ## **Acknowledgements**
# #### This kernel uses such good kernels:
#    - https://www.kaggle.com/deepakdeepu8978/how-popular-a-song-is-according-to-spotify

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from scipy import stats
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


filename='/kaggle/input/top50spotify2019/top50.csv'
df=pd.read_csv(filename,encoding='ISO-8859-1')
df.head()


# ***univariate analysis***  :

# first we check the target **'Popularity'**

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(df.shape[0]), np.sort(df.Popularity.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
plt.show()


# we will check each Attribute and its distribution and its dependance with the target and check for the importance of the each Attribute with the target ....   :)

# **1.Genre**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Genre',data=df,color=color[2])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Genre in data',fontsize=12)
plt.xticks(rotation='vertical')
plt.title('frequence of Genre',fontsize=12)
plt.show()


# seems most of the songs are from Dance_pop genre 

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="Genre", y="Popularity", data=df)
plt.ylabel('Popularity', fontsize=12)
plt.xlabel('Genre Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How Popularity changes with Genre ?", fontsize=15)
plt.show()


# **2.Artist.Name**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Artist.Name',data=df,color=color[4])
plt.xlabel('Artist.Name in data given',fontsize=12)
plt.ylabel('cont of song sung',fontsize=12)
plt.xticks(rotation='vertical')
plt.title('Artist.Name and no.of song cont',fontsize=12)
plt.show()


# seems majority of the songs are by  Ed Sheeran now let combine the both and check ...! 

# In[ ]:


grouped_df = df.groupby(["Artist.Name", "Genre"])["Track.Name"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('Artist.Name', 'Genre', 'Track.Name')

plt.figure(figsize=(12,8))
sns.heatmap(grouped_df)
plt.title("Frequency of Artist.Name Vs Genre")
plt.show()


# In[ ]:


grouped_df = df.groupby(["Artist.Name", "Genre"])["Track.Name"].aggregate("count").reset_index()

fig, ax = plt.subplots(figsize=(12,20))
ax.scatter(grouped_df['Track.Name'].values, grouped_df["Artist.Name"].values)
for i, txt in enumerate(grouped_df.Genre.values):
    ax.annotate(txt, (grouped_df['Track.Name'].values[i], grouped_df["Artist.Name"].values[i]), rotation=45, ha='center', va='center', color='green')
plt.xlabel('Reorder Ratio')
plt.ylabel('department_id')
plt.title("Reorder ratio of different aisles", fontsize=15)
plt.show()


# In[ ]:


from wordcloud import WordCloud
plt.style.use('seaborn')
wrds1 = df["Artist.Name"].str.split("(").str[0].value_counts().keys()

wc1 = WordCloud(scale=5,max_words=1000,colormap="rainbow",background_color="white").generate(" ".join(wrds1))
plt.figure(figsize=(12,18))
plt.imshow(wc1,interpolation="bilinear")
plt.axis("off")
plt.title("Artist Name with more songs in data ",color='b')
plt.show()


# **3.Beats.Per.Minute**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="Beats.Per.Minute", data=df, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Beats.Per.Minute', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency distribution by Beats.Per.Minute order", fontsize=15)
plt.show()


# lets once check the skew() 

# In[ ]:


skew=df.skew()
print(skew)


# yes we have Liveness with +ve skew and Popularity with -ve skew() but as Popularity is our target we cant change anything 
# and we will change Liveness while deling with that 

# In[ ]:


plt.figure(figsize=(12,12))
sns.jointplot(x=df["Beats.Per.Minute"].values, y=df['Popularity'].values, size=10, kind="kde",color=color[4])
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Beats.Per.Minute", fontsize=12)
plt.title("Beats.Per.Minute Vs Popularity", fontsize=15)
plt.show()


# seems like with good beats.per.Minute are more getting good  popularity 

# **4.Energy**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=df.Energy.values,data=df,color=color[5])
plt.xlabel('Energy label',fontsize=12)
plt.ylabel('count')
plt.xticks(rotation='vertical')
plt.title('Energy label count',fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.jointplot(x=df["Energy"], y=df['Popularity'], size=10,kind="kde",color=color[6])
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Energy", fontsize=12)
plt.title("Energy Vs Popularity", fontsize=15)
plt.show()


# In[ ]:


df['Loudness..dB..'].value_counts()


# seems like according to the Loudness they rated the songs 

# In[ ]:


grouped_df = df.groupby(["Loudness..dB.."])["Popularity"].aggregate("count").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['Loudness..dB..'].values, grouped_df['Popularity'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# less songs with -9 

# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(x='Loudness..dB..', y='Popularity', data=df)
plt.xlabel('Loudness..dB..', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
plt.show()


# **5.Danceability**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x='Danceability',data=df,color=color[8])
plt.xlabel('Danceability count',fontsize=12)
plt.ylabel('count',fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.jointplot(x=df["Danceability"], y=df['Popularity'], size=10,kind="kde",color=color[6])
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Danceability", fontsize=12)
plt.title("Danceability Vs Popularity", fontsize=15)
plt.show()


# i thing songs should be with more energetic to Dance 

# In[ ]:


grouped_df = df.groupby(['Energy','Danceability'])['Popularity'].aggregate('mean').reset_index()

grouped_df = grouped_df.pivot('Energy', 'Danceability', 'Popularity')

plt.figure(figsize=(12,8))
sns.heatmap(grouped_df)
plt.title("Frequency of Energy Vs Danceability")
plt.show()


# **6.Liveness **

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=df.Liveness.values,data=df,color=color[5])
plt.xlabel('Liveness label',fontsize=12)
plt.ylabel('count')
plt.xticks(rotation='vertical')
plt.title('Liveness label count',fontsize=12)
plt.show()


# In[ ]:


#skew change 
transform=np.asarray(df[['Liveness']].values)
df_transform = stats.boxcox(transform)[0]


# In[ ]:


fig=plt.subplots(figsize=(10,10))
plt.title('Dependence between Liveness and popularity',fontsize=12)
sns.regplot(x='Liveness', y='Popularity',
            ci=None, data=df)
sns.kdeplot(df.Liveness,df.Popularity)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=df['Acousticness..'].values,data=df,color=color[5])
plt.xlabel('Acousticness label',fontsize=12)
plt.ylabel('count')
plt.xticks(rotation='vertical')
plt.title('Acousticness label count',fontsize=12)
plt.show()


# In[ ]:


fig=plt.subplots(figsize=(10,10))
plt.title('Dependence between Acousticness and popularity',fontsize=12)
sns.regplot(x='Acousticness..', y='Popularity',
            ci=None, data=df)
sns.kdeplot(df.Liveness,df.Popularity)
plt.show()


# In[ ]:


print(df.dtypes)


# now, Since there are so many variables, let us first take the 'int' variables alone and then get the correlation with the target variable to see how they are related.

# In[ ]:


xcol = [col for col in df.columns if col not in['Popularity'] if df[col].dtypes=='int64']

label = []
vales = []

for col in xcol:
    label.append(col)
    vales.append(np.corrcoef(df[col].values,df.Popularity.values)[0,1])
    
corr_df = pd.DataFrame({'col_label':label,'col_values':vales})
corr_df =corr_df.sort_values(by = 'col_values')


ind = np.arange(len(label))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.col_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_label.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# In[ ]:


corr_df_sel = corr_df.ix[(corr_df['col_values']>0.1) | (corr_df['col_values'] < -0.3)]
corr_df_sel


# so this are the more highly correlated with the target 

# We had an understanding of important variables from the univariate analysis. But this is on a stand alone basis and also we have linearity assumption. Now let us build a non-linear model to get the important variables by building Extra Trees model.

# In[ ]:


train_y = df['Popularity'].values
num_df = df[xcol].drop(['Unnamed: 0'],axis=1)
feat_name = num_df.columns.values

from sklearn import ensemble 
model = ensemble.ExtraTreesRegressor(n_estimators=25,max_depth=30,max_features=0.3, n_jobs=-1, random_state=0) 
model.fit(num_df,train_y)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indi = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indi)), importances[indi], color=color[4], yerr=std[indi], align="center")
plt.xticks(range(len(indi)), feat_name[indi], rotation='vertical')
plt.xlim([-1, len(indi)])
plt.show()


# seems like 'Valence','Energy' are more imporant variable and lets check with the XGB

# In[ ]:


import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(num_df, train_y, feature_names=num_df.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# Using xgboost, the important variables are 'Valence' followed by 'Acousticness' and 'Beats.Per.Minute'

# In[ ]:


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model


# more to come , will soon update with the modeling part ....
# 
# *your one upvote can more me to work more*  :)

# **pleace upvote if you liked it thank you **
