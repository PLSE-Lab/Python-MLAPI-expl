#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


This is  data analysis ( currently in progress also need modifications )
Please upvote this


# ## Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
color = sns.color_palette()
import ggplot
from ggplot import *
import xgboost
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# ## input files

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:





# In[ ]:


#train_df= pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
train_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
train_df.shape


# In[ ]:


train_df.head()


# In[ ]:


#pro_df = pd.read_csv("properties_2016.csv")
pro_df = pd.read_csv("../input/properties_2016.csv")
pro_df.shape


# In[ ]:


pro_df.head()


# In[ ]:


pro_df.head()


# In[ ]:





# In[ ]:





# In[ ]:


train_df= pd.merge(train_df,pro_df, on='parcelid', how='left')
train_df.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


train_df.head()


# In[ ]:





# In[ ]:





# # log error
# 

# In[ ]:





# In[ ]:



plt.figure(figsize=(8,6))
plt.scatter(range (train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# # cleaning

# In[ ]:





# In[ ]:


ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


missing_value= train_df.isnull().sum(axis=0).reset_index()
missing_value.columns=['column_name', 'missing_count']
missing_value['missing_ratio']=missing_value['missing_count'] / train_df.shape[0]
missing_value.ix[missing_value['missing_ratio']>0.99]


# In[ ]:





# In[ ]:


missing_value= pro_df.isnull().sum(axis=0).reset_index()
missing_value.columns=['column_name','missing_count']
missing_value = missing_value.ix[missing_value['missing_count']>0]
missing_value = missing_value.sort_values(by='missing_count')

ind=np.arange(missing_value.shape[0])
width=0.9
fig, ax= plt.subplots(figsize=(12,25))
rects=ax.barh(ind, missing_value.missing_count.values, color='green')

ax.set_yticks(ind)
ax.set_yticklabels(missing_value.column_name.values,rotation='horizontal')

plt.show()


# In[ ]:





# In[ ]:


missingValueColumns = train_df.columns[train_df.isnull().any()].tolist()


# In[ ]:





# In[ ]:


msno.heatmap(train_df[missingValueColumns],figsize=(20,20))


# ## variable types

# In[ ]:





# In[ ]:


pd.options.display.max_rows = 65

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df


# In[ ]:





# In[ ]:


dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:


dataTypeDf = pd.DataFrame(train_df.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
fig,ax = plt.subplots()
fig.set_size_inches(20,5)
sns.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax,color="#34495e")
ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")


# In[ ]:





# In[ ]:





# # log error

# In[ ]:





# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values, bins=50, kde= False)
plt.xlabel('logerror', fontsize=12)
plt.show


# In[ ]:





# In[ ]:





# it contains some outlairs on both end

# ## transaction date

# In[ ]:





# In[ ]:


train_df['transaction_month']= train_df['transactiondate'].dt.month

cnt_srs = train_df['transaction_month']. value_counts()
plt.figure(figsize=(12,7))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha= 0.8, color=color[1])

plt.xticks(rotation='vertical')
plt.xlabel('Monthof transaction')
plt.ylabel('Numberof occurences')
plt.show()


# In[ ]:





# In[ ]:


(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()


# # Additional features

# In[ ]:





# In[ ]:


for c in pro_df.columns:
    pro_df[c]=pro_df[c].fillna(-1)
    if pro_df[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(pro_df[c].values))
        pro_df[c] = lbl.transform(list(pro_df[c].values))


# In[ ]:


#life of property
train_df['N-life'] = 2018 - train_df['yearbuilt']

#error in calculation of the finished living area of home
train_df['N-LivingAreaError'] = train_df['calculatedfinishedsquarefeet']/train_df['finishedsquarefeet12']

#proportion of living area
train_df['N-LivingAreaProp'] = train_df['calculatedfinishedsquarefeet']/train_df['lotsizesquarefeet']
train_df['N-LivingAreaProp2'] = train_df['finishedsquarefeet12']/train_df['finishedsquarefeet15']

#Amout of extra space
train_df['N-ExtraSpace'] = train_df['lotsizesquarefeet'] - train_df['calculatedfinishedsquarefeet'] 
train_df['N-ExtraSpace-2'] = train_df['finishedsquarefeet15'] - train_df['finishedsquarefeet12'] 

#Total number of rooms
train_df['N-TotalRooms'] = train_df['bathroomcnt']*train_df['bedroomcnt']

#Average room size
train_df['N-AvRoomSize'] = train_df['calculatedfinishedsquarefeet']/train_df['roomcnt'] 

# Number of Extra rooms
train_df['N-ExtraRooms'] = train_df['roomcnt'] - train_df['N-TotalRooms'] 

#Ratio of the built structure value to land area
train_df['N-ValueProp'] = train_df['structuretaxvaluedollarcnt']/train_df['landtaxvaluedollarcnt']

#Does property have a garage, pool or hot tub and AC?
train_df['N-GarPoolAC'] = ((train_df['garagecarcnt']>0) & (train_df['pooltypeid10']>0) & (train_df['airconditioningtypeid']!=5))*1 

train_df["N-location"] = train_df["latitude"] + train_df["longitude"]
train_df["N-location-2"] = train_df["latitude"]*train_df["longitude"]
train_df["N-location-2round"] = train_df["N-location-2"].round(-4)

train_df["N-latitude-round"] = train_df["latitude"].round(-4)
train_df["N-longitude-round"] = train_df["longitude"].round(-4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Important Features selection

# In[ ]:


from sklearn import model_selection, preprocessing
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

mergedFilterd = train_df.fillna(-999)
for f in mergedFilterd.columns:
    if mergedFilterd[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(mergedFilterd[f].values)) 
        mergedFilterd[f] = lbl.transform(list(mergedFilterd[f].values))
        
train_y = mergedFilterd.logerror.values
train_X = mergedFilterd.drop(["parcelid", "transactiondate", "logerror"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)


# In[ ]:





# In[ ]:





# In[ ]:


featureImportance = model.get_fscore()
features = pd.DataFrame()
features['features'] = featureImportance.keys()
features['importance'] = featureImportance.values()
features.sort_values(by=['importance'],ascending=False,inplace=True)
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
plt.xticks(rotation=90)
sns.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="#34495e")


# In[ ]:





# In[ ]:





# In[ ]:


dtype_df1 = features['features']
dtype_df1.columns = ["Count", "Column Type"]
dtype_df1.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


dtype_df2 = features['importance']
dtype_df2.columns = ["Count", "Column Type"]
dtype_df2.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #  Univariate Analysis
# taking float varible then taking co relation with target values

# In[ ]:


# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = train_df.mean(axis=0)
train_df_new = train_df.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
   labels.append(col)
   values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
   
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
#autolabel(rects)
plt.show()
   


# In[ ]:





# In[ ]:


corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
for col in corr_zero_cols:
    print(col, len(train_df_new[col].unique()))


# In[ ]:





# In[ ]:


corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
corr_df_sel


# In[ ]:





# # correlation map

# In[ ]:


cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# In[ ]:





# Let us seee how the finished square feet 12 varies with the log error.

# In[ ]:


col = "finishedsquarefeet12"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Finished Square Feet 12', fontsize=12)
plt.title("Finished square feet 12 Vs Log error", fontsize=15)
plt.show()


# In[ ]:





# Calculated finished square feet:

# In[ ]:


col = "calculatedfinishedsquarefeet"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, size=10, color=color[5])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Calculated finished square feet', fontsize=12)
plt.title("Calculated finished square feet Vs Log error", fontsize=15)
plt.show()


# In[ ]:





# Here as well the distribution is very similar to the previous one. No wonder the correlation between the two variables are also high.
# Bathroom Count:

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="bathroomcnt", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# log error changes based on this.

# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()


# In[ ]:





# In[ ]:


#3.03 is the mean value with which we replaced the Null values.
train_df['bedroomcnt'].ix[train_df['bedroomcnt']>7] = 7
plt.figure(figsize=(12,8))
sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)
plt.xlabel('Bedroom count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# # Tax amount

# In[ ]:


#taxamount
col = "taxamount"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df['taxamount'].values, y=train_df['logerror'].values, size=10, color='g')
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Tax Amount', fontsize=12)
plt.title("Tax Amount Vs Log error", fontsize=15)
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x="taxamount", y="logerror", data=train_df)
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()


# In[ ]:





# In[ ]:





# # Year Built

# In[ ]:


# log error varies with the yearbuilt variable.
from ggplot import *
ggplot(aes(x='yearbuilt', y='logerror'), data=train_df) +     geom_point(color='steelblue', size=1) +     stat_smooth()


# In[ ]:





# In[ ]:





# # latitude and longitude.

# In[ ]:


#logerror varies with respect to latitude and longitude.
ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) +     geom_point() +     scale_color_gradient(low = 'red', high = 'blue')


# In[ ]:





# In[ ]:





# In[ ]:





# We had an understanding of important variables from the univariate analysis. But this is on a stand alone basis and also we have linearity assumption. Now let us build a non-linear model to get the important variables by building Extra Trees model.

# In[ ]:


cols = ["bathroomcnt","bedroomcnt","roomcnt","numberofstories","logerror","calculatedfinishedsquarefeet"]
mergedFiltered = train_df[cols].dropna()
for col in cols:
    ulimit = np.percentile(mergedFiltered[col].values, 99.5)
    llimit = np.percentile(mergedFiltered[col].values, 0.5)
    mergedFiltered[col].ix[mergedFiltered[col]>ulimit] = ulimit
    mergedFiltered[col].ix[mergedFiltered[col]<llimit] = llimit


# In[ ]:





# # Calculated Finished Square Feet Vs Log Error
# 

# In[ ]:





# In[ ]:




plt.figure(figsize=(8,8))
sns.jointplot(x=mergedFiltered.calculatedfinishedsquarefeet.values, y=mergedFiltered.logerror.values, size=10,kind="hex",color="#34495e")
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Calculated Finished Square Feet', fontsize=12)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # No Of Storeys Vs Log Error

# In[ ]:





# In[ ]:


#

fig,ax= plt.subplots()
fig.set_size_inches(20,5)
sns.boxplot(x="numberofstories", y="logerror", data=mergedFiltered,ax=ax,color="#36495e")
ax.set(ylabel='Log Error',xlabel="No Of Storeys",title="No Of Storeys Vs Log Error")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Bedroom Vs Bathroom Vs Log Error

# In[ ]:





# In[ ]:




from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
fig = pylab.figure()
fig.set_size_inches(20,10)
ax = Axes3D(fig)

ax.scatter(mergedFiltered.bathroomcnt, mergedFiltered.bedroomcnt, mergedFiltered.logerror,color="#34495e")
ax.set_xlabel('Bathroom Count')
ax.set_ylabel('Bedroom Count')
ax.set_zlabel('Log Error');
pyplot.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # roomcount  vs log error

# In[ ]:





# In[ ]:


fig,ax= plt.subplots()
fig.set_size_inches(20,5)
sns.boxplot(x="roomcnt", y="logerror", data=mergedFiltered,ax=ax,color="#24495e")
ax.set(ylabel='Log Error',xlabel="Room Count",title="Room Count Vs Log Error")


# In[ ]:





# In[ ]:





# In[ ]:





# # Bedroom count vs log error

# In[ ]:


fig,ax= plt.subplots()
fig.set_size_inches(20,5)
sns.boxplot(x="bedroomcnt", y="logerror", data=mergedFiltered,ax=ax,color="#34495e")
ax.set(ylabel='Log Error',xlabel="Bedroom Count",title="Bedroom Count Vs Log Error")


# In[ ]:





# In[ ]:





# # No Of Storey Over The Years

# In[ ]:





# In[ ]:


fig,ax1= plt.subplots()
fig.set_size_inches(20,10)
train_df["yearbuilt"] = train_df["yearbuilt"].map(lambda x:str(x).split(".")[0])
yearMerged = train_df.groupby(['yearbuilt', 'numberofstories'])["parcelid"].count().unstack('numberofstories').fillna(0)
yearMerged.plot(kind='bar', stacked=True,ax=ax1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




