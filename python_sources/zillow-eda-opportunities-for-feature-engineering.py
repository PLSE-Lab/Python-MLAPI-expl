#!/usr/bin/env python
# coding: utf-8

# This EDA is mostly focused on searching for opportunties for feature engineering. Some pieces of code were copied and/or inspired by the other great EDAs:
# 
#  - https://www.kaggle.com/philippsp/exploratory-analysis-zillow
#  - https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize
#  - https://www.kaggle.com/captcalculator/a-very-extensive-zillow-exploratory-analysis
#  - https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655
# 
# Key Takeaways:
# 
# - There appears to be a slight seasonal time dependence that may, or may not, be useful for predicting time-dependent error
# - Some of the features exhibit heteroskedasticy (error variance is larger at lower feature values) that may be useful for feature engineering
# -  There are sometimes significant differences in feature value distribution between the train and test sets
# -  Some of the other public kernels are treating non-ordinal data as if it was ordinal, which will cause some problems during fitting

# In[ ]:


import pandas            as pd;
import numpy             as np;
import seaborn           as sns;
import matplotlib.pyplot as plt;

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


names = ['parcelid', 'air_conditioning_type', 'architectural_style', 
         'area_basement', 'num_bathroom', 'num_bedroom', 'framing_type',
         'building_quality', 'num_bathroom_calc', 'deck_type',
         'area_firstfloor_finished', 'area_total_calc',
         'area_living_finished', 'perimeter_living',
         'area_total', 'area_firstfloor_unfinished',
         'area_base', 'fips', 'num_fireplace', 'num_fullbath',
         'num_garagecar', 'area_garage', 'hashottuborspa',
         'heating_type', 'latitude', 'longitude',
         'area_lot', 'num_pool', 'area_pools', 'pooltypeid10',
         'pooltypeid2', 'pooltypeid7', 'property_land_use_code',
         'property_land_use_type', 'property_zoning_desc',
         'census_raw_tract_block', 'region_city', 'region_county',
         'region_neighborhood', 'region_zipcode', 'num_room', 'story_type',
         'num_34_bath', 'material_type', 'num_unit',
         'area_patio', 'area_shed', 'build_year',
         'num_stories', 'flag_fireplace', 'tax_assessed_structure_value',
         'tax_assessed_parcel_value', 'tax_assessment_year', 'tax_assessed_land_alue',
         'tax_property', 'tax_delinquency_flag', 'tax_delinquency_year',
         'census_tract_block']

train_df   = pd.read_csv('../input/train_2016.csv', parse_dates=["transactiondate"]);
prop_df    = pd.read_csv('../input/properties_2016.csv', names=names, header=0);
sample_df  = pd.read_csv('../input/sample_submission.csv')

# Convert property float.64 data to float.32 to save memory
for c, dtype in zip(prop_df.columns, prop_df.dtypes):
	if dtype == np.float64:
		prop_df[c] = prop_df[c].astype(np.float32)


# In[ ]:


Inspect what the datasets look like. The training set is only about 3% the size of the test set!


# In[ ]:


#print(prop_df.values.shape) # looking at large shapes seems to break Kaggle Kernel's memory constraints
train_df.head()


# In[ ]:


#print(prop_df.values.shape)
prop_df.head()


# In[ ]:


#print(sample_df.values.shape)
sample_df.head()


# In[ ]:


train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()


# In[ ]:


sample_df['parcelid'] = sample_df['ParcelId'];
sample_df = sample_df.drop(['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712'], axis=1)
sample_df = pd.merge(sample_df, prop_df, on='parcelid', how='left')
#print(sample_df.values.shape)
sample_df.head()


# Create an abs(logerror) column to inspect later.

# In[ ]:


train_df['abs_logerror'] = abs(train_df['logerror'].values)


# ## Time Dependence ##

# Are there seasonal patterns to the number of transactions?

# In[ ]:


train_df['transaction_month'] = train_df['transactiondate'].dt.month

cnt_srs = train_df['transaction_month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[ ]:


train_df['transaction_day'] = train_df['transactiondate'].dt.day

cnt_srs = train_df['transaction_day'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='blue')
plt.xticks(rotation='vertical')
plt.xlabel('Day of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# Check how many parcel values occur 1x, 2x, 3x times

# In[ ]:


train_df['parcelid'].value_counts().reset_index()['parcelid'].value_counts()


# How many missing values are there in the training data? Some features have almost all their entries missing. May want to remove these features.

# In[ ]:


missing_df = prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# ## Inspect Property Descriptions ##

# What features have the strongest correlation with logerror and abs(logerror)?

# In[ ]:


print('Correlation with Log Error')
print(train_df.corr(method='pearson').drop(['logerror', 'abs_logerror']).sort_values('logerror', ascending=False)['logerror'].head(10))
print('\n')
print('Correlation with Abs Log Error')
print(train_df.corr(method='pearson').drop(['logerror', 'abs_logerror']).sort_values('abs_logerror', ascending=False)['abs_logerror'].head(10))


# Plot partial depedence plots of the most strongly-correlated variables

# In[ ]:


fig  = plt.figure(figsize=(9, 9), dpi=100);

#fig.suptitle('House Characteristics')

axes1 = fig.add_subplot(331); axes2 = fig.add_subplot(332); axes3 = fig.add_subplot(333); 
axes4 = fig.add_subplot(334); axes5 = fig.add_subplot(335); axes6 = fig.add_subplot(336); 
axes7 = fig.add_subplot(337); axes8 = fig.add_subplot(338); axes9 = fig.add_subplot(339); 

sns.regplot(x='area_basement',        y='logerror', data=train_df, ax=axes1, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_shed',            y='logerror', data=train_df, ax=axes2, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_living_finished', y='logerror', data=train_df, ax=axes3, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_total_calc',      y='logerror', data=train_df, ax=axes4, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_base',            y='logerror', data=train_df, ax=axes5, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='perimeter_living',     y='logerror', data=train_df, ax=axes6, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_34_bath',          y='logerror', data=train_df, ax=axes7, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_fireplace',        y='logerror', data=train_df, ax=axes8, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_bathroom_calc',    y='logerror', data=train_df, ax=axes9, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});

axes9.set_xlim(0, 30)

fig.tight_layout();


# In[ ]:


fig  = plt.figure(figsize=(9, 9), dpi=100);

#fig.suptitle('House Characteristics')

axes1 = fig.add_subplot(331); axes2 = fig.add_subplot(332); axes3 = fig.add_subplot(333); 
axes4 = fig.add_subplot(334); axes5 = fig.add_subplot(335); axes6 = fig.add_subplot(336); 
axes7 = fig.add_subplot(337); axes8 = fig.add_subplot(338); axes9 = fig.add_subplot(339); 

sns.regplot(x='area_basement',        y='abs_logerror', data=train_df, ax=axes1, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_shed',            y='abs_logerror', data=train_df, ax=axes2, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_living_finished', y='abs_logerror', data=train_df, ax=axes3, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_total_calc',      y='abs_logerror', data=train_df, ax=axes4, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_base',            y='abs_logerror', data=train_df, ax=axes5, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='perimeter_living',     y='abs_logerror', data=train_df, ax=axes6, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_34_bath',          y='abs_logerror', data=train_df, ax=axes7, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_fireplace',        y='abs_logerror', data=train_df, ax=axes8, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_bathroom_calc',    y='abs_logerror', data=train_df, ax=axes9, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});

axes9.set_xlim(0, 30)

fig.tight_layout();


# Many of these strongly-correlated features have a lot of missing values, and may not be very reliable. Now look at which variables have the strongest correlations and have at least half their data points filled in. 

# In[ ]:


simple_train_df = train_df.copy()
simple_train_df = simple_train_df.drop(missing_df[missing_df['missing_count'] > 1500000]['column_name'].values, axis=1)


# In[ ]:


print('Correlation with Log Error')
print(simple_train_df.corr(method='pearson').drop(['logerror', 'abs_logerror']).sort_values('logerror', ascending=False)['logerror'].head(10))
print('\n')
print('Correlation with Abs Log Error')
print(simple_train_df.corr(method='pearson').drop(['logerror', 'abs_logerror']).sort_values('abs_logerror', ascending=False)['abs_logerror'].head(10))


# In[ ]:


fig  = plt.figure(figsize=(9, 9), dpi=100);

#fig.suptitle('House Characteristics')

axes1 = fig.add_subplot(331); axes2 = fig.add_subplot(332); axes3 = fig.add_subplot(333); 
axes4 = fig.add_subplot(334); axes5 = fig.add_subplot(335); axes6 = fig.add_subplot(336); 
axes7 = fig.add_subplot(337); axes8 = fig.add_subplot(338); axes9 = fig.add_subplot(339); 

sns.regplot(x='area_living_finished',         y='abs_logerror', data=train_df, ax=axes1, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='area_total_calc',              y='abs_logerror', data=train_df, ax=axes2, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_bathroom_calc',            y='abs_logerror', data=train_df, ax=axes3, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_fullbath',                 y='abs_logerror', data=train_df, ax=axes4, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_bathroom',                 y='abs_logerror', data=train_df, ax=axes5, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_bedroom',                  y='abs_logerror', data=train_df, ax=axes6, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='tax_assessed_structure_value', y='abs_logerror', data=train_df, ax=axes7, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='build_year',                   y='abs_logerror', data=train_df, ax=axes8, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='tax_assessed_parcel_value',    y='abs_logerror', data=train_df, ax=axes9, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});

fig.tight_layout();


# In[ ]:


fig  = plt.figure(figsize=(9, 9), dpi=100);

#fig.suptitle('House Characteristics')

axes1 = fig.add_subplot(331); axes2 = fig.add_subplot(332); axes3 = fig.add_subplot(333); 
axes4 = fig.add_subplot(334); axes5 = fig.add_subplot(335); axes6 = fig.add_subplot(336); 

sns.regplot(x='region_county',          y='abs_logerror', data=train_df, ax=axes1, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='building_quality',       y='abs_logerror', data=train_df, ax=axes2, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='tax_property',           y='abs_logerror', data=train_df, ax=axes3, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='num_unit',               y='abs_logerror', data=train_df, ax=axes4, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='heating_type',           y='abs_logerror', data=train_df, ax=axes5, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='tax_assessed_land_alue', y='abs_logerror', data=train_df, ax=axes6, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});

axes4.set_xlim(0, 20)

fig.tight_layout();


# Notice that many of these data points are very heteroskedastic. That is, the mean log error at low values is much larger than it is at large values. We may want to take advantage of this for feature engineering.

# ## Inspect Time Data ##

# How does logerror and abs(logerror) vary with time?

# In[ ]:


fig  = plt.figure(figsize=(9, 6), dpi=100);

axes1 = fig.add_subplot(221); axes2 = fig.add_subplot(222);
axes3 = fig.add_subplot(223); axes4 = fig.add_subplot(224);

sns.regplot(x='transaction_month', y='logerror',     data=train_df, ax=axes1, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='transaction_day',   y='logerror',     data=train_df, ax=axes2, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='transaction_month', y='abs_logerror', data=train_df, ax=axes3, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});
sns.regplot(x='transaction_day',   y='abs_logerror', data=train_df, ax=axes4, scatter_kws={"s": 10, "color":"orange"}, line_kws={"color":"black"});

fig.tight_layout();


# ## Inspect Categorical Data ##

# Inspect (a lot) of categorical data.
# 
# TLDR:
# 
# - Many of the columns consist of features that naively look ordinal (e.g. integers 1-30) but aren't, and should be hot-encoded into a more suitable form.
# - There are sometimes substantial differences between train and test distributions.

# In[ ]:


feature = 'air_conditioning_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'architectural_style'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'framing_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'building_quality'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'deck_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'fips'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_fireplace'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_fullbath'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_garagecar'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'hashottuborspa'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'heating_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_pool'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'pooltypeid10'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'pooltypeid2'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'pooltypeid7'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


print(len(train_df['region_city'].unique()))

feature = 'region_city'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


print(len(train_df['region_county'].unique()))

feature = 'region_county'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


print(len(train_df['region_neighborhood'].unique()))

feature = 'region_neighborhood'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


print(len(train_df['region_zipcode'].unique()))

feature = 'region_zipcode'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_room'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'story_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_34_bath'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'material_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_unit'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'num_stories'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'flag_fireplace'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'tax_assessment_year'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'tax_delinquency_flag'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'tax_delinquency_year'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'property_land_use_code'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'property_land_use_type'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# In[ ]:


feature = 'property_zoning_desc'

fig  = plt.figure(figsize=(12, 9), dpi=100);
axes1 = fig.add_subplot(411); axes2 = fig.add_subplot(412); axes3 = fig.add_subplot(413); axes4 = fig.add_subplot(414); 
sns.countplot(train_df[feature].fillna('NaN'), ax=axes1); sns.countplot(sample_df[feature].fillna('NaN'),  ax=axes2);
bar_df1 = train_df[[feature,     "logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
bar_df2 = train_df[[feature, "abs_logerror"]].fillna('NaN').groupby([feature], as_index=False).mean()
sns.barplot(x=feature, y='logerror',     data=bar_df1, ax=axes3);
sns.barplot(x=feature, y='abs_logerror', data=bar_df2, ax=axes4);
fig.tight_layout()


# To be continued...

# In[ ]:




