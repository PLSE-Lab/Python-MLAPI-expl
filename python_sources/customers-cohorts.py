#!/usr/bin/env python
# coding: utf-8

# # Summary
# Different techniques for customer segmentation:
# 
# - cohort analysis
# - Recency, Frequency, Monetary Value analysis
# - Clustering with kmeans
# - Recency, Frequency, Monetary Value, Tenure analysis
# 
# To interpret the obtained clusters we have used:
# - snake plots
# - relative importance of attributes
# 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[3]:


data = pd.read_excel('../input/Online Retail.xlsx')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.loc[data.CustomerID.isnull()].head()


# In[9]:


data.duplicated().sum()


# In[10]:


data = data.drop_duplicates()


# In[11]:


data = data.dropna(subset=['CustomerID'])


# In[12]:


data.isnull().sum().sum()


# In[13]:


data.shape


# In[14]:


data['Country'].value_counts().plot.bar(figsize=(18,5))
plt.show()


# There are some StockCode with different descriptions:

# In[15]:


(data.groupby('StockCode')['Description'].nunique() > 1).sum()


# In[16]:


data['StockCode'].nunique(), data['Description'].nunique()


# There are some negative values for 'UnitPrice' which may be returns. We will remove them for now:

# In[17]:


np.logical_or(data['Quantity'] <=0, (data['UnitPrice'] <=0)).value_counts(normalize=True)


# In[18]:


data = data.query('Quantity > 0 and UnitPrice > 0')


# We create the cohorts:

# In[19]:


data['InvoiceDay'] = data['InvoiceDate'].dt.date
data['InvoiceDay'] = pd.to_datetime(data['InvoiceDay'])


# In[20]:


grouping = data.groupby('CustomerID')['InvoiceDay']


# In[21]:


data['CohortDay'] = grouping.transform('min')


# In[22]:


data.head()


# In[23]:


data['CohortIndex'] = (data['InvoiceDay'] - data['CohortDay']).dt.days


# In[24]:


data.head()


# In[25]:


data.tail()


# In[26]:


data.info()


# In[27]:


data['CohortIndex'] = data['CohortIndex'].add(1)


# In[28]:


data.head()


# In[29]:


import datetime as dt
def get_month(x): return dt.datetime(x.year, x.month, 1)


# In[30]:


data['InvoiceMonth'] = data['InvoiceDate'].apply(get_month)


# In[31]:


data.head()


# In[32]:


grouping = data.groupby('CustomerID')['InvoiceMonth']


# In[33]:


data['CohortMonth'] = grouping.transform('min')


# In[34]:


data.tail()


# Let's use months for CohortIndex, instead of days:

# In[35]:


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day


# In[36]:


invoice_year, invoice_month, _ = get_date_int(data, 'InvoiceMonth')
cohort_year, cohort_month, _ = get_date_int(data, 'CohortMonth')


# In[37]:


years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month


# In[38]:


data['CohortIndex'] = years_diff * 12 + months_diff + 1


# In[39]:


data.tail()


# In[40]:


grouping = data.groupby(['CohortMonth','CohortIndex'])


# In[41]:


grouping['CustomerID'].nunique().head(10)


# In[42]:


cohort_data_1 = grouping['CustomerID'].apply(pd.Series.nunique)
cohort_data = grouping['CustomerID'].nunique()


# In[43]:


(cohort_data_1!=cohort_data).sum()


# In[44]:


cohort_data = cohort_data.reset_index()


# In[45]:


cohort_data.head()


# In[46]:


cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')


# In[47]:


cohort_counts


# In[48]:


cohort_sizes = cohort_counts.iloc[:,0]


# In[49]:


retention = cohort_counts.div(cohort_sizes, axis=0)


# Customer retention rate across time per cohort:

# In[50]:


retention.round(3) * 100


# In[51]:


import seaborn as sns
retention.index = retention.index.date

plt.figure(figsize=(8, 6))
plt.title('Retention by Monthly Cohorts')
sns.heatmap(retention, annot=True, cmap='BuGn')
plt.show()


# Let's check the average price across time and across cohorts.

# In[52]:


grouping = data.groupby(['CohortMonth', 'CohortIndex'])


# In[53]:


cohort_data = grouping['UnitPrice'].mean()


# In[54]:


cohort_data = cohort_data.reset_index()


# In[55]:


cohort_data.head()


# In[56]:


average_quantity = cohort_data.pivot(index='CohortMonth', 
                                     columns='CohortIndex', 
                                     values='UnitPrice').round(1)

average_quantity


# In[57]:


average_quantity.index = average_quantity.index.date

plt.figure(figsize=(8, 6))
plt.title('Average Spend by Monthly Cohorts')
sns.heatmap(average_quantity, annot=True, cmap='Blues')
plt.show()


# Let's check the number of different products bought  across time and across cohorts.

# In[58]:


cohort_data = grouping['StockCode'].nunique().reset_index()


# In[59]:


cohort_data.head()


# In[60]:


different_products = cohort_data.pivot(index='CohortMonth', 
                                       columns='CohortIndex', 
                                       values='StockCode').round(1)
different_products.index = different_products.index.date

different_products


# In[61]:



plt.figure(figsize=(12, 8))
sns.heatmap(different_products, annot=True, fmt=".0f", cmap='Reds', linewidths=0.1)
plt.show()


# ## Recency, Frequency, Monetary (RFM) segmentation
# 

# In[62]:


print("Min: {} | Max: {}".format(data['InvoiceDate'].dt.date.min(), data['InvoiceDate'].dt.date.max()))


# In[63]:


snapshot_date = data['InvoiceDate'].max() + dt.timedelta(days=1)
snapshot_date


# In[64]:


data['TotalSum'] = data['UnitPrice']*data['Quantity']
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


# In[65]:


data.head()


# In[66]:


datamart = data.groupby('CustomerID').    agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSum': 'sum'
    })


# In[67]:


datamart.head()


# In[68]:


datamart.rename(
    columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSum': 'MonetaryValue'}, inplace=True)


# In[69]:


datamart.head()


# In[70]:


# Create labels for Recency, Frequency and MonetaryValue
r_labels = range(3, 0, -1)
f_labels = range(1, 4)
m_labels = range(1, 4)


# In[71]:


r_groups = pd.qcut(datamart['Recency'], q=3, labels=r_labels)
f_groups = pd.qcut(datamart['Frequency'], q=3, labels=f_labels)
m_groups = pd.qcut(datamart['MonetaryValue'], q=3, labels=m_labels)


# In[72]:


datamart = datamart.assign(R=r_groups, F=f_groups, M=m_groups)


# In[73]:


datamart['RFM_Score'] = datamart[['R', 'F', 'M']].sum(axis=1)


# In[74]:


datamart['RFM_Segment'] = datamart.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis=1)


# In[75]:


datamart.head()


# Best 5 clients:

# In[76]:


datamart.sort_values('RFM_Score').head()


# Worst 5 clients:

# In[77]:


datamart.sort_values('RFM_Score', ascending=False).head()


# We use more understable labels to categorize the customers:

# In[78]:


def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Top'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 9)):
        return 'Middle'
    else:
        return 'Low'


# In[79]:


datamart['RFM_Level'] = datamart.apply(rfm_level, axis=1)


# In[80]:


datamart.head()


# In[81]:


rfm_level_agg = datamart.groupby('RFM_Level').agg({'Recency': 'mean', 
                                                   'Frequency':'mean', 
                                                   'MonetaryValue':['mean', 'count']}).round(1)


# In[82]:


rfm_level_agg


# ## K-means
# we are going to use kmeans to find a meaninfull number of groups. The features used need to fit some assumptions:
# 
# - all features must have the same mean
# - all features must have the same variance
# - no skewed features

# In[83]:


datamart[['Recency', 'Frequency', 'MonetaryValue']].describe()


# In[84]:


datamart.shape


# In[85]:


datamart[['Recency', 'Frequency', 'MonetaryValue']].    plot(kind='kde', figsize=(16, 10), subplots=True, sharex=False, layout=(3, 1))
plt.show()


# In[86]:


from scipy.stats import boxcox


# In[87]:


boxcox_recency, lam_recency = boxcox(datamart['Recency'])


# In[88]:


def plot_transformation(label, before_values, after_values):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(label)
    ax[0].set_xlabel('BEFORE')
    sns.distplot(before_values, ax=ax[0])
    ax[1].set_xlabel('AFTER')
    sns.distplot(after_values, ax=ax[1])
    plt.show()


# In[89]:


plot_transformation('Recency', datamart['Recency'].values, boxcox_recency)


# In[90]:


boxcox_frequency, lam_frequency = boxcox(datamart['Frequency'])
plot_transformation('Frequency', datamart['Frequency'].values, boxcox_frequency)


# In[91]:


boxcox_monetaryvalue, lam_monetaryvalue = boxcox(datamart['MonetaryValue'])
plot_transformation('MonetaryValue', datamart['MonetaryValue'].values, boxcox_monetaryvalue)


# In[92]:


datamart_rfm = datamart[['Recency', 'Frequency', 'MonetaryValue']]


# In[93]:


from sklearn.preprocessing import StandardScaler

datamart_unskewed = datamart_rfm.copy()
datamart_unskewed['Recency'] = boxcox_recency
datamart_unskewed['Frequency'] = boxcox_frequency
datamart_unskewed['MonetaryValue'] = boxcox_monetaryvalue


# In[94]:


scaler = StandardScaler()
datamart_normalized = scaler.fit_transform(datamart_unskewed)
datamart_normalized = pd.DataFrame(data=datamart_normalized, 
                                   index=datamart_rfm.index, 
                                   columns=datamart_rfm.columns)


# In[95]:


datamart_normalized.head()


# In[96]:


from sklearn.cluster import KMeans


# In[97]:


sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(datamart_normalized)
    sse[k] = kmeans.inertia_


# In[98]:


sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[99]:


k_choosen = 3


# In[100]:


kmeans = KMeans(n_clusters=k_choosen, random_state=1)
kmeans.fit(datamart_normalized)


# In[101]:


datamart_cluster = datamart_rfm.assign(Cluster=kmeans.labels_)


# In[102]:


datamart_cluster.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count'],
}).round(0)


# In[103]:


datamart_cluster.head()


# ### Snake plot

# In[104]:


datamart_melt = pd.melt(datamart_normalized.assign(Cluster=kmeans.labels_).reset_index(),
                        id_vars=['CustomerID', 'Cluster'],
                        value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                        var_name='Metric', value_name='Value'
                       )


# In[105]:


datamart_melt.head()


# In[106]:


plt.title('Snake plot of normalized variables')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.lineplot(data=datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()


# In[107]:


datamart_melt.describe().round(2)


# ### Relative importance of each attribute

# In[108]:


datamart_cluster.head()


# In[109]:


cluster_avg = datamart_cluster.groupby(['Cluster']).mean() 


# In[110]:


population_avg = datamart_rfm.mean()


# In[111]:


relative_imp = cluster_avg / population_avg - 1


# In[112]:


relative_imp.round(2)


# In[113]:


plt.title('Relative importance of attributes')
sns.heatmap(relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()


# ## Recency, Frequency, Monetary value, Tenure (RFMT) segmentation
# We add Tenure to our model (the number of days since a client did his first purchase)

# In[114]:


def get_recency(x):
    return (snapshot_date - x.max()).days

def get_tenure(x):
    return (snapshot_date - x.min()).days
    
datamart_rfmt = data.groupby('CustomerID').agg({'InvoiceDate': [get_recency, get_tenure],
                                           'InvoiceNo': 'count', 
                                           'TotalSum': 'sum'})


# In[115]:


datamart_rfmt.head()


# In[116]:


datamart_rfmt.columns = ['Recency', 'Tenure', 'Frequency', 'MonetaryValue']


# In[117]:


datamart_rfmt.head()


# In[118]:


boxcox_recency, lam_recency = boxcox(datamart_rfmt['Recency'])
boxcox_tenure, lam_tenure = boxcox(datamart_rfmt['Tenure'])
boxcox_frequency, lam_frequency = boxcox(datamart_rfmt['Frequency'])
boxcox_monetaryValue, lam_monetaryValue = boxcox(datamart_rfmt['MonetaryValue'])


# In[119]:


datamart_unskewed = datamart_rfmt.copy()
datamart_unskewed['Recency'] = boxcox_recency
datamart_unskewed['Tenure'] = boxcox_tenure
datamart_unskewed['Frequency'] = boxcox_frequency
datamart_unskewed['MonetaryValue'] = boxcox_monetaryvalue


# In[120]:


scaler = StandardScaler()
datamart_normalized = scaler.fit_transform(datamart_unskewed)
datamart_normalized = pd.DataFrame(data=datamart_normalized, 
                                   index=datamart_rfmt.index, 
                                   columns=datamart_rfmt.columns)


# In[121]:


sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(datamart_normalized)
    sse[k] = kmeans.inertia_


# In[122]:


sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[123]:


k_choosen = 3

kmeans = KMeans(n_clusters=k_choosen, random_state=1)
kmeans.fit(datamart_normalized)


# In[124]:


datamart_cluster = datamart_rfmt.assign(Cluster=kmeans.labels_)


# In[125]:


datamart_cluster.groupby('Cluster').agg({
    'Recency': 'mean',
    'Tenure': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count'],
}).round(0)


# In[126]:


datamart_cluster.groupby('Cluster').mean() 


# In[127]:


cluster_avg = datamart_cluster.groupby(['Cluster']).mean() 
population_avg = datamart_cluster[['Recency', 'Tenure', 'Frequency', 'MonetaryValue']].mean()
relative_imp = cluster_avg / population_avg - 1


# In[128]:


relative_imp


# In[129]:


sns.heatmap(relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()


# In[130]:


datamart_melt = pd.melt(
    datamart_normalized.assign(Cluster=kmeans.labels_).reset_index(), 
    id_vars=['CustomerID', 'Cluster'],
    value_vars=['Recency', 'Tenure', 'Frequency', 'MonetaryValue'], 
    var_name='Metric', value_name='Value'
)


# In[131]:


plt.title('Snake plot of normalized variables')
plt.xlabel('Metric')
plt.ylabel('Value')
sns.lineplot(data=datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()


# In[132]:


datamart_normalized.head()


# References:
# 
# - https://en.wikipedia.org/wiki/RFM_(customer_value)
# - https://www.datacamp.com/courses/customer-segmentation-in-python
# - https://towardsdatascience.com/find-your-best-customers-with-customer-segmentation-in-python-61d602f9eee6
# 
