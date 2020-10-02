#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style()  
pd.set_option('display.max_columns', 100)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


def read_files(train_fn, test_fn, store_fn):
    train = pd.read_csv(train_fn, parse_dates = ['Date'], infer_datetime_format = True)
    test = pd.read_csv(test_fn, parse_dates = ['Date'], infer_datetime_format = True)
    store = pd.read_csv(store_fn)
    
    return train, test, store


# In[ ]:


train, test, store = read_files('../input/train.csv', '../input/test.csv', '../input/store.csv')


# In[ ]:


train.head()


# **Check the distribution of Sales and Customers.**
# 
# Filter only when the store is open

# In[ ]:


train.query('Open == 1')[['Sales', 'Customers']].hist(bins=100, figsize=(10,4), xrot=45, sharey=True);


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)

open_df = train.query('Open == 1')[['Sales','DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday']]
sns.violinplot('DayOfWeek', 'Sales', data=open_df, ax=ax1)
sns.violinplot('Promo', 'Sales', data=open_df, ax=ax2)
sns.violinplot('StateHoliday', 'Sales', data=open_df, ax=ax3)
sns.violinplot('SchoolHoliday', 'Sales', data=open_df, ax=ax4)

fig.set_size_inches(15,6)
fig.tight_layout()
fig.show()


# In[ ]:


train.groupby('Date')['Store'].size().plot(kind='line');


# In[ ]:


**There are days in the training set where some stores are missing. Lets add those missing stores with Open flag as '0' and Customers and Sales values as '0'**


# In[ ]:


def add_missing_dates(train, all_stores):
    train_m = pd.DataFrame()
    store_by_date = train.groupby('Date')['Store'].nunique().reset_index()
    for i in store_by_date.query('Store != 1115')['Date']:
        diff_stores = all_stores.difference(set(train[train['Date']==i].Store))
        s = list(diff_stores)
        missing = pd.DataFrame(data={
                                 'Date': [i]*len(s), 
                                 'Store': s, 
                                 'Customers': [0]*len(s),
                                 'Sales': [0]*len(s),
                                 'Open': [0]*len(s),
                                 'Promo': [0]*len(s),
                                 'SchoolHoliday': [0]*len(s),
                                 'StateHoliday': ['0']*len(s)
                                }) 
    
        train_m = train_m.append(missing)
        train_m['DayOfWeek'] = train_m.Date.dt.dayofweek+1 
    
    return train_m[['Store','DayOfWeek','Date','Sales','Customers','Open','Promo','StateHoliday','SchoolHoliday']]


# In[ ]:


missing_train = add_missing_dates(train, set(store.Store))
train = pd.concat([train, missing_train])
train.groupby('Date')['Store'].size().plot(kind='line');


# In[ ]:


fig = plt.figure(figsize=(8,4))

rand_stores = np.random.randint(0,1115, 20)
for i in rand_stores:
    data = train[(train["Store"] == i)][['Date','Customers']]
    plt.plot_date(data.Date, data.Customers,'-', alpha=0.5)

fig.autofmt_xdate() 


# In[ ]:



sp = train[['Store','Date','Sales']]
sales_pivot = sp.pivot(index='Store', columns='Date', values='Sales')
col_names = ['col_%d' % i for i in range(len(sales_pivot.columns))]
sales_pivot.columns = col_names


cp = train[['Store','Date','Customers']]
customers_pivot = cp.pivot(index='Store', columns='Date', values='Customers')
customers_pivot.columns = col_names
customers_pivot.head()


# In[ ]:


from sklearn import cluster

cl = cluster.hierarchical.AgglomerativeClustering(n_clusters=5)
clusters = cl.fit_predict(X=customers_pivot)

cl_df = pd.DataFrame({'Store' : customers_pivot.index, 'cluster': clusters}).set_index('Store')
chart_df = train.set_index('Store').join(cl_df)[['Date', 'Customers', 'cluster']]
chart_df.cluster.value_counts().sort_index()


# In[ ]:



def plot_clusters(df, clu, xcol='Date', ycol='Customers',cluster_label='cluster'):
    fig, axes = plt.subplots(1, cl.n_clusters, sharex=True, sharey=True)
    
    for ax, l in zip(axes, np.unique(cl.labels_)):
        tdf = df.where(df[cluster_label] == l).dropna()
        
        for i in tdf.index.unique():
            data = tdf.loc[i].set_index(xcol).resample('m').agg({ycol: 'sum', cluster_label:'max'}).reset_index()
            
            ax.plot_date(x=data[xcol], 
                         y=data[ycol], 
                         linestyle='solid', 
                         xdate=True, 
                         ydate=False, 
                         alpha=0.6)

    
    fig.set_size_inches(10,4)
    fig.tight_layout()
    fig.autofmt_xdate()


# In[ ]:


plot_clusters(chart_df, clu=cl)


# **There is a clear split of clusters depending on the trend of the Customers feature.
# TODO: do some more analysis on each cluster**

# In[ ]:


test.head()


# In[ ]:


print ("Train features:")
print (train.isnull().any())
print ("\n")
print ("Test features:")
print (test.isnull().any())


# In[ ]:


print ("Train features:")
print (train.StateHoliday.value_counts())
print ("\n")
print (train.SchoolHoliday.value_counts())
print ("\n")
print (train.Promo.value_counts())
print ("\n")
print ("Test features:")
print (test.StateHoliday.value_counts())
print ("\n")
print (test.SchoolHoliday.value_counts())
print ("\n")
print (test.Promo.value_counts())


# In[ ]:


test[test.Open.isnull()].DayOfWeek.value_counts().sort_index()


# Open is NULL on all weekdays except Sunday and on dates which are not holiday, so i assume the store is open. Also i will split the date feature in [year, month, day, weekofyear] features. There are two ['0'] in train.StateHoliday so merge that as well

# In[ ]:


def split_date(df, date_col):
    n_date_year = df[date_col].dt.year
    n_date_month = df[date_col].dt.month
    n_date_weeknum = df[date_col].dt.weekofyear
    n_date_day = df[date_col].dt.day
    
    return df.assign(date_year=n_date_year, date_month=n_date_month, date_weeknum=n_date_weeknum, date_day=n_date_day)
 
train = split_date(train, 'Date')
test = split_date(test, 'Date')

train.StateHoliday = train.StateHoliday.map({'0':'0', 'a':'1', 'b': '2', 'c':'3'})
test.StateHoliday  = test.StateHoliday.map({'0':'0', 'a':'1', 'b': '2', 'c':'3'})
test.Open = test.Open.fillna(1)


# In[ ]:


store.head()


# In[ ]:


store.info()


# In[ ]:


def myPinterval(x):
    if x=='Feb,May,Aug,Nov':  return([0,1,0,0,1,0,0,1,0,0,1,0])
    elif x=='Jan,Apr,Jul,Oct':  return([1,0,0,1,0,0,1,0,0,1,0,0])
    elif x== 'Mar,Jun,Sept,Dec': return([0,0,1,0,0,1,0,0,1,0,0,1])
    else: return(np.repeat(0,12).tolist())

#Convert the Promointerval from a string column to a set of columns with flag [0/1]
proInt = store.PromoInterval.apply(myPinterval).tolist()
proInt = pd.DataFrame(proInt, columns = ['ProInt'+ str(i) for i in range(1,13)] , dtype=np.int8)
store = store.drop('PromoInterval',1).join(proInt)

#Fill NA with the median CompetitionDistance, TODO: look for a better solution
distmean = store.CompetitionDistance.median()
store.CompetitionDistance = store.CompetitionDistance.fillna(distmean) 

##Convert CompetitionOpenSince to a date field and set NA = 1970/01/01
store['CompetitionOpenSinceDay'] = 1
store['CompetitionOpenSinceDT'] = pd.to_datetime(dict(year=store.CompetitionOpenSinceYear, month=store.CompetitionOpenSinceMonth, day=store.CompetitionOpenSinceDay))
store = store.drop(['CompetitionOpenSinceYear','CompetitionOpenSinceMonth','CompetitionOpenSinceDay'], axis='columns')
ifnulldt = pd.to_datetime('1970-01-01')
store.CompetitionOpenSinceDT = store.CompetitionOpenSinceDT.fillna(ifnulldt) 

##Convert PromoSince to a date field and set NA = 1970/01/01
store['Promo2Mon'] = 1
store['Promo2Day'] = 1
store['Promo2SinceDT'] = pd.to_datetime(dict(year=store.Promo2SinceYear, month=store.Promo2Mon, day=store.Promo2Day))
store = store.drop(['Promo2SinceYear','Promo2Mon','Promo2Day'], axis='columns') 
mask = store.Promo2SinceWeek.isnull() == False
store.loc[mask, 'Promo2SinceWeek'] = store[mask].Promo2SinceWeek.apply(lambda x: np.timedelta64(np.int(x), 'W'))
store['Promo2SinceDT'] = store['Promo2SinceDT'] + store['Promo2SinceWeek']
store = store.drop(['Promo2SinceWeek'], axis='columns')
store.Promo2SinceDT = store.Promo2SinceDT.fillna(ifnulldt)


# In[ ]:


store.info()


# In[ ]:


train_df = train.set_index('Store').join(store.set_index('Store'), how='inner').reset_index()
test_df = test.set_index('Store').join(store.set_index('Store'), how='inner').reset_index()

train_df.DayOfWeek = train_df.DayOfWeek.astype(str)
dummies = pd.get_dummies(train_df[['Assortment', 'StoreType', 'StateHoliday','DayOfWeek']])
train_df = train_df.join(dummies)
train_df = train_df.drop(['Assortment', 'StoreType', 'StateHoliday','DayOfWeek'], axis=1)

test_df.DayOfWeek = test_df.DayOfWeek.astype(str)
dummies = pd.get_dummies(test_df[['Assortment', 'StoreType', 'StateHoliday','DayOfWeek']])
test_df = test_df.join(dummies)
test_df = test_df.drop(['Assortment', 'StoreType', 'StateHoliday','DayOfWeek'], axis=1)


# In[ ]:


train_df = train_df.assign(days_since_comp = train_df['Date'] - train_df['CompetitionOpenSinceDT'])
train_df = train_df.assign(days_since_promo = train_df['Date'] - train_df['Promo2SinceDT'])

test_df = test_df.assign(days_since_comp = test_df['Date'] - test_df['CompetitionOpenSinceDT'])
test_df = test_df.assign(days_since_promo = test_df['Date'] - test_df['Promo2SinceDT'])

train_df.days_since_comp = (train_df.days_since_comp / np.timedelta64(1, 'D')).astype(int)
train_df.days_since_promo = (train_df.days_since_promo / np.timedelta64(1, 'D')).astype(int) 

test_df.days_since_comp = (test_df.days_since_comp / np.timedelta64(1, 'D')).astype(int)
test_df.days_since_promo = (test_df.days_since_promo / np.timedelta64(1, 'D')).astype(int) 

train_df.loc[train_df.CompetitionOpenSinceDT.dt.year <= 1970,'days_since_comp']
train_df.loc[train_df.days_since_comp < 0,'days_since_comp'] = 0
train_df.loc[train_df.Promo2SinceDT.dt.year <= 1970,'days_since_promo'] = 0
train_df.loc[train_df.days_since_promo < 0, 'days_since_promo'] = 0

test_df.loc[test_df.CompetitionOpenSinceDT.dt.year <= 1970,'days_since_comp']
test_df.loc[test_df.days_since_comp < 0,'days_since_comp'] = 0
test_df.loc[test_df.Promo2SinceDT.dt.year <= 1970,'days_since_promo'] = 0
test_df.loc[test_df.days_since_promo < 0, 'days_since_promo'] = 0

train_df.drop(['CompetitionOpenSinceDT','Promo2SinceDT'], axis=1, inplace=True)
test_df.drop(['CompetitionOpenSinceDT','Promo2SinceDT'], axis=1, inplace=True)


# In[ ]:


def remove_promo_interval_flag(df):
    df = df.assign(is_promo = ((df['date_month'] == 1)  & (df['ProInt1'] == 1))  |
                              ((df['date_month'] == 2)  & (df['ProInt2'] == 1))  | 
                              ((df['date_month'] == 3)  & (df['ProInt3'] == 1))  | 
                              ((df['date_month'] == 4)  & (df['ProInt4'] == 1))  | 
                              ((df['date_month'] == 5)  & (df['ProInt5'] == 1))  | 
                              ((df['date_month'] == 6)  & (df['ProInt6'] == 1))  | 
                              ((df['date_month'] == 7)  & (df['ProInt7'] == 1))  | 
                              ((df['date_month'] == 8)  & (df['ProInt8'] == 1))  | 
                              ((df['date_month'] == 9)  & (df['ProInt9'] == 1))  | 
                              ((df['date_month'] == 10) & (df['ProInt10'] == 1)) | 
                              ((df['date_month'] == 11) & (df['ProInt11'] == 1)) |
                              ((df['date_month'] == 12) & (df['ProInt12'] == 1)))
    
    df.is_promo = df.is_promo.astype(np.int8)
    df = df.drop(['ProInt1' ,'ProInt2' ,'ProInt3',
                   'ProInt4' ,'ProInt5' ,'ProInt6',
                   'ProInt7' ,'ProInt8' ,'ProInt9',
                   'ProInt10','ProInt11','ProInt12'], axis=1)
    
    return df

train_df = remove_promo_interval_flag(train_df)
test_df  = remove_promo_interval_flag(test_df)

train_df.head()


# In[ ]:


train_df = train_df.set_index(['Store', 'Date']).sort_index()
test_df = test_df.set_index(['Store', 'Date']).sort_index()


# In[ ]:


def create_model(store_list, clf):
    customer_score = []
    customer_pred = []
    customer_fimp = []
    sales_score = []
    sales_pred = []
    sales_fimp = []
    clf_dict = {}

    for i in store_list:
        clf_list = []
        print("Fitting store: ", i)
        X = train_df.loc[i]
        y1 = X.pop('Customers')
        y2 = X.pop('Sales')

        clf.fit(X[:-100], y1[:-100])
        y1_pred = clf.predict(X[-100:])
        y1_score = clf.score(X, y1)
        print("----Customer pred score: ", y1_score)
        
        customer_score.append(y1_score)
        customer_pred.append(y1_pred)
        
        if hasattr(clf, 'feature_importances_'):
            customer_fimp.append(clf.feature_importances_)
        
        clf_list.append(clf)
        
        clf.fit(X[:-100], y2[:-100])
        y2_pred = clf.predict(X[-100:])
        y2_score = clf.score(X, y2)
        print("----Sales pred score: ", y2_score)

        sales_score.append(y2_score)
        sales_pred.append(y2_pred)
        if hasattr(clf, 'feature_importances_'):
            sales_fimp.append(clf.feature_importances_)
        
        clf_list.append(clf)
        clf_dict[i] = clf_list
        
    return customer_score, customer_pred, customer_fimp, sales_score, sales_pred, sales_fimp, clf_dict


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=200)

store_ix = store.Store.tolist()[0:500]

y1 = train_df.loc[store_ix, 'Customers']
y2 = train_df.loc[store_ix, 'Sales']
columns = train_df.drop(['Customers', 'Sales'], axis=1).columns

y1_score, y1_pred, y1_fimp, y2_score, y2_pred, y2_fimp, clf_dict = create_model(store_ix[0:600], clf)

#model_result = pd.DataFrame({'Y1_true':y1[-100:].values.tolist(), 'y1_pred': np.array(y1_pred).flatten()}, index=y1[-100:].index.get_level_values(1))


# In[ ]:


plt.figure(figsize=(8,3))
plt.plot(y1_score)
plt.plot(y2_score)
plt.ylim(0.5,1)
plt.show()


# In[ ]:


model_fimp = pd.DataFrame(np.array(y1_fimp), index=store_ix[0:600], columns=columns)
model_fimp.mean().sort_values().plot(kind='barh', title='Feature Importance', figsize=(6,6));


# In[ ]:





# In[ ]:




