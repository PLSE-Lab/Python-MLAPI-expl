#!/usr/bin/env python
# coding: utf-8

# # 120 Years of Olympic History

# ### Importing modules

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Loading data

# In[ ]:


data = pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')
regions = pd.read_csv('/kaggle/input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
hosts = pd.read_csv('/kaggle/input/olympic-host-cities/olym.csv',header=0,encoding = 'unicode_escape')
gdp = pd.read_csv('/kaggle/input/worldwide-gdp-history-19602016/gdp_data.csv')


# ### Looking at the data format

# In[ ]:


data.head()


# In[ ]:


hosts.head(20)


# ### Grabbing summer stats and dummy coding Medal

# In[ ]:


#winter = data.loc[data['Season'] == 'Winter']
df = data.loc[data['Season'] == 'Summer']
df = df.drop(['Season'], axis=1)

df['Medal'] = df['Medal'].replace(['Bronze','Silver','Gold'],1)
df['Medal'] = df['Medal'].fillna(0)
df.head()


# In[ ]:


hosts = hosts.loc[hosts['Winter'].isnull()]
hosts.head(20)


# ### Getting medal counts by country

# In[ ]:


medals = df.groupby(['NOC','Year'])[['Medal']].agg('sum').reset_index()

usa = medals.loc[medals['NOC'] == 'USA'].drop('NOC',axis=1)
gbr = medals.loc[medals['NOC'] == 'GBR'].drop('NOC',axis=1)
chn = medals.loc[medals['NOC'] == 'CHN'].drop('NOC',axis=1)
medals2 = medals.merge(hosts, on='Year', suffixes=('', '_host')).drop(['City','Country','Summer','Winter','Latitude','Longitude'], axis=1)
medals2 = medals2.merge(usa, 'inner', on='Year', suffixes=('','_USA')).merge(gbr, 'inner', on='Year', suffixes=('','_GBR')).merge(chn, 'inner', on='Year', suffixes=('','_CHN'))
medals2 = medals2.drop(['NOC','Medal'], axis=1).rename(columns={'NOC_host': 'Host'})
medals2


# ### Looking at athletes sent vs. medals won

# In[ ]:


medals = df.groupby(['NOC','Year'])[['Medal']].agg('sum').reset_index()

athletes = df.groupby(['NOC','Year'])[['Name']].agg('count').reset_index().rename(columns={'Name': 'Athletes'})
athletes = athletes.merge(medals, on=['NOC','Year'])
athletes


# In[ ]:


np.corrcoef(athletes['Athletes'],athletes['Medal'])


# #### Looking at the correlational coefficient, we can conclude that there is a strong correlation between sending more athletes and winning more medals.

# ### Adding GDP Data as a potential predictor

# In[ ]:


gdp['Code'].replace('DEU','GER')


# In[ ]:


df = athletes.merge(gdp, left_on = ['NOC','Year'], right_on = ['Code','Year']).drop(['Country','Code'], axis = 1).dropna()
df = df.merge(hosts, on='Year', suffixes=('', '_host')).drop(['City','Country','Summer','Winter','Latitude','Longitude'], axis=1).rename(columns={'NOC_host': 'Host'})
df


# In[ ]:


dfHost = athletes.merge(hosts, on='Year', suffixes=('', '_host')).drop(['City','Country','Summer','Winter','Latitude','Longitude'], axis=1).rename(columns={'NOC_host': 'Host'})
dfHost


# ### Add variable to track if country is hosting

# In[ ]:


def is_host(row):
    if row['NOC'] == row['Host'] :
        return 1
    return 0

dfHost['is_host'] = dfHost.apply(lambda row: is_host(row), axis=1)
df['is_host'] = df.apply(lambda row: is_host(row), axis=1)
df


# ### Generate regression coefficients

# In[ ]:


def lin_reg(X,y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)

    return model

def mod_perf(model, X, y):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true=y, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y, y_pred=y_pred)
    print('MLR MSE: {:0.2f}\n'.format(mse))
    print('MLR MAE: {:0.2f}\n'.format(mae))
    print('Variance score: %.2f' % r2_score(y, y_pred))
    return


# In[ ]:


dfNum = df.drop(['NOC','Host'],axis=1)
X = dfNum.drop(['Medal'],axis=1)
y = dfNum['Medal']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

med_all = lin_reg(X_train, y_train)

pd.Series(med_all.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# #### Countries can expect to win an extra medal for every ~8 athletes they send to the olympics
# #### GDP and GDP per Capita have no effect on medal count
# #### Hosting the olympics increases your expected medal count

# In[ ]:


print('Training Performance:\n')
mod_perf(med_all, X_train, y_train)


# In[ ]:


print('Testing Performance:\n')
mod_perf(med_all, X_test, y_test)


# In[ ]:


med_int = med_all.intercept_.round(3)

def pred_medals(ath,year,gdpg,is_host=0):
    med = med_int + .1248*ath + .0965*year + .0289*gdpg + 6.6871*is_host
    return med


# ### Check for multicollinearity

# In[ ]:


dfNum.corr()


# In[ ]:


sns.pairplot(dfNum)


# In[ ]:


X = dfHost.drop(['Medal','Host','NOC','Athletes'],axis=1)
y = dfHost['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=123)

ml_reg  = lin_reg(X_train, y_train)

pd.Series(ml_reg.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# In[ ]:


dfHost_only = dfHost[dfHost['NOC'].isin(hosts['NOC'])]

X = dfHost_only.drop(['Medal','Host','NOC','Athletes'],axis=1)
y = dfHost_only['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=123)

ml_reg  = lin_reg(X_train, y_train)

pd.Series(ml_reg.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# In[ ]:


X = dfHost.drop(['Medal','Host','NOC'],axis=1)
y = dfHost['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=123)

ml_reg  = lin_reg(X_train, y_train)

pd.Series(ml_reg.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# In[ ]:


X = dfHost_only.drop(['Medal','Host','NOC'],axis=1)
y = dfHost_only['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=123)

ml_reg  = lin_reg(X_train, y_train)

pd.Series(ml_reg.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# #### There is DEFINITELY a home field advantage
#  * The host tends to win 127 extra medals versus all competitors and 99 extra medals versus past hosts.
#  * However, this is almost entirely due to the host country's ability to send more athletes.
#  * When accounting for the increased athlete count, host countries only win an extra 12 medals.
#  * When only looking at countries who have hosted in the past, host countries actually win 7 LESS medals.

# ### Make data frames specific to USA, GBR, and CHN

# In[ ]:


usaNum = dfHost.loc[dfHost['NOC'] == 'USA'].sort_values(by=['Year'])
gbrNum = dfHost.loc[dfHost['NOC'] == 'GBR'].sort_values(by=['Year'])
chnNum = dfHost.loc[dfHost['NOC'] == 'CHN'].sort_values(by=['Year'])


# ### Build functions for extrapolating athlete counts

# In[ ]:


from scipy.interpolate import interp1d
usaAth = interp1d(usaNum['Year'],usaNum['Athletes'], fill_value='extrapolate')
gbrAth = interp1d(gbrNum['Year'],gbrNum['Athletes'], fill_value='extrapolate')
chnAth = interp1d(chnNum['Year'],chnNum['Athletes'], fill_value='extrapolate')


# ### Predict USA 2020 medal count

# In[ ]:


X = usaNum.drop(['Medal','NOC','Host'],axis=1)
y = usaNum['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

med_usa = lin_reg(X_train, y_train)

pd.Series(med_usa.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# In[ ]:


usa_int = med_usa.intercept_.round(3)

def pred_medals_usa(ath,year,host=0):
    med = usa_int + .2992*ath + .2707*year + 31.2275*host
    return med


# In[ ]:


mod_perf(med_usa,X_train,y_train)


# In[ ]:


mod_perf(med_usa,X_test,y_test)


# In[ ]:


usaNum.tail()


# In[ ]:


year = 2020
ath = usaAth(year)
gdpg = 2.9
gdpp = 54541.7
print('USA 2020 Medal Count: %.0f' % pred_medals_usa(ath,year))


# In[ ]:


print('2020 Medal Count  w/ Base Model: %.0f' % pred_medals(ath,year,gdpg))


# ### Predict GBR 2020 medal count

# In[ ]:


X = gbrNum.drop(['Medal','NOC','Host'],axis=1)
y = gbrNum['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

med_gbr = lin_reg(X_train, y_train)

pd.Series(med_gbr.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# In[ ]:


gbr_int = med_gbr.intercept_.round(3)

def pred_medals_gbr(ath,year,host=0):
    med = gbr_int + .0774*ath - 0.8806*year - 20.4233*host
    return med


# In[ ]:


mod_perf(med_gbr,X_train,y_train)


# In[ ]:


gbrNum.tail()


# In[ ]:


year = 2020
ath = gbrAth(year)
gdpg = 1.4
gdpp = 42986
print('GBR 2020 Medal Count: %.0f' % pred_medals_gbr(ath,year))


# In[ ]:


print('2020 Medal Count  w/ Base Model: %.0f' % pred_medals(ath,year,gdpg))


# ### Predict CHN 2020 medal count

# In[ ]:


X = chnNum.drop(['Medal','NOC','Host'],axis=1)
y = chnNum['Medal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

med_chn = lin_reg(X_train, y_train)

pd.Series(med_chn.coef_, index=X_train.columns).sort_values(ascending=False).round(4)


# In[ ]:


chn_int = ml_reg.intercept_.round(3)

def pred_medals_chn(ath,year,host=0):
    med = chn_int + 0.0605*ath + 1.1477*year + 65.5159*host
    return med


# In[ ]:


mod_perf(med_chn,X_train,y_train)


# In[ ]:


chnNum.tail()


# In[ ]:


year = 2020
ath = chnAth(year)
gdpg = 6.6
gdpp = 7755
print('CHN 2020 Medal Count: %.0f' % pred_medals_chn(ath,year))


# In[ ]:


print('2020 Medal Count  w/ Base Model: %.0f' % pred_medals(ath,year,gdpg))


# In[ ]:




