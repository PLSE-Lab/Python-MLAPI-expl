#!/usr/bin/env python
# coding: utf-8

# Parameters that have to be changed for different forecast days

# In[ ]:


val = False #True - predict for validation period; False - predict for evaluation period
dend = 1914 if val else 1942 #first day of prediction period

wdayF = 3 #Forecasts for Mondays
hs = [8]
lags = ['lag0','lag1','lag2','lag3','lag6'] # 'lag1' actually translates to lag(1+h), 'lag0' to lag(h)

wdaysT = [3,7] #weekdays used for training; 1 Saturday - 7 Friday
wDayShifts = [[2,3,4,5,6],[0,1,2,3,6]]
wEndShifts = [[0,1],[4,5]]

dstart = 1100


# Features that I use

# In[ ]:


features = ['Level',
    
            'mday','wOfMonth','quarter','is_weekend','year','month','wday','snap_CA', #calendar features
            'snap_WI','snap_TX','2events','event_type_1','event_name_1',
           
           'sell_price','sell_price_rel_diff','sell_price_abs_diff','sell_price_cumrel','sell_price_max', #sell_price features
           'sell_price_min','sell_price_mean','sell_price_norm','sell_price_roll_sd4','sell_price_rollNorm', 
            
           'item_id','dept_id','cat_id','store_id','state_id','roll_mean_t7','roll_mean_t14','roll_mean_t28', #sales features
           'roll_mean_t56','roll_mean_t84','roll_mean_t168','rolling_sd_t7','rolling_sd_t14',
           'rolling_sd_t28','last8wdays','last4wdays','last2wdays','lastWorkDaysMean','lastWeekendMean',
           'lastWeekTrend','lastMonthTrend','last2MonthTrend','roll_quant10','roll_quant25','roll_quant50',
           'roll_quant75','roll_quant90']  


targetEncode = ['event_name_1','event_type_1','month','wday','mday', #intra id target encoding; added with _tar appendix to features
                'wOfMonth','quarter','2events','snap_CA','snap_WI','snap_TX']  
targetEncode2 = ['item_id','dept_id','store_id','id_enc'] #target encoding (not intra id); added with _tar appendix to features

features = features + lags


# LGBM Parameters to use

# In[ ]:


params = {'seed':20,
          'objective':'quantile', 
          'alpha':0.005,
          'num_leaves':63,
          'max_depth':15,                    
          'lambda':0.1, 
          'bagging_fraction':0.66,
          'bagging_freq':1,
          'colsample_bytree':0.77,
          "force_row_wise" : True,
          'learning_rate':0.1
        }


# ## Preliminary stuff ##
# 
# Code to install newest LGBM version

# In[ ]:


get_ipython().run_cell_magic('bash', '', "\ngit clone --recursive https://github.com/microsoft/LightGBM ; cd LightGBM\nmkdir build ; cd build\nexport CMAKE_CXX_FLAGS='-O3 -mtune=native'\ncmake ..\nmake -j$(nproc)\ncd ../python-package/\npython setup.py install --precompile")


# Importing libraries

# In[ ]:


import numpy as np # linear algebra
import numpy.ma as ma
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import time, gc, sys
import lightgbm as lgb

pd.set_option('display.max_columns', 500)


# Solving memory leak problem in pandas <br>
# https://github.com/pandas-dev/pandas/issues/2659#issuecomment-12021083 <br>
# (After pandas operations in the following code, call: libc.malloc_trim(0) and gc.collect())

# In[ ]:


from ctypes import cdll, CDLL
try:
    cdll.LoadLibrary("libc.so.6")
    libc = CDLL("libc.so.6")
    libc.malloc_trim(0)
except (OSError, AttributeError):
    libc = None

__old_del = getattr(pd.DataFrame, '__del__', None)

def __new_del(self):
    if __old_del:
        __old_del(self)
    libc.malloc_trim(0)

if libc:
    print('Applying monkeypatch for pd.DataFrame.__del__', file=sys.stderr)
    pd.DataFrame.__del__ = __new_del
else:
    print('Skipping monkeypatch for pd.DataFrame.__del__: libc or malloc_trim() not found', file=sys.stderr)


# Loading csv files

# In[ ]:


# Load data; weights contains weights and scaling coefficients for WRMSSE (calculated in getScalingCoeffs notebooks)
selling_prices = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv")
calendar = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/calendar.csv")
weights = pd.read_csv("/kaggle/input/getscalingcoeffs-uncertainty-evaluation/weightsAndScaling.csv") 
if val:
    salesRaw = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv")
else:
    salesRaw = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sales_train_evaluation.csv") 


# ## Data preprocessing ##

# Weights

# In[ ]:


weights['id'] = weights['Agg_Level_1']+'_'+weights['Agg_Level_2']
weights = weights[['id','Weight']]
weights.columns = ['id','weight']
weights['weight'] = (weights['weight']/weights['weight'].mean()).clip(lower=0.2,upper=30.).astype(np.float32)
weights.head(3)


# Calendar

# In[ ]:


calendar['d'] = calendar['d'].str[2:].astype(np.int16) #'d' column has form 'd_1', 'd_2' etc., delete 'd_'
calendar['mday'] = calendar['date'].str[-2:].astype(np.int8)
calendar['wOfMonth'] = calendar['mday']//7
calendar['event_name_1'] = (calendar['event_name_1'].factorize()[0] + 1).astype(np.int8) 
calendar['event_type_1'] = (calendar['event_type_1'].factorize()[0] + 1).astype(np.int8) 
calendar['2events'] = (~calendar['event_type_2'].isna()).astype(np.int8)
calendar['quarter'] = ((calendar['month']-1)//3).astype(np.int8)
calendar['is_weekend'] = calendar['wday'].isin([1,2]).astype(np.int8)
calendar['year'] = (calendar['year']-calendar['year'].min()).astype(np.int8)
calendar = calendar.astype({'wm_yr_wk':np.int32,'wday':np.int8,'month':np.int8,
                            'snap_CA':np.int8,'snap_TX':np.int8,'snap_WI':np.int8})
calendar.drop(columns=['date','weekday','event_name_2','event_type_2'],inplace=True) #drop date column (there are still columns wday, month, year left -> no info lost)
calendar = calendar[calendar['d']>=dstart]
calendar.head(3)


# Selling prices 

# In[ ]:


### Building the 42840 rows from the 30490 ones (using mean selling price)
selling_prices = selling_prices[selling_prices['wm_yr_wk']>=calendar['wm_yr_wk'].min()]

selling_prices['total'] = 'Total'
selling_prices['cat_id'] = selling_prices['item_id'].map(lambda x: x.split('_')[0])
selling_prices['dept_id'] = selling_prices['item_id'].map(lambda x: x.split('_')[0] + '_' + x.split('_')[1])
selling_prices['state_id'] = selling_prices['store_id'].map(lambda x: x.split('_')[0])

level_dfs = []
group_ids = ( 'total', 'state_id', 'store_id', 'cat_id', 'dept_id', 
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], 'item_id', ['state_id','item_id'], ['item_id', 'store_id']) 
for ii,grou in enumerate(group_ids): 
    if type(grou)==list:
        tmp = selling_prices.groupby(['wm_yr_wk']+grou).mean().reset_index()
        tmp['id'] = tmp[grou[0]] + '_' + tmp[grou[1]]
    else:
        tmp = selling_prices.groupby(['wm_yr_wk',grou]).mean().reset_index()
        tmp['id'] = tmp[grou] + '_X'
    for f in [x for x in ['state_id','store_id','cat_id','dept_id','item_id'] if x not in grou]:
        if type(grou)==list and (selling_prices.groupby(['wm_yr_wk']+grou)[f].nunique()==1).all():
            tmp = tmp.merge(selling_prices.groupby(['wm_yr_wk']+grou)[f].first().reset_index(),on=['wm_yr_wk']+grou)
        elif type(grou)==str and (selling_prices.groupby(['wm_yr_wk',grou])[f].nunique()==1).all():
            tmp = tmp.merge(selling_prices.groupby(['wm_yr_wk',grou])[f].first().reset_index(),on=['wm_yr_wk',grou])
        else:
            tmp[f] = np.nan
    if grou=='total':
        tmp.drop(columns=['total'],inplace=True)
    level_dfs += [tmp.copy()]
selling_prices = pd.concat(level_dfs,sort=False).reset_index().drop(columns='index')
selling_prices.drop(columns=['state_id','store_id','cat_id','dept_id','item_id'],inplace=True)
del level_dfs, tmp
#Pandas memory leak issue
libc.malloc_trim(0)
gc.collect()
selling_prices.head(3)


# In[ ]:


### insert nan prices for missing periods
wm_yr_wk_uniques = selling_prices['wm_yr_wk'].unique()
outOfStore_id = []
outOfStore_wm_yr_wk = []
for idd,avail_wk in selling_prices.groupby('id').wm_yr_wk.unique().iteritems():
    outOfStore_id += (len(wm_yr_wk_uniques) - len(avail_wk)) * [idd]
    outOfStore_wm_yr_wk += list(wm_yr_wk_uniques[~np.isin(wm_yr_wk_uniques,avail_wk)])
selling_prices = selling_prices.append(pd.DataFrame({'id':outOfStore_id,'wm_yr_wk':outOfStore_wm_yr_wk,'sell_price':np.nan}),ignore_index=True,sort=True)
del wm_yr_wk_uniques, outOfStore_id, outOfStore_wm_yr_wk

selling_prices = selling_prices.sort_values(by=['id', 'wm_yr_wk']) #sort values s.t. the groupby followed by shifts etc. works correctly in the following lines
selling_prices['sell_price_rel_diff'] =  (selling_prices['sell_price']/selling_prices.groupby('id')['sell_price'].shift(1)).astype(np.float16) #price change compared to last week
selling_prices['sell_price_abs_diff'] = (selling_prices['sell_price'] - selling_prices.groupby('id')['sell_price'].shift(1)).astype(np.float16) #price change compared to last week
selling_prices['sell_price_cumrel'] = ((selling_prices['sell_price'] - selling_prices.groupby('id')['sell_price'].cummin()) / (1e-4 + selling_prices.groupby('id')['sell_price'].cummax() - selling_prices.groupby('id')['sell_price'].cummin())).astype(np.float16) # (current prize - lowest price up to now) normalized by prize span until now; 
selling_prices['sell_price_max'] = selling_prices['id'].map(selling_prices.groupby('id')['sell_price'].max()).astype(np.float16)
selling_prices['sell_price_mean'] = selling_prices['id'].map(selling_prices.groupby('id')['sell_price'].mean()).astype(np.float16)
selling_prices['sell_price_min'] = selling_prices['id'].map(selling_prices.groupby('id')['sell_price'].min()).astype(np.float16)
selling_prices['sell_price_norm'] = (selling_prices['sell_price']/selling_prices['sell_price_max']).astype(np.float16)
selling_prices['sell_price_rollNorm'] = (selling_prices['sell_price']/selling_prices.groupby('id')['sell_price'].rolling(window=4,min_periods=2).mean().reset_index((0,1),drop=True)).astype(np.float16)
selling_prices['sell_price_roll_sd4'] = selling_prices.groupby('id')['sell_price'].rolling(window=4,min_periods=3).std().reset_index((0,1),drop=True).astype(np.float16) #standard deviation of the selling prices in the last 7 weeks
selling_prices = selling_prices.astype({'sell_price':np.float16,'wm_yr_wk':np.int32}) #reduce memory consumption
selling_prices.head(3)


# Sales

# In[ ]:


### Building the 42840 rows from the 30490 ones in sales
salesRaw['id'] = salesRaw['id'].str[:-11] #delete the 'evaluation'/'validation' suffix in the 'id' column

salesRaw['total'] = 'Total' 
level_dfs = []
group_ids = ( 'total', 'state_id', 'store_id', 'cat_id', 'dept_id', 
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], 'item_id', ['state_id','item_id'], ['item_id', 'store_id']) 
for ii,grou in enumerate(group_ids): 
    if type(grou)==list:
        tmp = salesRaw.groupby(grou).sum().reset_index()
        tmp['id'] = tmp[grou[0]] + '_' + tmp[grou[1]]
    else:
        tmp = salesRaw.groupby(grou).sum().reset_index()
        tmp['id'] = tmp[grou] + '_X'
    for f in [x for x in ['state_id','store_id','cat_id','dept_id','item_id'] if x not in grou]:
        if (salesRaw.groupby(grou)[f].nunique()==1).all():
            tmp = tmp.merge(salesRaw.groupby(grou)[f].first().reset_index(),on=grou)
        else:
            tmp[f] = np.nan
    if grou=='total':
        tmp.drop(columns=['total'],inplace=True)
    tmp['Level']=ii+1
    level_dfs += [tmp.copy()]
salesRaw = pd.concat(level_dfs,sort=False).reset_index().drop(columns='index')
salesRaw['Level'] = salesRaw['Level'].astype(np.int8)
del level_dfs, tmp
#Pandas memory leak issue
libc.malloc_trim(0)
gc.collect()
salesRaw.head(3)


# In[ ]:


#Sales - have to be processed for every h -> write function
def getProcessedSales(h):
    sales = salesRaw.copy()
    sales = sales.astype(dict(zip(['d_'+str(x) for x in range(1,dend)],(dend-1)*[np.float32])))
    sales = sales.sort_values(by='id')
    
    print('Check 1')
    
    # remove by interpolation the extreme events Christmas and Thanksgiving and other clear outliers
    christmas = [331,697,1062,1427,1792] #d_x is Christmas
    thanksgiving = [300, 664, 1035, 1399, 1763] 
    outliers = [1413]
    rmDays = outliers + christmas + thanksgiving #completely remove those days later (interpolate here for better rolling means etc)
    for d in rmDays:
        sales['d_'+str(d)] = 0.5 * (sales['d_'+str(d-1)] + sales['d_'+str(d+1)])    
        
    #Normalize sales (intra-id processing; first adjust mean s.t. sales for all ids have similar means, then remove trends)
    salesMat = sales[['d_'+str(ii) for ii in range(1,dend)]].values.astype(np.float32)
    newInStore = np.argmax(salesMat!=0,axis=1)//7 * 7 #always new for weekstart; set sales before release to nan instead of zero.
    for ii in range(len(salesMat)):
        salesMat[ii,:newInStore[ii]] = np.nan
    #mean adjustment
    means = np.apply_along_axis(lambda row: np.nanmean(row[row!=0]), axis=1, arr=salesMat) #normalize with the mean of NONZERO elements
    salesMat = (salesMat.T/means).T
    factors = pd.DataFrame({'id':sales['id'],'factor':means}) #to get original data multiply prediction with factor
    #trend adjustment
    rollMean = np.apply_along_axis(lambda row: np.convolve(row, np.ones(28)/28,mode='valid'), axis=1, arr=salesMat)
    rollMean[np.isnan(rollMean)] = np.apply_along_axis(lambda row: np.convolve(row, np.ones(7)/(7),mode='valid'), axis=1, arr=salesMat)[:,28-7:][np.isnan(rollMean)]
    adjust = (np.nanmax(rollMean,axis=1) - rollMean.T).T
    adjust[np.isnan(adjust)] = 0
    salesMat[:,-(adjust.shape[1]-h):] += adjust[:,:-h] #shift the adjustment by h to be able to undo it for prediction period
    trendAdjust = pd.DataFrame({'id':sales['id'],'adjust':adjust[:,-1]})
    #write to sales dataframe
    sales.loc[:,['d_'+str(ii) for ii in range(1,dend)]] = salesMat    
    
    del salesMat, newInStore, means, adjust
    
    sales.drop(columns=['d_'+str(x) for x in range(1,dstart)],inplace=True) #drop columns before dstart
    for d in range(dend,dend+h): #create new columns for prediction days until horizon
        sales['d_'+str(d)] = 0
    
    #restructure the table: for each object (defined by its id), the columns d_1,...,d_1969 are replaced by 1969 rows with 
    #columns 'd' - day id and 'demand' - number sold articles on that day
    sales = sales.melt(id_vars=("id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "Level"),var_name='d',value_name='demand') 
    sales['d'] = sales['d'].str[2:].astype(np.int16) #'d' column has form 'd_1', 'd_2' etc., delete 'd_'
    
    #print('Check 3')
    
    #encode columns as integers
    sales['item_id'] = sales['item_id'].factorize()[0].astype(np.int16) + 1 
    sales['dept_id'] = sales['dept_id'].factorize()[0].astype(np.int8) + 1
    sales['cat_id'] = sales['cat_id'].factorize()[0].astype(np.int8) + 1
    sales['store_id'] = sales['store_id'].factorize()[0].astype(np.int8) + 1
    sales['state_id'] = sales['state_id'].factorize()[0].astype(np.int8) + 1
    sales['state_id'] = sales['state_id'].astype(np.int8) + 1
    sales['id_enc'] = sales['id'].factorize()[0].astype(np.int16) + 1 
    
    #merge with calendar s.t. we can build target encodings as well for calendar features
    sales = sales.merge(calendar,on='d')
    
    #print('Check 4')
    
    #targetEncode; intra id target encoding; will be added with _tar appendix to the features;
    for col in targetEncode:
        sales = sales.merge(sales[sales['d']<dend].groupby(['id',col])['demand'].mean().rename(col+'_tar').reset_index())
        sales[col+'_tar'] = sales[col+'_tar'].astype(np.float16)
        features.append(col+'_tar')
        #Pandas memory leak issue
        libc.malloc_trim(0)
        gc.collect()
    #targetEncode2; not intra id; will be added with _tar appendix to the features;
    for col in targetEncode2: 
        sales = sales.merge(sales[sales['d']<dend].groupby(col)['demand'].mean().rename(col+'_tar').reset_index(),on=col)
        sales[col+'_tar'] = sales[col+'_tar'].astype(np.float16)
        features.append(col+'_tar')
        #Pandas memory leak issue
        libc.malloc_trim(0)
        gc.collect()

    #print('Check 5')
    
    #compute further sales features
    sales = sales.sort_values(by=['id', 'd']) #sort values s.t. the groupby followed by shifts etc. works correctly in the following
    sales['demand'] = sales['demand'].astype(np.float32)
    sales['lag0'] = sales.groupby('id')['demand'].shift(h).astype(np.float16)
    #rolling means
    sales['roll_mean_t7'] = sales.groupby('id')['lag0'].rolling(window=7,min_periods=7).mean().reset_index(0,drop=True).astype(np.float16)
    sales['roll_mean_t14'] = sales.groupby('id')['lag0'].rolling(window=14,min_periods=7).mean().reset_index(0,drop=True).astype(np.float16)
    sales['roll_mean_t28'] = sales.groupby('id')['lag0'].rolling(window=28,min_periods=14).mean().reset_index(0,drop=True).astype(np.float16)
    sales['roll_mean_t56'] = sales.groupby('id')['lag0'].rolling(window=56,min_periods=14).mean().reset_index(0,drop=True).astype(np.float16)
    sales['roll_mean_t84'] = sales.groupby('id')['lag0'].rolling(window=84,min_periods=28).mean().reset_index(0,drop=True).astype(np.float16)
    sales['roll_mean_t168'] = sales.groupby('id')['lag0'].rolling(window=168,min_periods=28).mean().reset_index(0,drop=True).astype(np.float16)
    #rolling quantiles
    sales['roll_quant10'] = sales.groupby('id')['lag0'].rolling(window=35,min_periods=21).quantile(0.1).reset_index(0,drop=True).astype(np.float16)
    sales['roll_quant25'] = sales.groupby('id')['lag0'].rolling(window=21,min_periods=14).quantile(0.25).reset_index(0,drop=True).astype(np.float16)
    sales['roll_quant50'] = sales.groupby('id')['lag0'].rolling(window=14,min_periods=7).quantile(0.5).reset_index(0,drop=True).astype(np.float16)
    sales['roll_quant75'] = sales.groupby('id')['lag0'].rolling(window=21,min_periods=14).quantile(0.75).reset_index(0,drop=True).astype(np.float16)
    sales['roll_quant90'] = sales.groupby('id')['lag0'].rolling(window=35,min_periods=21).quantile(0.9).reset_index(0,drop=True).astype(np.float16)
    #rolling sd features
    sales['rolling_sd_t7'] = sales.groupby('id')['lag0'].rolling(window=7,min_periods=7).std().reset_index(0,drop=True).astype(np.float16) #standard deviation of demand 28-34 days earlier
    sales['rolling_sd_t14'] = sales.groupby('id')['lag0'].rolling(window=14,min_periods=7).std().reset_index(0,drop=True).astype(np.float16) #standard deviation of demand 28-34 days earlier
    sales['rolling_sd_t28'] = sales.groupby('id')['lag0'].rolling(window=28,min_periods=14).std().reset_index(0,drop=True).astype(np.float16)
    #last same wdays
    sales['last8wdays'] = sales.groupby('id')['demand'].shift(h+7-h%7)
    for ii in range(2,9):
        sales['last8wdays'] += sales.groupby('id')['demand'].shift(h+ii*7-h%7)
        if ii == 2:
            sales['last2wdays'] = (sales['last8wdays']/2).astype(np.float16)
        if ii == 4:
            sales['last4wdays'] = (sales['last8wdays']/4).astype(np.float16)
    sales['last8wdays'] = (sales['last8wdays']/8).astype(np.float16)
    sales['last4wdays'].fillna(sales['last2wdays'],inplace=True) #to reduce nb of nan entries
    sales['last8wdays'].fillna(sales['last4wdays'],inplace=True)
    #others
    sales['lastWorkDaysMean'] = np.zeros(len(sales),dtype=np.float16)
    sales['lastWeekendMean'] = np.zeros(len(sales),dtype=np.float16)
    for ii in range(len(wdaysT)):
        sales.loc[sales['wday']==wdaysT[ii],'lastWorkDaysMean'] = sales.groupby('id')['demand'].shift(h+wDayShifts[ii][0])[sales['wday']==wdaysT[ii]]
        for jj in range(1,len(wDayShifts[ii])):
            sales.loc[sales['wday']==wdaysT[ii],'lastWorkDaysMean'] += sales.groupby('id')['demand'].shift(h+wDayShifts[ii][jj])[sales['wday']==wdaysT[ii]]
        sales.loc[sales['wday']==wdaysT[ii],'lastWeekendMean'] = ((sales.groupby('id')['demand'].shift(h+wEndShifts[ii][0])+sales.groupby('id')['demand'].shift(h+wEndShifts[ii][1]))/2).astype(np.float16)[sales['wday']==wdaysT[ii]] #only given that one considers only the forecast wday
        sales['lastWorkDaysMean'] /= 5    
    sales['lastWeekTrend'] = sales['roll_mean_t7'].shift(7) - sales['roll_mean_t7']
    sales['lastMonthTrend'] =  sales['roll_mean_t28'].shift(28) - sales['roll_mean_t28']
    sales['last2MonthTrend'] =  sales['roll_mean_t56'].shift(56) - sales['roll_mean_t56']
    # additional lags: 'lag0' actually translates to lag(h), 'lag1' to lag(h+1) and so on
    tmp = pd.DataFrame({'id':sales['id'].values,'d':sales['d'].values,'demand':sales['demand'].values}) #somehow this is necessary; going directly for sales[['id','demand']].groupby('id').shift(...) or even sales.groupby('id').shift(...) produces memory errors
    tmp = tmp.sort_values(by=['id', 'd']) #same sorting as for sales; lag stuff below should work correctly
    for lag in lags:
        if lag != 'lag0':
            sales[lag] = tmp.groupby('id')['demand'].shift(h+int(lag[3:])).astype(np.float16).values
    
    del tmp
    #Pandas memory leak issue
    libc.malloc_trim(0)
    gc.collect()
    
    #print('Check 6')
    
    # consider only days in wdaysT for training (and forecasting...)
    sales = sales[sales['wday'].isin(wdaysT)] #drop all other days
    #remove rows that have nan values 
    sales = sales[(~sales.isna().any(axis=1)) | (sales['d'] >= dend-28)] 
    #remove rmDays
    sales = sales[~sales['d'].isin(rmDays)] 
    sales['demand'].fillna(0,inplace=True)
    
    #Discard everything of CA_2 before 2016 and of WI_1, WI_2 before 2013 (due to the great shifts at that dates shown in the EDA kernel of heads or tails:
    #https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda )
    sales = sales[~(sales['id'].str.contains('CA_2') & (sales['year']<5))]
    sales = sales[~(sales['id'].str.contains('|'.join(['WI_1','WI_2'])) & (sales['year']<2))]
    
    #print('Check 7')
    
    #merge with selling prices
    sales = selling_prices.merge(sales,on=['id', 'wm_yr_wk'])
    sales.drop(columns='wm_yr_wk',inplace=True) #not used as feature, was just needed for merge
    
    #print('Check 8')
    
    #merge with weights for LGBM
    sales = weights.merge(sales,on='id')
    
    #print('Check 9')
    
    return sales, factors, trendAdjust


# ## Start Modelling ##

# In[ ]:


predictions = pd.DataFrame({'id':pd.concat(weights['id']+'_'+str(q) for q in [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])})
for h in hs:
    print('Modelling for horizon h =',h)
    print(20*'_')
    sales, factors, trendAdjust = getProcessedSales(h)
    
    train = sales[sales['d'] < dend-28] #use last 28 days for early stopping
    train = lgb.Dataset(train[features],train[['demand']],weight=train['weight'])
    
    validES = sales[(sales['d'] >= dend - 28) & (sales['d'] < dend)] #validation set for  early stopping
    validES = lgb.Dataset(validES[features],validES[['demand']],reference=train,weight=validES['weight'])
    
    ids = []
    preds = []
    for q in [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]:
        print('Quartile', q)
        print(20*'_')
        params['alpha']=q
        pred = sales[sales['d']==dend+h-1].copy() 
    
        model = lgb.train(params,train_set=train,valid_sets=[train,validES],valid_names=['train','validES'],verbose_eval = 100,
                         early_stopping_rounds=100,num_boost_round=3000)

        lgb.plot_importance(model,importance_type="gain", precision=0, height=0.5, figsize=(10, 15))
        
        pred['demand'] = model.predict(pred[features])
        pred = pred[['id','demand']] #bring to correct format for WRMSSE function
        pred.columns = ['id','F'+str(h)]
        # undo preprocessing
        pred = pred.merge(factors,on='id')  #trendAdjust = pd.DataFrame({'id':sales.id.str[:-11],'adjust'
        pred = pred.merge(trendAdjust,on='id')
        #trend
        pred['F'+str(h)] = pred['F'+str(h)] - pred['adjust']
        #mean
        pred['F'+str(h)] = pred['F'+str(h)] * pred['factor']
        pred.drop(columns=['adjust','factor'],inplace=True) 
        #clip predicted values below zero
        pred['F'+str(h)] = pred['F'+str(h)].clip(lower=0)
        
        #save predictions for this quantile to merge later on to predictions dataframe
        ids += list((pred['id'] + '_' + str(q)).values)
        preds += list(pred['F'+str(h)].values)
        
        del model, pred
        #Pandas memory leak issue
        libc.malloc_trim(0)
        gc.collect()
        print()
    
    predictions = predictions.merge(pd.DataFrame({'id':ids,'F'+str(h):preds}),on='id')
    predictions['F'+str(h)] = predictions['F'+str(h)].astype(np.float32)
    del train, validES
    #Pandas memory leak issue
    libc.malloc_trim(0)
    gc.collect()
    print()
    print()


# In[ ]:


predictions.to_csv('preds.csv',index=False)


# In[ ]:


predictions.describe()

