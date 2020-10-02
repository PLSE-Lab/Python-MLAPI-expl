#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Start with importing essentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# #### 1. Read the train set and property set of Zillow dataset, and name them as train and properties.

# In[ ]:


df_train = pd.read_csv('../input/train_2016_v2.csv')
df_property = pd.read_csv('../input/properties_2016.csv')


# In[ ]:


df_train.shape


# In[ ]:


df_train.head()


# In[ ]:


df_property.shape


# In[ ]:


df_property.head()


# #### 2. Merge train and properties to one dataframe on parcelid and call it as df_train. Drop the column of 'parcelid' and 'transactiondate'. Check the first 5 rows to see how this merged dataset looks like.

# In[ ]:


df_train = df_train.merge(df_property, how='left', on='parcelid')


# In[ ]:


df_train.shape


# In[ ]:


df_train.drop(['parcelid','transactiondate'],axis=1,inplace=True)


# In[ ]:


df_train.shape


# In[ ]:


df_train.head()


# #### 3.  (a) Generate a dataframe called missing_df from df_train, in which there are two columns, one is the column names of our features, the other column is the missing_count (the number of missing values) of that feature. The table should be ordered by missing_count decendingly.  

# In[ ]:


missing_df = df_train.isna().sum().to_frame().reset_index()  #create dataframe with missing data information of each column
missing_df.columns = ['column name','#missing values'] # rename column names 
missing_df = missing_df.loc[missing_df['#missing values']>0] # only keep those who have more than 1 missing value
missing_df.sort_values(by='#missing values',ascending=True,inplace=True) # sort missing value
missing_df 


# In[ ]:


missing_df.shape


# #### 3.(b) Draw a horizontal bar plot to visualize it. 

# In[ ]:


missing_df.plot.barh(x='column name', y = '#missing values', figsize=(15,20))


# In[ ]:


plt.figure(figsize=(20,30))
plt.barh(missing_df['column name'],missing_df['#missing values'],log=True)


# #### 4. Generate the correlation matrix for all the numerical features, and plot it by using heatmap or related visualization methods. 

# In[ ]:


catcols = ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid','buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood','regionidzip','storytypeid','typeconstructiontypeid','yearbuilt','taxdelinquencyflag']
numcols = [x for x in df_train.columns if x not in catcols]


# In[ ]:


numcols


# In[ ]:


plt.figure(figsize = (14,12))
sns.heatmap(data=df_train[numcols].corr())
plt.show()


# #### 5. From the results from Step 4,  list those features having a strong correlation. Generate a list called dropcols, and put those redundent variables into it.

# In[ ]:


dropcols = []
dropcols.append('finishedsquarefeet12')
dropcols.append('finishedsquarefeet13')
dropcols.append('finishedsquarefeet15')
dropcols.append('finishedsquarefeet6')
dropcols.append('finishedsquarefeet50')
dropcols.append('calculatedbathnbr')
dropcols.append('fullbathcnt')


# #### 6. Some variables where it is NA can be considered as the object does not exist. Such as 'hashottuborspa', if it is NA, we can assume the house doesn't contain the hot tub or spa. So we need to fix this kind of variables.

# (a) Fix the hashottuborspa variable, fill the na part as None.

# In[ ]:


df_train['hashottuborspa'].isnull().sum()


# In[ ]:


df_train['hashottuborspa']=df_train['hashottuborspa'].fillna('None', inplace=True)
# index = df_train.hashottuborspa.isnull()
# df_train.loc[index,'hashottuborspa'] = 'None'


# (b) Assume if the pooltype id and its related features is null then pool/hottub doesn't exist.

# In[ ]:


df_train[df_train.pooltypeid10.isnull()
         |df_train.pooltypeid2.isnull()
         |df_train.pooltypeid7.isnull()].hashottuborspa.fillna('None', inplace =True)


# In[ ]:


df_train.loc[0:5,['hashottuborspa']]


# (c) taxdelinquencyflag - assume if it is null then doesn't exist

# In[ ]:


df_train['taxdelinquencyflag'] = df_train['taxdelinquencyflag'].fillna("doesn't exist",inplace=True)


# In[ ]:


df_train.loc[0:5,['taxdelinquencyflag']]


# (d) If Null in garage count (garagecarcnt) it means there are no garages, and no garage means the size (garagetotalsqft) is 0 by default

# In[ ]:


df_train.loc[df_train.garagecarcnt.isnull(),'garagetotalsqft']=0
# setting value for items matching condition 


# In[ ]:


df_train.loc[0:10,['garagecarcnt','garagetotalsqft']]


# #### 7. There are more missing values in the 'poolsizesum' than in 'poolcnt'. Fill in median values for poolsizesum where pool count is >0 and missing.

# In[ ]:


poolsizesum_median = df_train.loc[df_train['poolcnt']>0,'poolsizesum'].median()
poolsizesum_median


# In[ ]:


df_train.loc[(df_train['poolcnt']>0) & (df_train['poolsizesum'].isnull()),'poolsizesum']=poolsizesum_median
df_train.loc[(df_train['poolcnt']==0),'poolsizesum']=0


# In[ ]:


df_train.loc[:5,'poolsizesum']


# #### 8. The number of missing value of 'fireplaceflag' is more than the 'fireplacecnt'. So we need to mark the missing 'fireplaceflag' as Yes when fireplacecnt>0, then the rest of 'fireplaceflag' should be marked as No. Then for the missing part in fireplacecnt, we can consider the number of fire place is 0.

# In[ ]:


df_train.fireplaceflag.value_counts()


# In[ ]:


df_train.fireplaceflag='No'


# In[ ]:


df_train.fireplaceflag.value_counts()


# In[ ]:


df_train.loc[df_train['fireplacecnt']>0,'fireplaceflag']='Yes'
df_train.loc[df_train['fireplacecnt'].isnull(),'fireplaceflag']=0


# In[ ]:


df_train.fireplaceflag.value_counts()


# #### 9. Fill some features with the most common value for those variables where this might be a sensible approach:

# (a) AC Type (airconditioningtypeid)- Mostly 1's, which corresponds to central AC. It is reasonable to assume most other properties where this feature is missing are similar.

# In[ ]:


df_train.airconditioningtypeid.fillna(1,inplace= True)


# (b) heating or system (heatingorsystemtypeid)- Mostly 2, which corresponds to central heating so seems reasonable to assume most other properties have central heating.

# In[ ]:


df_train.heatingorsystemtypeid.fillna(2,inplace= True)


# #### 10. If the features where missing proportion is too much, we can directly delete them. Here we set 97% as our threshold (This is subjective) and add them into the dropcols. Then drop those features in dropcols from the full table.

# In[ ]:


missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()
missingvalues_prop.columns = ['name','proportion']
missingvalues_prop = missingvalues_prop.sort_values (by = 'proportion', ascending=False)
print(missingvalues_prop)
missingvaluescols = missingvalues_prop [missingvalues_prop['proportion']>0.97].field.tolist()
dropcols = dropcols + missingvaluescols
df_train = df_train.drop (dropcols,axis=1)


# #### 11. We can also use some machine learning algorithm to fill the missing data. 
# In this dataset, there's quite a few variables which are probably dependant on longtitude and latitude data. It is reasonable to fill in some of the missing variables using geographically nearby properties (by using the longtitude and latitude information).

# In[ ]:


a = np.array([True,False])
print(~a)


# The following code comes from the link:
# https://www.kaggle.com/auroralht/restoring-the-missing-geo-data

# In[ ]:


from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

## Works on categorical feature
def fillna_knn( df, base, target, fraction = 1, threshold = 10, n_neighbors = 5 ):
    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 
    whole = [ target ] + base
    
    miss = df[target].isnull()
    notmiss = ~miss 
    nummiss = miss.sum()
    
    enc = OneHotEncoder()
    X_target = df.loc[ notmiss, whole ].sample( frac = fraction )
    
    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )
    
    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()
    X = X_target[ base  ]
    
    print( 'fitting' )
    n_neighbors = n_neighbors
    clf = neighbors.KNeighborsClassifier( n_neighbors, weights = 'uniform' )
    clf.fit( X, Y )
    
    print( 'the shape of active features: ' ,enc.active_features_.shape )
    
    print( 'predicting' )
    Z = clf.predict(df.loc[miss, base])
    
    numunperdicted = Z[:,0].sum()
    if numunperdicted / nummiss *100 < threshold :
        print( 'writing result to df' )    
        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )
        print( 'num of unperdictable data: ', numunperdicted )
        return enc
    else:
        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )

#function to deal with variables that are actually string/categories
def zoningcode2int( df, target ):
    storenull = df[ target ].isnull()
    enc = LabelEncoder( )
    df[ target ] = df[ target ].astype( str )

    print('fit and transform')
    df[ target ]= enc.fit_transform( df[ target ].values )
    print( 'num of categories: ', enc.classes_.shape  )
    df.loc[ storenull, target ] = np.nan
    print('recover the nan value')
    return enc

### Example: 
### If you want to impute buildingqualitytypeid with geological information:
"""
fillna_knn( df = df_train,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'buildingqualitytypeid', fraction = 0.15, n_neighbors = 1 )
"""

## Works on regression
def fillna_knn_reg( df, base, target, n_neighbors = 5 ):
    cols = base + [target]
    X_train = df[cols]
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train[base].values.reshape(-1, 1))
    rescaledX = scaler.transform(X_train[base].values.reshape(-1, 1))

    X_train = rescaledX[df[target].notnull()]
    Y_train = df.loc[df[target].notnull(),target].values.reshape(-1, 1)

    knn = KNeighborsRegressor(n_neighbors, n_jobs = -1)    
    # fitting the model
    knn.fit(X_train, Y_train)
    # predict the response
    X_test = rescaledX[df[target].isnull()]
    pred = knn.predict(X_test)
    df.loc[df_train[target].isnull(),target] = pred
    return


# **Find out some features you can use this knn to fill the missing data, and use the above funtion to impute them**

# In[ ]:


df_train.columns.values


# In[ ]:


df_train.loc[:,['latitude','longitude']]


# In[ ]:


df_train.loc[:,['latitude','longitude']].isnull().sum()


# In[ ]:


fillna_knn( df = df_train,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'regionidcity', fraction = 0.15, n_neighbors = 1 )

