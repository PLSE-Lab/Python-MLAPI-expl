#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('bmh')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 20, 10


# In[ ]:


def nullremove(df):
    for i in range(0,len(df.columns)):
        print(i)
    


# In[ ]:


def descr(df):
    print(df.describe().round())


# In[ ]:


app=pd.read_csv("../input/application_train.csv")


# In[ ]:


print(app.shape)


# In[ ]:


application=app.copy()
print(application.shape)


# In[ ]:


application_object=application.select_dtypes("object")
print(application_object.shape)
print(application_object.columns)


# In[ ]:


application_object_columns=application_object.columns.tolist()
application_columns=application.columns.tolist()
print(len(application_object_columns))
print(len(application_columns))


# In[ ]:


app_if = list(set(application_columns) - set(application_object_columns))
print(len(app_if))


# In[ ]:


app_df=application[app_if]

print(app_df.shape)


# In[ ]:


app_df.dtypes


# In[ ]:


def nullremove(df,percentage):
   # print(len(df))
    wantedcolumnlist=[]
    columnlist=df.columns.tolist()
    #print(len(columnlist))
    for i in range(len(columnlist)):
        columnname=columnlist[i]
        nan_rows = df[columnname].isnull().sum()
     #   print(i,nan_rows)
        nullcount=(nan_rows/len(df))*100
        #print(nullcount)
        if nullcount>percentage:
            print(columnname) 
        else:
            wantedcolumnlist.append(columnname)    
        
    df=df[wantedcolumnlist]
    return df
    


# In[ ]:


#app_df=nullremove(app_df,80)


# In[ ]:


app_df.isna().sum()


# In[ ]:


app_df['DAYDIFF']=abs(app_df['DAYS_BIRTH'])-abs(app_df['DAYS_EMPLOYED'])


# In[ ]:


app_df=app_df[app_df['DAYDIFF']>0]


# In[ ]:


print(app_df.shape)


# In[ ]:


def meansep(df):
    highermean=pd.DataFrame()
    lowermean=pd.DataFrame()
    columns=df.columns.tolist()
    for i in range(len(columns)):
        mean=df[columns[i]].mean()
        if mean>10:
            highermean[columns[i]]=df[columns[i]]
        else:
            if columns[i]=='DAYS_BIRTH' or columns[i]=='DAYS_EMPLOYED' or columns[i]=='TARGET':
                highermean[columns[i]]=df[columns[i]]
            else:
                lowermean[columns[i]]=df[columns[i]]
    return highermean,lowermean


# In[ ]:


app_dff=app_df.copy()


# In[ ]:


app_dff.isna().sum()


# In[ ]:


app_dff=nullremove(app_dff,50)


# In[ ]:


hdf,ldf=meansep(app_dff)
print(hdf.shape,ldf.shape)


# In[ ]:


hdf.isna().sum()


# In[ ]:


hdf.dtypes


# In[ ]:


hdf['AMT_ANNUITY']=hdf['AMT_ANNUITY'].fillna(hdf['AMT_ANNUITY'].mean())
hdf['AMT_GOODS_PRICE']=hdf['AMT_GOODS_PRICE'].fillna(hdf['AMT_GOODS_PRICE'].mean())


# In[ ]:


descr(hdf)


# In[ ]:


hdf=hdf[hdf['AMT_INCOME_TOTAL']<=2000000]
hdf=hdf[hdf['AMT_CREDIT']<=900000]
hdf=hdf[hdf['AMT_GOODS_PRICE']<=900000]
hdf=hdf[hdf['AMT_ANNUITY']<=35000]


# In[ ]:


hdf[hdf['DAYS_EMPLOYED']==-17912]


# In[ ]:


application[application['AMT_CREDIT']==4050000.0]


# In[ ]:


hdf.columns.tolist()


# In[ ]:


hdf[['AMT_GOODS_PRICE',
  'AMT_INCOME_TOTAL',
 'AMT_ANNUITY',
 'DAYS_EMPLOYED',
  'DAYS_BIRTH',
  'AMT_CREDIT',
 'DAYDIFF']].hist(figsize=(16, 10), bins=50, xlabelsize=8, ylabelsize=8);


# In[ ]:


correlation = hdf[['AMT_GOODS_PRICE',
  'AMT_INCOME_TOTAL',
 'AMT_ANNUITY',
 'DAYS_EMPLOYED',
  'DAYS_BIRTH',
  'AMT_CREDIT',
 'DAYDIFF']].corr()
plt.figure(figsize=(7,7))
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap=flatui)
plt.title('Correlation between different fearures')


# In[ ]:


from sklearn.preprocessing import Normalizer
normalized_application = Normalizer().fit_transform(hdf[['AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','AMT_CREDIT','DAYDIFF']])
#print (normalized_application)
correlation=pd.DataFrame(normalized_application).corr()
correlation.columns=['AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','AMT_CREDIT','DAYDIFF']
plt.figure(figsize=(8,8))
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap=flatui)
plt.title('Correlation between different fearures')


# In[ ]:


from numpy import log


# In[ ]:


hdf.head(2)


# In[ ]:


hd=hdf.copy()
hd['AMT_INCOME_TOTAL_log']=log(hdf['AMT_INCOME_TOTAL'])





# In[ ]:


hd.head(2)


# In[ ]:


hd[['AMT_GOODS_PRICE',
  'AMT_INCOME_TOTAL_log',
 'AMT_ANNUITY',
  'AMT_CREDIT',
 'DAYDIFF']].hist(figsize=(16, 10), bins=50, xlabelsize=8, ylabelsize=8);


# In[ ]:


from sklearn.preprocessing import Normalizer
normalized_application = Normalizer().fit_transform(hd[['AMT_GOODS_PRICE',
  'AMT_INCOME_TOTAL_log',
 'AMT_ANNUITY',
  'AMT_CREDIT',
 'DAYDIFF']])
#print (normalized_application)
correlation=pd.DataFrame(normalized_application).corr()
correlation.columns=['AMT_GOODS_PRICE',
  'AMT_INCOME_TOTAL_log',
 'AMT_ANNUITY',
  'AMT_CREDIT',
 'DAYDIFF']
plt.figure(figsize=(8,8))
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap=flatui)
plt.title('Correlation between different fearures')


# In[ ]:


plt.scatter(hd['AMT_ANNUITY'],hd['AMT_CREDIT'])
plt.xlabel("amt annuity")
plt.ylabel("amt credit")


# In[ ]:


plt.scatter(hd['AMT_GOODS_PRICE'],hd['DAYDIFF'])
plt.xlabel("amt good")
plt.ylabel("amt credit")


# In[ ]:


hd.dtypes


# In[ ]:


hd.columns.tolist()


# In[ ]:


hd=hd[['SK_ID_CURR',
 'AMT_GOODS_PRICE',
 'DAYS_EMPLOYED',
 'TARGET',
 'AMT_CREDIT',
 'DAYS_BIRTH',
 'AMT_ANNUITY',
 'HOUR_APPR_PROCESS_START',
 'DAYDIFF',
 'AMT_INCOME_TOTAL_log']]

hd.to_csv("tt.csv",index=False)


# In[ ]:


hd.shape


# In[ ]:


hd.isna().sum()


# In[ ]:


ldf.isna().sum()


# In[ ]:


appobj=application[application_object_columns]


# In[ ]:


appobj.head(2)


# In[ ]:


appobj.isna().sum()


# In[ ]:





# In[ ]:




