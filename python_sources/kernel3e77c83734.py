#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_orig = pd.read_csv("train.csv")


# In[ ]:


data_origtest=pd.read_csv("test_1.csv")


# In[ ]:


data=data_orig
datatest=data_origtest


# In[ ]:


data=data.drop(['ID'],axis=1)
data=data.drop(['Schooling'],axis=1)
data=data.drop(['Married_Life'],axis=1)
data=data.drop(['MIC'],axis=1)
data=data.drop(['MOC'],axis=1)
data=data.drop(['Worker Class'],axis=1)
data=data.drop(['Enrolled'],axis=1)
data=data.drop(['Hispanic'],axis=1)
data=data.drop(['Reason'],axis=1)
data=data.drop(['Full/Part'],axis=1)
data=data.drop(['State'],axis=1)
data=data.drop(['Detailed'],axis=1)
data=data.drop(['MSA'],axis=1)
data=data.drop(['REG'],axis=1)
data=data.drop(['MOVE'],axis=1)
data=data.drop(['COB FATHER'],axis=1)
data=data.drop(['COB MOTHER'],axis=1)
data=data.drop(['COB SELF'],axis=1)
data=data.drop(['Teen'],axis=1)


# In[ ]:


datatest=datatest.drop(['ID'],axis=1)
datatest=datatest.drop(['Schooling'],axis=1)
datatest=datatest.drop(['Married_Life'],axis=1)
datatest=datatest.drop(['MIC'],axis=1)
datatest=datatest.drop(['MOC'],axis=1)
datatest=datatest.drop(['Worker Class'],axis=1)
datatest=datatest.drop(['Enrolled'],axis=1)
datatest=datatest.drop(['Hispanic'],axis=1)
datatest=datatest.drop(['Reason'],axis=1)
datatest=datatest.drop(['Full/Part'],axis=1)
datatest=datatest.drop(['State'],axis=1)
datatest=datatest.drop(['Detailed'],axis=1)
datatest=datatest.drop(['MSA'],axis=1)
datatest=datatest.drop(['REG'],axis=1)
datatest=datatest.drop(['MOVE'],axis=1)
datatest=datatest.drop(['COB FATHER'],axis=1)
datatest=datatest.drop(['COB MOTHER'],axis=1)
datatest=datatest.drop(['COB SELF'],axis=1)
datatest=datatest.drop(['Teen'],axis=1)


# In[ ]:


data=pd.get_dummies(data,columns=['Cast','Sex','Tax Status','Area','Summary','Citizen'])


# In[ ]:


datatest=pd.get_dummies(datatest,columns=['Cast','Sex','Tax Status','Area','Summary','Citizen'])


# In[ ]:


data['Fill'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)

datatest['Fill'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)


# In[ ]:


data['MLU'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)


# In[ ]:


datatest['MLU'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)


# In[ ]:


data['Live'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)

datatest['Live'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)


# In[ ]:


data['PREV'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)

datatest['PREV'].replace({
    'NO':0,
    'YES':1,
    '?':np.nan
},inplace=True)


# In[ ]:


data.info()


# In[ ]:


data=data.fillna(data.mean())
datatest=datatest.fillna(datatest.mean())


# In[ ]:


X_train=data.drop('Class',axis=1)
Y_train=data['Class']


# In[ ]:


#Performing Min_Max Normalization
min_max_scaler = preprocessing. MinMaxScaler()
np_scaledx = min_max_scaler. fit_transform(X_train)
X_train = pd. DataFrame(np_scaledx)


np_scaledtest = min_max_scaler. fit_transform(datatest)
datatest = pd. DataFrame(np_scaledtest)


# In[ ]:


X_test=datatest
Y_test=[0]*99523


# In[ ]:


data.info()


# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB
nb = NB()
nb.fit(X_train,Y_train)


# In[ ]:


nb.score(X_test,Y_test)


# In[ ]:


y_pred=nb.predict(X_test)
res1 = pd.DataFrame(y_pred)


# In[ ]:


B=data_origtest['ID']


# In[ ]:


final= pd.concat([B,res1],axis=1).reindex()
final = final. rename(columns={0: "Class"})
final. head()


# In[ ]:


final.info()


# In[ ]:


final.to_csv("submission1.csv",index=False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
 csv = df.to_csv(index=False)
 b64 = base64.b64encode(csv.encode())
 payload = b64.decode()
 html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
 html = html.format(payload=payload,title=title,filename=filename)
 return HTML(html)
create_download_link('submission1.csv')


# In[ ]:




