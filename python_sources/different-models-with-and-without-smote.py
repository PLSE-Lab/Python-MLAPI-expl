#!/usr/bin/env python
# coding: utf-8

# In[290]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[291]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV


# In[ ]:





# In[292]:


session_df= pd.read_csv('../input/session_related.csv')
#sessionData.head()


# In[293]:


delivery_df= pd.read_csv('../input/generic_outliers_data.csv')
#genericData.head()


# In[294]:


data_df = pd.read_csv('../input/delivery_related.csv')
#deliveryData.head()


# In[295]:


firstmerge = pd.merge(deliveryData,sessionData,on='OrderId')


# In[296]:


MergedDataset = pd.merge(firstmerge,genericData,on='CustId')
#MergedDataset.head()


# In[297]:


IndependentAttributes = pd.DataFrame()


# In[298]:


CustId = MergedDataset['CustId']
OrderId = MergedDataset['OrderId']
EmailId = MergedDataset['EmailId']
MobileNo = MergedDataset['MobileNo']
MacAddress = MergedDataset['MacAddress']
AvgPurchase = MergedDataset['AvgPurchase']
City = MergedDataset['City']
IsValidGeo = MergedDataset['IsValidGeo']
IsValidAddress = MergedDataset['IsValidAddress']
IsDeliveryRejected = MergedDataset['IsDeliveryRejected']
ReplacementDate = MergedDataset['ReplacementDate']
IsOneTimeUseProduct = MergedDataset['IsOneTimeUseProduct']
Session_Pincode = MergedDataset['Session_Pincode']
DeliveryDate = MergedDataset['DeliveryDate']
OrderDate = MergedDataset['OrderDate']
Fraud = MergedDataset['Fraud']


# In[299]:


IndependentAttributes['CustId'] = CustId
IndependentAttributes['OrderId'] = OrderId
IndependentAttributes['EmailId'] = EmailId
IndependentAttributes['MobileNo'] = MobileNo
IndependentAttributes['MacAddress'] = MacAddress
IndependentAttributes['Session_Pincode'] = Session_Pincode
IndependentAttributes['AvgPurchase'] = AvgPurchase
IndependentAttributes['City'] = City
IndependentAttributes['OrderDate'] = OrderDate
IndependentAttributes['DeliveryDate'] = DeliveryDate


# In[300]:


df1 = pd.DataFrame(IndependentAttributes['OrderDate'])
df1['DeliveryDate'] = IndependentAttributes['DeliveryDate']
df1['OrderDate'] = pd.to_datetime(df1['OrderDate'], format='%d/%m/%Y')
df1['DeliveryDate'] = pd.to_datetime(df1['DeliveryDate'], format='%d/%m/%Y')
DaysDifference = (df1['DeliveryDate'] - df1['OrderDate'])
df1.drop(['DeliveryDate', 'OrderDate'], axis='columns', inplace=True)

IndependentAttributes['DaysDifference'] = DaysDifference


# In[301]:


IndependentAttributes['ReplacementDate'] = ReplacementDate
IndependentAttributes['IsDeliveryRejected'] = IsDeliveryRejected
IndependentAttributes['IsOneTimeUseProduct'] = IsOneTimeUseProduct
IndependentAttributes['IsValidAddress'] = IsValidAddress
IndependentAttributes['IsValidGeo'] = IsValidGeo
IndependentAttributes['Fraud'] = Fraud


# In[302]:


IndependentAttributes.IsDeliveryRejected.replace(('yes', 'no'), (1, 0), inplace=True)
IndependentAttributes.IsOneTimeUseProduct.replace(('yes', 'no'), (1, 0), inplace=True)
IndependentAttributes.IsValidGeo.replace(('YES', 'NO'), (1, 0), inplace=True)
IndependentAttributes.IsValidAddress.replace(('yes', 'no'), (1, 0), inplace=True)
IndependentAttributes.Fraud.replace(('normal', 'suspicious', 'fraudulent'), (0, 1, 1), inplace=True)


# In[303]:


IndependentAttributes.MacAddress.replace(regex=r'-', value='', inplace=True)


# In[304]:


IndependentAttributes.DeliveryDate.replace(regex=r'/', value='-', inplace=True)
IndependentAttributes.OrderDate.replace(regex=r'/', value='-', inplace=True)
IndependentAttributes.ReplacementDate.replace(regex=r'/', value='-', inplace=True)


# In[305]:


IndependentAttributes['DaysDifference'] = IndependentAttributes.DaysDifference.dt.days
IndependentAttributes.head()


# In[306]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[307]:


df = pd.DataFrame() 
df = reduce_mem_usage(IndependentAttributes)


# In[308]:


Fraud_txn = df[df['Fraud']== 1]
normal_txn = df[df['Fraud']== 0]

# print("---------------------------")
# print("From the training dataset:")
# print("---------------------------")
# print("  Total Customers : %i"\
#       %(len(df)))
# print("")
# print("  Total Normal transactions  : %i"\
#       %(len(normal_txn)))

# print("  Normal transactions Rate   : %i %% "\
#      % (1.*len(normal_txn)/len(df)*100.0))
# print("-------------------------")

# print("  Fraudulent transactions         : %i"\
#       %(len(Fraud_txn)))

# print("  Fraudulent transactions Rate    : %i %% "\
#      % (1.*len(Fraud_txn)/len(df)*100.0))
# print("-------------------------")


# In[ ]:





# In[309]:


from sklearn.preprocessing import LabelEncoder
categorical = ['IsValidAddress','IsDeliveryRejected','IsOneTimeUseProduct','IsValidGeo','CustId','City']

label_encoder = LabelEncoder()
for col in categorical:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

df=df.iloc[:df.shape[0]]


# In[310]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
features = pd.DataFrame(df)
labels = pd.DataFrame(features['Fraud'])
features = features.drop(['Fraud','EmailId','MacAddress','City'], axis=1)
features = features.apply(LabelEncoder().fit_transform)
#features.head()


# In[311]:


#labels.head()


# In[312]:


# Scaling the Train and Test feature set 
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#features = scaler.fit_transform(features)
features.head()


# In[313]:


labels.head()


# In[314]:


from sklearn.model_selection import train_test_split
train, test, labels_train, labels_test = train_test_split(features, labels, test_size=0.60, random_state = 42)


# In[315]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, labels_train, test_size=0.20, random_state = 42)


# In[316]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
x_train_std_os,y_train_os = sm.fit_sample(X_train,y_train)


# In[ ]:





# In[ ]:





# In[317]:



from sklearn.ensemble.forest import RandomForestClassifier
rf = RandomForestClassifier(random_state=0).fit(x_train_std_os,y_train_os)
rf_pred = rf.predict(X_val)


# In[318]:


from sklearn import metrics


# In[319]:


###Just gave the output of few algo with smote which classifies both classes and without smote it doesnt classify both classes


# In[320]:


temp1 = metrics.f1_score(y_val,rf_pred,average=None)[1]
temp2 = metrics.f1_score(y_val,rf_pred,average=None)[0]

# F1Fraud= temp1*100
# F1Normal= temp2*100


F1Fraud = str(round(temp1, 2)*100)
F1Normal = str(round(temp2, 2)*100)


#print(F1Fraud,F1Normal,sep="\n")


# ###Just gave the output of few algo with smote which classifies both classes and without smote it doesnt classify both classes

# In[321]:


# import json
# data = {"F1 Fraud":F1Fraud,"F1 Normal":F1Normal, "model": "SVM"}
# jstr = json.dumps(data)

# with open('F1.json', 'w') as outfile: 
#     jstr
# import json
# with open('data.json', 'w') as outfile:
#     data = {"F1":metrics.f1_score(y_val,pred,average=None), "model": "SVM"}
#     json.dump(data, outfile)

# data = {"F1":metrics.f1_score(y_val,pred,average=None), "model": "SVM"}    
# import json
# with open('data.txt', 'w') as f:
#     json.dump(data, f, ensure_ascii=False)

import json

data = {
    'f1_fraud': F1Fraud,
    'f1_normal': F1Normal,
    'model': 'SVM'
}


with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)
    
with open("data_file.json", "r") as read_file:
    data = json.load(read_file)
    
print(data)


# In[322]:


features.head()


# In[323]:


# y_hats2 = model.predict(X)

# features['y_hats'] = y_hats2

rf_pred


# In[324]:


rf_pred


# In[325]:


res = pd.DataFrame(rf_pred)
results = res.rename(columns={0:'Fraud'})


# In[326]:


results.head()


# In[327]:


final_export = pd.DataFrame()


# In[328]:


OrderId = features['OrderId']
#OrderDate = features['OrderDate']
#City = features['City']
AvgPurchase = features['AvgPurchase']
PredictedLabel = results['Fraud']
#EmailId = features['EmailId']


# In[329]:


final_export['OrderId'] = OrderId 
final_export['CustId'] = CustId 
final_export['City'] = City 
final_export['PredictedLabel'] = PredictedLabel 
final_export['AvgPurchase'] = AvgPurchase
final_export['EmailId'] = EmailId
final_export['DeliveryDate']=DeliveryDate
final_export['OrderDate'] = OrderDate 


# In[330]:


final_export.head()


# In[ ]:





# In[336]:


export_csv = final_export.to_csv (r'abc.csv', index = None, header=True)


# In[ ]:




