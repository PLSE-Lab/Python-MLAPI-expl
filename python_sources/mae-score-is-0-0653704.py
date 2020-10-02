# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#code to perform opration on zilow data.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('../input/train_2016.csv')
prop = pd.read_csv('../input/properties_2016.csv')
test_data=pd.read_csv('../sample_submission.csv')

train_data=train_data.merge(prop,how='left',on='parcelid')

train_data=train_data.drop(['transactiondate'],axis=1)
test_data['parcelid']=test_data['ParcelId']
test_data=test_data.merge(prop,how='left',on='parcelid')
test_data=test_data.drop(['ParcelId','201610','201611','201612','201710','201711','201712'],axis=1)

#==========Cleaning_train_data.=============================================================

train_data=train_data.fillna(train_data.median())


train_data=train_data.drop(['propertycountylandusecode'],axis=1)
train_data.taxdelinquencyflag=train_data.taxdelinquencyflag.fillna('Y')
train_data.propertyzoningdesc=train_data.propertyzoningdesc.fillna('LAR1')
#==========Implementing_LabelEncoder===========================================================
encoder=LabelEncoder()
encoder.fit(train_data.propertyzoningdesc)
train_data.propertyzoningdesc=encoder.transform(train_data.propertyzoningdesc)

#==========correlation_filter_on_train_data====================================================
train_data=train_data.drop(['airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid','decktypeid','heatingorsystemtypeid','longitude','poolcnt','pooltypeid10','pooltypeid2'
,'pooltypeid7','regionidzip','threequarterbathnbr','fireplaceflag','fireplacecnt'],axis=1)
 
#=========Implementing_One_Hot_encoding(on_train_data)===========================================
vect=['taxdelinquencyflag']
for column in vect:
	temp=pd.get_dummies(pd.Series(train_data[column]))
	train_data=pd.concat([train_data,temp],axis=1)
	train_data=train_data.drop([column],axis=1)


#=========Cleaning test_data.=================================================================
test_data=test_data.fillna(test_data.median())
test_data=test_data.drop(['propertycountylandusecode'],axis=1)
test_data.taxdelinquencyflag=test_data.taxdelinquencyflag.fillna('Y')
test_data.propertyzoningdesc=test_data.propertyzoningdesc.fillna('LAR1')
#==========Implementing_LabelEncoder==========================================================
encoder=LabelEncoder()
encoder.fit(test_data.propertyzoningdesc)
test_data.propertyzoningdesc=encoder.transform(test_data.propertyzoningdesc)
#==========correlation_filter_on_test_data====================================================
test_data=test_data.drop(['airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid','decktypeid','heatingorsystemtypeid','longitude','poolcnt','pooltypeid10','pooltypeid2'
,'pooltypeid7','regionidzip','threequarterbathnbr','fireplaceflag','fireplacecnt'],axis=1)
 
#=========Implementing_One_Hot_encoding(on_test_data)=========================================
for column in vect:
	temp=pd.get_dummies(pd.Series(test_data[column]))
	test_data=pd.concat([test_data,temp],axis=1)
	test_data=test_data.drop([column],axis=1)

from sklearn.model_selection import train_test_split
y=train_data.logerror.values
X=train_data.drop('logerror',axis=1)
print X.shape
print y.shape



#print test_data.shape


#X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

from random import randrange

from catboost import CatBoostRegressor
#model=RandomForestRegressor(n_estimators=randrange(280,400),criterion='mse',max_depth=randrange(3,7),min_samples_split=randrange(3,8),max_features='log2', max_leaf_nodes=5,n_jobs=-1)
#model=GradientBoostingRegressor(learning_rate=0.1, n_estimators=randrange(500,800),criterion='friedman_mse', min_samples_split=randrange(3,6),min_samples_leaf=randrange(3,6),min_weight_fraction_leaf=0.3, max_depth=randrange(3,7),random_state=42, max_features='sqrt',max_leaf_nodes=2)
model = CatBoostRegressor(iterations=randrange(200,500), learning_rate=0.03,depth=randrange(3,6), l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE')
model.fit(X,y)
submit=model.predict(test_data)


#========creating_csv_file=====================================================================

sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = submit

print('Writing csv ...')
sub.to_csv('prediction_file.csv', index=False, float_format='%.4f') 



