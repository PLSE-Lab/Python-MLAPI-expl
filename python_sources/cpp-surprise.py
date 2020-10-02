#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import numpy as np
pd.set_option('display.max_columns', None)
import itertools
from surprise import Reader, Dataset
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering

from zipfile import ZipFile
              
import time
import gc


# # Load Dataset

# In[ ]:


ds_dir = '../input/coupon-purchase-prediction'


# In[ ]:


#unzip dataset
with ZipFile(os.path.join(ds_dir,"coupon_detail_train.csv.zip"), 'r') as zipObj:
   zipObj.extractall()
with ZipFile(os.path.join(ds_dir,"coupon_list_test.csv.zip"), 'r') as zipObj:
   zipObj.extractall()
with ZipFile(os.path.join(ds_dir,"coupon_list_train.csv.zip"), 'r') as zipObj:
   zipObj.extractall()
with ZipFile(os.path.join(ds_dir,"coupon_visit_train.csv.zip"), 'r') as zipObj:
   zipObj.extractall()
with ZipFile(os.path.join(ds_dir,"sample_submission.csv.zip"), 'r') as zipObj:
   zipObj.extractall()
with ZipFile(os.path.join(ds_dir,"user_list.csv.zip"), 'r') as zipObj:
   zipObj.extractall()


# In[ ]:


#Dataset
cd_train = pd.read_csv('coupon_detail_train.csv')
cl_test = pd.read_csv('coupon_list_test.csv')
cl_train = pd.read_csv('coupon_list_train.csv')
#cv_train = pd.read_csv('coupon_visit_train.csv')
#pref_loc = pd.read_csv(os.path.join(ds_dir,'prefecture_locations.csv'))
sample_sub = pd.read_csv('sample_submission.csv')
user_list = pd.read_csv('user_list.csv')


# #Translator
# pref = pd.read_csv(os.path.join(dsdir,'pref.csv'),delimiter=';',index_col='jpn')
# pref_office = pd.read_csv(os.path.join(dsdir,'pref_office.csv'),delimiter=';',index_col='jpn')
# small_area_name = pd.read_csv(os.path.join(dsdir,'small_area_name.csv'),delimiter=';',index_col='jpn')
# big_area_name = pd.read_csv(os.path.join(dsdir,'big_area_name.csv'),delimiter=';',index_col='jpn')
# capsule_text = pd.read_csv(os.path.join(dsdir,'capsule_text.csv'),delimiter=';',index_col='jpn')
# genre_name = pd.read_csv(os.path.join(dsdir,'genre.csv'),delimiter=';',index_col='jpn')

# ## Translate JPN TO EN

# #CAPSULE TEXT
# cl_test.CAPSULE_TEXT = cl_test.CAPSULE_TEXT.replace(capsule_text.to_dict()['en'])
# cl_train.CAPSULE_TEXT = cl_train.CAPSULE_TEXT.replace(capsule_text.to_dict()['en'])
# 
# #GENRE NAME
# cl_test.GENRE_NAME = cl_test.GENRE_NAME.replace(genre_name.to_dict()['en'])
# cl_train.GENRE_NAME = cl_train.GENRE_NAME.replace(genre_name.to_dict()['en'])
# 
# #PREF NAME
# cl_test.ken_name = cl_test.ken_name.replace(pref.to_dict()['en'])
# cl_train.ken_name = cl_train.ken_name.replace(pref.to_dict()['en'])
# pref_loc.PREF_NAME = pref_loc.PREF_NAME.replace(pref.to_dict()['en'])
# user_list.PREF_NAME = user_list.PREF_NAME.replace(pref.to_dict()['en'])
# 
# #PREFECTUAL_OFFICE
# pref_loc.PREFECTUAL_OFFICE = pref_loc.PREFECTUAL_OFFICE.replace(pref_office.to_dict()['en'])
# 
# #SMALL_AREA_NAME
# cd_train.SMALL_AREA_NAME = cd_train.SMALL_AREA_NAME.replace(small_area_name.to_dict()['en'])
# cl_test.small_area_name = cl_test.small_area_name.replace(small_area_name.to_dict()['en'])
# cl_train.small_area_name = cl_train.small_area_name.replace(small_area_name.to_dict()['en'])
# 
# #large_area_name
# cl_test.large_area_name = cl_test.large_area_name.replace(big_area_name.to_dict()['en'])
# cl_train.large_area_name = cl_train.large_area_name.replace(big_area_name.to_dict()['en'])

# # Preprocessing and Convert Data to Surprise Data

# In[ ]:


cd_train = cd_train[['PURCHASEID_hash','USER_ID_hash','COUPON_ID_hash']]
cd_train = pd.merge(cd_train,cd_train.groupby(['USER_ID_hash', 'COUPON_ID_hash']).size().reset_index(name="PURCHASE_COUNT"),left_on=['USER_ID_hash', 'COUPON_ID_hash'],right_on=['USER_ID_hash', 'COUPON_ID_hash'],how='left')
cd_train.drop('PURCHASEID_hash',axis=1,inplace=True)
cd_train['PURCHASE_COUNT'] = np.log(cd_train['PURCHASE_COUNT']+1)


# In[ ]:


user_list = user_list[['USER_ID_hash']]


# In[ ]:


cl_train = cl_train[['COUPON_ID_hash']]
cl_test = cl_test[['COUPON_ID_hash']]


# In[ ]:


#Permutation of User-CouponTest
clist = cl_test.COUPON_ID_hash.unique().tolist()
ulist = user_list.USER_ID_hash.unique().tolist()

relations = [r for r in itertools.product(ulist, clist)]
relations = pd.DataFrame(relations,columns=['USER_ID_hash', 'COUPON_ID_hash'])


# In[ ]:


reader = Reader(rating_scale=(cd_train['PURCHASE_COUNT'].min()-0.5,cd_train['PURCHASE_COUNT'].max()+0.5))
data = Dataset.load_from_df(cd_train, reader)
trainset = data.build_full_trainset()


# In[ ]:


def predict(row):
    return model.predict(row.USER_ID_hash,row.COUPON_ID_hash).est

def clean_prediction(row):
    data = row.PURCHASED_COUPONS
    data = str("".join(str(data))[2:-2].replace("', '"," "))
    return data


# # Models

# In[ ]:


surprises = {
    'SVD' : SVD(verbose=1,random_state=0),
    'SVDpp' : SVDpp(verbose=1,random_state=0),
    'SlopeOne' : SlopeOne(),
    'NMF' : NMF(verbose=1,random_state=0),
    'NormalPredictor' : NormalPredictor(),
    'BaselineOnly_ALS' : BaselineOnly(verbose=1,bsl_options={'method':'als'}),
    'BaselineOnly_SGD' : BaselineOnly(verbose=1,bsl_options={'method':'sgd'}),
    'CoClustering' : CoClustering(verbose=1,random_state=0),
    #'KNNBaseline_ALS' : KNNBaseline(verbose=1,bsl_options={'method':'als'}),
    #'KNNBaseline_SGD' : KNNBaseline(verbose=1,bsl_options={'method':'sgd'}),
    #'KNNBasic' : KNNBasic(verbose=1),
    #'KNNWithMeans' : KNNWithMeans(verbose=1),
    #'KNNWithZScore' : KNNWithZScore(verbose=1),
}


# # Fit Predict Submission

# In[ ]:


for key, item in surprises.items():
    print(key)
    model_name = key
    start = time.time()
    model = item.fit(trainset)
    print(model_name,'Fit Time :',time.time()-start)

    start = time.time()
    submission = relations.copy()
    submission['PURCHASE_COUNT'] = submission.apply(predict, axis=1)
    print(model_name,'Predict Time :',time.time()-start)

    submission.sort_values('PURCHASE_COUNT', ascending=False, inplace=True)
    submission = submission.groupby(['USER_ID_hash']).head(10).reset_index()
    submission = submission.groupby('USER_ID_hash')['COUPON_ID_hash'].apply(list).reset_index(name='PURCHASED_COUPONS')
    submission['PURCHASED_COUPONS'] = submission.apply(clean_prediction, axis=1)
    submission.to_csv('cpp_surprise_'+model_name+'.csv', index=False)
    print(model_name,'Done')
    
    del submission
    del model
    gc.collect()


# In[ ]:




