import time
start = time.time()
import pandas as pd
import json
from geopy.distance import geodesic
import numpy as np
import glob

# import the datasets: 

path="../input/"
location=pd.read_csv(path+"Location.csv")
category=pd.read_csv(path+"Category.csv")
ItemInfo_test=pd.read_csv(path+"ItemInfo_test.csv")
ItemInfo_train=pd.read_csv(path+"ItemInfo_train.csv")
ItemPairs_test=pd.read_csv(path+"ItemPairs_test.csv")
ItemPairs_train=pd.read_csv(path+"ItemPairs_train.csv")
Random_submission=pd.read_csv(path+"Random_submission.csv")


# for test purpose, keep the following subdataset:
# This part is to be deleted in the final version
'''
ItemPairs_train=ItemPairs_train.head(100000)
unique_item=np.array(pd.concat([ItemPairs_train.itemID_1,ItemPairs_train.itemID_2]))
unique_item=np.unique(unique_item)
unique_item=pd.DataFrame(unique_item,columns=['itemID'])

ItemInfo=ItemInfo_train.merge(unique_item,on="itemID",how='inner')
'''

'''
-- define function to parse additional parameters of the ad
-- Expand the parameters from JSON to dummy variables taking valuee 0 or 1
'''
def CustomParser(data):
    if type(data) is not str:
        data='{}'
    j1 = json.loads(data)
    return j1

'''
-- Define function to compaire the parameters 
-- return 1 is the pair matches EXACTLY, 0 if not. Retuen NaN if one if the parameters=NaN
'''
# checked the similarity in product parameters (now is doing exact match.maybe do a partial match?)
def param_compare(data,param):
    item_1=param+'_1'
    item_2=param+'_2'
    if (type(data[item_1]) is float) or (type(data[item_2]) is float):
        return np.nan
    elif data[item_1]==data[item_2]:
        return 1
    else:
        return 0
    

'''
--Define function to merge datasets 
--Merge ItemPairs datasets with ItemInfo, location, and category
--looks like location ID is more specific than region ID, and
--category ID is more specific than parent ID. 
'''
def group_merge(base_data, merge_data,key_ID_1,key_ID_2):
    merge_temp=merge_data.rename(lambda x:x+'_1',axis='columns')
     # need to change how when running the full dataset
    combine_data=base_data.merge(merge_temp, on=key_ID_1,how='left') # need to change how when running the full dataset
     # need to change how when running the full dataset
    merge_temp=merge_data.rename(lambda x:x+'_2',axis='columns')
    combine_data=combine_data.merge(merge_temp,on=key_ID_2,how='left')
    
    return combine_data
'''
-Define function for distance calculation
'''
def dist(data):
    return geodesic((data.lat_1,data.lon_1), (data.lat_2,data.lon_2)).km
'''
-- Define function to count the number of words in images, title, and description
-- that appear in both ads
'''
def count_same(data,feature_1,feature_2,split_method):
    add_count_name=0
    if (type(data[feature_1]) is float) or (type(data[feature_2]) is float): 
        add_count_name=np.nan
        return add_count_name
    new_list=data[feature_1].split(split_method)
    for item in new_list:
        if item in data[feature_2]:
            add_count_name+=1
    return add_count_name

'''
-- Define function to compare if regionID, parentCategoryID,locationID, or metroID
-- are the same for both ads 
-- return values 0 or 1. Return NaN if one of the variables=NaN
'''
def pair_compare(data,pair):
    pair_1=pair+'_1'
    pair_2=pair+'_2'
    if np.isnan(data[pair_1]) or np.isnan(data[pair_2]):
        return np.nan
    elif data[pair_1]==data[pair_2]:
        return 1
    else:
        return 0
    
'''
-- Define function to capture additional variables
'''
def change_addition_var_name(data_before , data_after, _suffix):
    header_1=pd.DataFrame(list(data_before.columns),columns=['Intem_Info'])
    header_1['place_holder'] =pd.Series(np.random.randn(len(list(data_before.columns))), index=header_1.index)
    header_2=pd.DataFrame(list(data_after.columns),columns=['Intem_Info'])
    
    param=header_2.merge(header_1,how='outer')
    param=list(param[param.place_holder.isna()].Intem_Info.values)
    
    new_name_map=dict(((name,name+_suffix) for name in param))
    
    return new_name_map , param


'''
--define the aggregate cleaning function to clean the training and test datasets
--calls all the previous functions
'''

def data_clean(split_pair_dataset, ItemInfo_dataset, export_folder, export_file):
    

    
    count=0   
    
    for ItemPairs in split_pair_dataset:
        
        # merge in all item info
        Info_combine=group_merge(ItemPairs,ItemInfo_dataset,'itemID_1','itemID_2')
        
        '''
        This section is to clean up product attri
        '''
        ## parse product attri 
        for item in ['_1','_2']:
            Info_combine['parse_attrsJSON'+item]=Info_combine['attrsJSON'+item].apply(CustomParser)
            Info_combine_clean=pd.concat([Info_combine['parse_attrsJSON'+item].apply(pd.Series),Info_combine],axis=1)
            
            new_var=change_addition_var_name(Info_combine,Info_combine_clean,item)
            new_var_map=new_var[0]
            Info_combine_clean.rename(columns=new_var_map, inplace=True)
      
            Info_combine=Info_combine_clean
            
            Info_combine_clean.drop(['parse_attrsJSON'+item,'attrsJSON'+item],axis=1,inplace=True)
        
        # compare product attri and generate float param variables to reduce size
        for param in new_var[1]:
            new_col=param+'_compare'
            if (param+'_1' not in Info_combine_clean.columns) or (param+'_2' not in Info_combine_clean.columns):
                 Info_combine_clean[new_col]=np.nan
                 if param+'_1' in Info_combine_clean.columns:
                     Info_combine_clean.drop([param+'_1'],axis=1,inplace=True)
                 if param+'_2' in Info_combine_clean.columns:
                     Info_combine_clean.drop([param+'_2'],axis=1,inplace=True)
            else:
                Info_combine_clean[new_col]=Info_combine_clean.apply(param_compare,args=(param,),axis=1)
                Info_combine_clean.drop([param+'_1',param+'_2'],axis=1,inplace=True)
    
        '''
        This part is to find out similarities of 
        images, title, and description between the two products. 
        Count the number of same words appearing in each of the features
        '''
        Info_combine_clean['same_img_count']=Info_combine_clean.apply(count_same,args=('images_array_1','images_array_2',','),axis=1)
        Info_combine_clean['same_title_count']=Info_combine_clean.apply(count_same,args=('title_1','title_2',' '),axis=1)
        Info_combine_clean['same_descri_count']=Info_combine_clean.apply(count_same,args=('description_1','description_2',' '),axis=1)
        
        Info_combine_clean.drop(['images_array_1','images_array_2','title_1','title_2','description_1','description_2'],axis=1,inplace=True)
        
        
        '''
        This part if to calculate distanceI(in km) and price difference for each pair
        '''
        Info_combine_clean['price_diff']=abs(np.log(Info_combine_clean.price_1)-np.log(Info_combine_clean.price_2))
        Info_combine_clean['dist']=Info_combine_clean.apply(dist,axis=1)
        
        Info_combine_clean.drop(['price_1','price_2','lat_1','lat_2','lon_1','lon_2'],axis=1,inplace=True)
        
        
        '''
        This part is to compare regionID, parentCategoryID,locationID, metroID for each pair
        '''
        Info_combine_clean=group_merge(Info_combine_clean,location,'locationID_1','locationID_2')
        Info_combine_clean=group_merge(Info_combine_clean,category,'categoryID_1','categoryID_2')
    
        for pair in ['regionID','parentCategoryID','locationID' , 'metroID','categoryID']:
            new_column=pair+'_pair'
            Info_combine_clean[new_column]=Info_combine_clean.apply(pair_compare,args=(pair,),axis=1)
            
        Info_combine_clean.drop(['regionID_1','regionID_2','parentCategoryID_1','parentCategoryID_2','locationID_1','locationID_2',
                                 'metroID_1','metroID_2','categoryID_1','categoryID_2'],axis=1,inplace=True)
        
        Info_combine_clean.to_csv(export_folder + export_file+str(count)+".csv" )
        

        count+=1
        

'''
-- Clean up the train and test datasets
-- To boost speas, split the datasets into 100 chunks and run seperately
'''        

# divide the data into 100 smaller sizes to speed up the program
ItemPairs_train.drop(['generationMethod'],axis=1,inplace=True)
ItemPairs_train_split=np.array_split(ItemPairs_train,100)
ItemPairs_test_split=np.array_split(ItemPairs_test,100)

## For train dataset
data_clean(ItemPairs_train_split, ItemInfo_train, "", "train split")
## put together all the data for modelling
files=glob.glob('train split*.csv')
agg_clean_train_data = pd.read_csv(files[0])
for f in files[1::]:
    agg_clean_train_data=pd.read_csv(f).append(agg_clean_train_data,ignore_index=True)


## For test dataset
data_clean(ItemPairs_test_split, ItemInfo_test,"", "test split")
files=glob.glob('test split*.csv')
agg_clean_test_data = pd.read_csv(files[0])
for f in files[1::]:
    agg_clean_test_data=pd.read_csv(f).append(agg_clean_test_data,ignore_index=True)

## Keep only the features that appear on both test and training datasets 
header_train=pd.DataFrame(list(agg_clean_train_data.columns),columns=['variables'])
header_test=pd.DataFrame(list(agg_clean_test_data.columns),columns=['variables'])
variables=(header_train.merge(header_test,how='inner'))

variables=variables[~variables.variables.str.contains('_1|_2')]

# predict model:
data=agg_clean_train_data[list(variables.values.flatten())+['itemID_1','itemID_2','isDuplicate']]
submit=agg_clean_test_data[list(variables.values.flatten())+['itemID_1','itemID_2']]

train=data.sample(frac=0.8)
test=data.loc[~data.index.isin(train.index)]

X_train=train.drop(['isDuplicate','Unnamed: 0','itemID_1','itemID_2'],axis=1)
y_train=train['isDuplicate']

X_test=test.drop(['isDuplicate','Unnamed: 0','itemID_1','itemID_2'],axis=1)
y_test=test['isDuplicate']
'''
xgboost
'''
dtrain = xgb.DMatrix(X_train, label =y_train)
dtest=xgb.DMatrix(X_test, label =y_test)
evallist  = [(dtest,'eval'), (dtrain,'train')]

param={'booster':'gbtree','eta' : 0.03, 'objective': 'binary:logistic','max_depth': 10,
       'tree_method':'exact','eval_metric':'auc','colsample_bytree':0.7}
num_round=300
bst = xgb.train(param,dtrain,num_round,evallist,early_stopping_rounds=10)
	
bst.get_score(importance_type='gain')
bst.best_iteration
bst.best_ntree_limit
#
X_test=submit.drop(['Unnamed: 0','itemID_1','itemID_2'],axis=1)
dX_test=xgb.DMatrix(X_test)
y_test=bst.predict(dX_test,ntree_limit=bst.best_ntree_limit)

pred_submit=pd.DataFrame(y_test,columns=['probability'])
pred_submit.to_csv('pred_submit.csv')

end = time.time()
elapsed = end - start
print(time.strftime("%H:%M:%S", time.gmtime(elapsed)))  # '04:15' runtime for train dataset 
