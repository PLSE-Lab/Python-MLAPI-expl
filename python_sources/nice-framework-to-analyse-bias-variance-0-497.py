"""
Contributes from 
https://www.kaggle.com/the1owl/surprise-me
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493) 
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st

Learned a lot from those creative kernels

Created on Fri Dec 15 17:14:07 2017

@author: Reno_Lei
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb


def split_train_dev_test(submission,dev_date,test_date,visit_data):
    '''split the set to be train,dev,test,and submission, the submission is from
        sample sub file, so it is a little bit different to the train set. Here I
        split the test_data similar to the submission data, and dev data similar to the
        train data
    '''
    test_date = pd.to_datetime(test_date)
    dev_date = pd.to_datetime(dev_date)
    
    sub = submission['id'].str.rsplit('_', expand=True, n=1).rename(columns = {0:'air_store_id',1:'visit_date'})
    
    sub_start_date = sub['visit_date'].min()
    
    test_date_date_range = pd.DataFrame(pd.date_range(test_date,sub_start_date))
    
    sub_store_id = pd.DataFrame(pd.unique(sub['air_store_id']))
        
    test = pd.DataFrame.from_dict({(i,j): [i,j]
                           for i in sub_store_id[0]
                           for j in test_date_date_range[0]},
                       orient='index').rename(columns = {0:'air_store_id',1:'visit_date'})
    
    test['visit_date'] = test['visit_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    test.index = range(len(test))
    visit_date_remove_visitors = visit_data[['air_store_id','visit_date']]
    train = visit_date_remove_visitors[pd.to_datetime(visit_date_remove_visitors['visit_date']) < dev_date]
    dev = visit_date_remove_visitors[pd.to_datetime(visit_date_remove_visitors['visit_date']) >= dev_date]
    dev = dev[pd.to_datetime(visit_date_remove_visitors['visit_date']) < test_date]
    dev.index = range(len(dev))
    print('*****************The split_train_dev_test finished**************')
    print('The visit\'s shape is : ' + str(visit_data.shape))
    print('The train\'s shape is : ' + str(train.shape))
    print('The dev\'s shape is : ' + str(dev.shape))
    print('The test\'s shape is : ' + str(test.shape))
    print('The sub\'s shape is : ' + str(sub.shape))
    
    return train,dev,test,sub

def merge_store_data(train,dev,test,sub,store_info,holiday_info,visit_data):
    
    train = pd.merge(store_info,train,on='air_store_id')
    dev = pd.merge(store_info,dev,on='air_store_id')
    test = pd.merge(store_info,test,on='air_store_id')
    sub = pd.merge(store_info,sub,on='air_store_id')

    train = pd.merge(visit_data,train,on=['air_store_id','visit_date'])
    dev = pd.merge(visit_data,dev,on=['air_store_id','visit_date'])
    test = pd.merge(visit_data,test,on=['air_store_id','visit_date'])
    #sub = pd.merge(visit_data,sub,on=['air_store_id','visit_date'])

    train = pd.merge(holiday_info,train,right_on = 'visit_date',left_on='calendar_date')
    dev = pd.merge(holiday_info,dev,right_on = 'visit_date',left_on='calendar_date')
    test = pd.merge(holiday_info,test,right_on = 'visit_date',left_on='calendar_date')
    sub = pd.merge(holiday_info,sub,right_on = 'visit_date',left_on='calendar_date')
    
    print('*****************The merge_store_data finished**************')
    print('The train\'s shape is : ' + str(train.shape))
    print('The dev\'s shape is : ' + str(dev.shape))
    print('The test\'s shape is : ' + str(test.shape))
    print('The sub\'s shape is : ' + str(sub.shape))
    return train,dev,test,sub

def merge_reserve_data(train,dev,test,sub,air_reserve,hpg_reserve,store_id_relation,visit_data):
    
    data = {'ar':air_reserve,'hr':hpg_reserve}
    data['hr'] = pd.merge(data['hr'], store_id_relation, how='inner', on=['hpg_store_id'])
    
    for df in ['ar','hr']:
        data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
        data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
        data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
        data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
        data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
        tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
        tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
        data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
    
    train['visit_date'] = pd.to_datetime(train['visit_date']).dt.date
    dev['visit_date'] = pd.to_datetime(dev['visit_date']).dt.date
    test['visit_date'] = pd.to_datetime(test['visit_date']).dt.date
    sub['visit_date'] = pd.to_datetime(sub['visit_date']).dt.date
    
    for df in ['ar','hr']:
        train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
        dev = pd.merge(dev, data[df], how='left', on=['air_store_id','visit_date'])
        test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
        sub = pd.merge(sub, data[df], how='left', on=['air_store_id','visit_date'])

    print('*****************The merge_reserve_data finished**************')
    print('The train\'s shape is : ' + str(train.shape))
    print('The dev\'s shape is : ' + str(dev.shape))
    print('The test\'s shape is : ' + str(test.shape))
    print('The sub\'s shape is : ' + str(sub.shape))
    return train,dev,test,sub

def merge_add_features(train,dev,test,sub,air_store_info):
    train['visit_date'] = pd.to_datetime(train['visit_date'])
    train['dow'] = train['visit_date'].dt.dayofweek
    train['year'] = train['visit_date'].dt.year
    train['month'] = train['visit_date'].dt.month
    train['visit_date'] = train['visit_date'].dt.date
    
    dev['visit_date'] = pd.to_datetime(dev['visit_date'])
    dev['dow'] = dev['visit_date'].dt.dayofweek
    dev['year'] = dev['visit_date'].dt.year
    dev['month'] = dev['visit_date'].dt.month
    dev['visit_date'] = dev['visit_date'].dt.date

    test['visit_date'] = pd.to_datetime(test['visit_date'])
    test['dow'] = test['visit_date'].dt.dayofweek
    test['year'] = test['visit_date'].dt.year
    test['month'] = test['visit_date'].dt.month
    test['visit_date'] = test['visit_date'].dt.date

    sub['visit_date'] = pd.to_datetime(sub['visit_date'])
    sub['dow'] = sub['visit_date'].dt.dayofweek
    sub['year'] = sub['visit_date'].dt.year
    sub['month'] = sub['visit_date'].dt.month
    sub['visit_date'] = sub['visit_date'].dt.date

    unique_stores = sub['air_store_id'].unique()
    stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)
    #OPTIMIZED BY JEROME VALLET
    tmp = train.groupby(['air_store_id','dow']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()
    tmp.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors','max_visitors','count_observations']
    stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
    
    stores = pd.merge(stores, air_store_info, how='left', on=['air_store_id']) 
    # NEW FEATURES FROM Georgii Vyshnia
    stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
    stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))
    lbl = LabelEncoder()
    for i in range(10):
        stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
        stores['air_area_name'+str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
    stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])
    

    train=train.drop(['calendar_date','day_of_week','air_genre_name','air_area_name','latitude','longitude'], axis=1)
    dev=dev.drop(['calendar_date','day_of_week','air_genre_name','air_area_name','latitude','longitude'], axis=1)
    test=test.drop(['calendar_date','day_of_week','air_genre_name','air_area_name','latitude','longitude'], axis=1)
    sub=sub.drop(['calendar_date','day_of_week','air_genre_name','air_area_name','latitude','longitude'], axis=1)
    
    
    train = pd.merge(train, stores, how='left', on=['air_store_id','dow'])
    dev = pd.merge(dev, stores, how='left', on=['air_store_id','dow'])
    test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])
    sub = pd.merge(sub, stores, how='left', on=['air_store_id','dow'])
    
    #train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)
    sub['id'] = sub.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)
   
    for X in [train,dev,test,sub]:
        X['total_reserv_sum'] = X['rv1_x'] + X['rv1_y']
        X['total_reserv_mean'] = (X['rv2_x'] + X['rv2_y']) / 2
        X['total_reserv_dt_diff_mean'] = (X['rs2_x'] + X['rs2_y']) / 2
        
        # NEW FEATURES FROM JMBULL
        X['date_int'] = X['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
        X['var_max_lat'] = X['latitude'].max() - X['latitude']
        X['var_max_long'] = X['longitude'].max() - X['longitude']
        
        # NEW FEATURES FROM Georgii Vyshnia
        X['lon_plus_lat'] = X['longitude'] + X['latitude'] 
    lbl = LabelEncoder()
    train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])
    test['air_store_id2'] = lbl.transform(test['air_store_id'])
    dev['air_store_id2'] = lbl.transform(dev['air_store_id'])
    sub['air_store_id2'] = lbl.transform(sub['air_store_id'])
    
    print('*****************The merge_add_features finished**************')
    print('The train\'s shape is : ' + str(train.shape))
    print('The dev\'s shape is : ' + str(dev.shape))
    print('The test\'s shape is : ' + str(test.shape))
    print('The sub\'s shape is : ' + str(sub.shape))
    return train,dev,test,sub

def RMSLE(h, y):
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)))

def rmsle(h, y): 
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


def train_model(train,dev,test,sub,train_all_data = False):
    print('*****************The null values for each sets**************')
    print('The train\'s shape is : ' + str(train.isnull().sum()))
    print('The dev\'s shape is : ' + str(dev.isnull().sum()))
    print('The test\'s shape is : ' + str(test.isnull().sum()))
    print('The sub\'s shape is : ' + str(sub.isnull().sum()))
    
    col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
    train = train.fillna(-1)
    dev = dev.fillna(-1)
    test = test.fillna(-1)
    sub = sub.fillna(-1)
    
    if train_all_data == True:
        train = pd.concat([train,dev,test])

    model1 = KNeighborsRegressor(n_jobs=-1, n_neighbors=6)
    
    #model2 = GradientBoostingRegressor(learning_rate=0.2, random_state=3, n_estimators=100, subsample=0.8,   max_depth =10)
       
    modelgbm = lgb.LGBMRegressor(objective='regression_l1',
                        num_leaves=50,
                        learning_rate=0.02,
                        boosting = 'rf',
                        n_estimators=20)
    
    models = [modelgbm]
    
    for model in models:
        model.fit(train[col], np.log1p(train['visitors'].values))
        model_name = str(model).split('(')[0]
        print(model)
        train['predict_'+str(model_name)] = np.expm1(model.predict(train[col]))
        dev['predict_'+str(model_name)] = np.expm1(model.predict(dev[col]))
        test['predict_'+str(model_name)] = np.expm1(model.predict(test[col]))
        sub['predict_'+str(model_name)] = np.expm1(model.predict(sub[col]))
    
    
        print(str(model_name)+' train set: ', rmsle(train['predict_'+model_name].values, train['visitors'].values))
        print(str(model_name)+' dev set: ', rmsle(dev['predict_'+str(model_name)], dev['visitors'].values))
        print(str(model_name)+' tes set: ', rmsle(test['predict_'+str(model_name)], test['visitors'].values))

    for model in models:    
        sub['visitors'] = sub['predict_'+str(model_name)]
    sub['visitors'] = np.expm1(sub['visitors']/len(models)).clip(lower=0.)
    
    sub1 = sub[['id','visitors']].copy()
    #sub1.to_csv('output/submission_new.csv', index=False)
    
    
    return train,dev,test,sub
    

def analysis_error(train,dev,test,sub):
    model_colums = [x for x in train.columns if 'predict_' in x]
    for group in [train,dev,test]:
        for model_predict in model_colums:
            group['error'+model_predict] = RMSLE(group[model_predict],group['visitors'])
            for column in ['dow','air_store_id']:
                error_by_date_train = group.groupby(column,as_index = False)['error'+model_predict].mean()
                sns.barplot(error_by_date_train[column],error_by_date_train['error'+model_predict])
                plt.ylabel(model_predict)
                plt.xlabel(column)
                plt.title(model_predict)
                plt.show()
                #plt.savefig(group+model_predict+'.png')

def produce_submission_file(sub,sumbmissoin_info):
    sub['predict_visitors'] = (sub['predict_visitors_model1'] + sub['predict_visitors_model2'])/2
    sub1 = sub[['id','predict_visitors']].copy()
    sub1.to_csv('output/submission.csv', index=False)
    

data = {
    'vis': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'sub': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv')
    }


if __name__ == '__main__':
    
    train,dev,test,sub = split_train_dev_test(data['sub'],'2017-03-14','2017-04-14',data['vis'])
    
    train,dev,test,sub = merge_store_data(train,dev,test,sub,data['as'],data['hol'],data['vis'])
    
    train,dev,test,sub = merge_reserve_data(train,dev,test,sub,data['ar'],data['hr'],data['id'],data['vis'])

    train,dev,test,sub = merge_add_features(train,dev,test,sub,data['as'])

    train,dev,test,sub = train_model(train,dev,test,sub,train_all_data = False)
#
    analysis_error(train,dev,test,sub)
#    
    #reno.produce_submission_file(sub,data['sub'])