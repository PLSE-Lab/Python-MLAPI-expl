# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_file = '../input/train.json'
test_file = '../input/test.json'



train = pd.read_json(train_file)
test = pd.read_json(test_file)

def clean_data(data,train_data = True):
    #extract useful numeric feature from string, and add new possibly useful feature
    #created date: year, month, day
    #description: length of description
    #features: length of features
    #photo: number of photos
    #address: length of display length
    #manager: number of entries per manager, average low rate
    #building: number of entries per building, average low rate
    
    data['created'] = pd.to_datetime(data['created'])
        
    #1 Numerical
    #get year, month, day, length of description, num of features, num of photos
    #length of address (use display address, which is shown)
    data['created_year']=np.zeros(len(data))
    data['created_month']=np.zeros(len(data))
    data['created_day']=np.zeros(len(data))
    data['desc_len']=np.zeros(len(data))
    data['feat_num']=np.zeros(len(data))
    data['photo_num']=np.zeros(len(data))
    data['add_len']=np.zeros(len(data))
    data['manager_count']=np.zeros(len(data))
    data['building_count'] = np.zeros(len(data))
    
    m_count = data['manager_id'].value_counts()
    b_count = data['building_id'].value_counts()    
    
    #partition by quantile to 9 grids: Up,up-left, up-right, center..., low
    lat_partition = data['latitude'].quantile([0,1/3,2/3,1])
    lon_partition = data['longitude'].quantile([0,1/3,2/3,1])
    loc_dict={'up-l':0,'up':1,'up-r':2,'c-l':3,'c':4,'c-r':5,'low-l':6,'low':7,'low-r':8}
    #loc=['low-l','low','low-r','c-l','c','c-r','up-l','up','up-r']
    loc_grid=np.array([['up-r','up','up-l'],['c-r','c','c-l'],['low-r','low','low-l']])
    data['loc']=np.array(['']*len(data))
    data['loc_index']=np.zeros(len(data))
    
    for index in data.index:
        data.set_value(index,'created_year',data['created'][index].year) #actually all are 2016, so no use
        data.set_value(index,'created_month',data['created'][index].month) #only 4,5,6
        data.set_value(index,'created_day',data['created'][index].day)
        data.set_value(index,'desc_len',len(str.split(data['description'][index],' ')))
        data.set_value(index,'feat_num',len(data['features'][index]))
        data.set_value(index,'photo_num',len(data['photos'][index]))
        data.set_value(index,'add_len',len(str.split(data['display_address'][index],' ')))
        # location indicator
        loca = loc_grid[len(lon_partition[(lon_partition>data['longitude'][index]).tolist()])-1,
                  len(lat_partition[(lat_partition>data['latitude'][index]).tolist()])-1]
        data.set_value(index,'loc',loca)
        data.set_value(index,'loc_index',loc_dict[loca])
        
        m_id = data.ix[index]['manager_id']
        b_id = data.ix[index]['building_id']
        data.set_value(index,'manager_count',m_count[m_id])
        data.set_value(index,'building_count',b_count[b_id])
    
    if train_data: #training set
        m_int_count = pd.groupby(data[['interest_level','manager_id','price']],['manager_id','interest_level']).count()
        b_int_count = pd.groupby(data[['interest_level','building_id','price']],['building_id','interest_level']).count()
        
        data['manager_own_rate']=np.zeros(len(data))
        data['manager_ave_rate']=np.zeros(len(data))
        
        data['building_own_rate']=np.zeros(len(data))
        data['building_ave_rate']=np.zeros(len(data))
        
        # own low rate
        for index in data.index:
            m_id = data.ix[index]['manager_id']
            b_id = data.ix[index]['building_id']
            try:
                data.set_value(index,'manager_own_rate',
                                (m_int_count.ix[m_id,'low'].values/m_count.ix[m_id])[0])
                data.set_value(index,'building_own_rate',
                                (b_int_count.ix[b_id,'low'].values/b_count.ix[b_id])[0])
            except:
                data.set_value(index,'manager_own_rate',0) #no low
        
        # average low rate(for managers with same entry counts, useful for low counts)
        for index in data.index:
            ma_count = data.ix[index]['manager_count']
            bu_count = data.ix[index]['building_count']
            data.set_value(index,'manager_ave_rate',data[data['manager_count']==ma_count]['manager_own_rate'].mean())
            data.set_value(index,'building_ave_rate',data[data['building_count']==bu_count]['building_own_rate'].mean())

    else: #testing set
        data['manager_ave_rate']=np.zeros(len(data))
        data['building_ave_rate']=np.zeros(len(data))
        
        train_m_count_rate = train[['manager_count','manager_ave_rate']].groupby('manager_count').mean() # each count's ave rate
        train_b_count_rate = train[['building_count','building_ave_rate']].groupby('building_count').mean() # each count's ave rate
        
        m_ave_dict = dict()
        b_ave_dict = dict()
        for index in data.index:
            m_id = data.ix[index]['manager_id']
            b_id = data.ix[index]['building_id']
        
            if m_id in m_ave_dict.keys():
                data.set_value(index,'manager_ave_rate',m_ave_dict[m_id])
            else:
                if m_id in train['manager_id']:
                    m_ave_dict[m_id]=train[train['manager_id']==m_id]['manager_ave_rate'].iloc[0]
                else: #m_id not in dataing set, check m_count
                    if data.ix[index]['manager_count'] in train_m_count_rate.index:
                        m_ave_dict[m_id] = train_m_count_rate.ix[data.ix[index]['manager_count']][0]
                    else:
                        m_ave_dict[m_id] = 0.694683
                data.set_value(index,'manager_ave_rate',m_ave_dict[m_id])
    
            if b_id in b_ave_dict.keys():
                data.set_value(index,'building_ave_rate',b_ave_dict[b_id])
            else:
                if b_id in data['building_id']:
                    b_ave_dict[b_id]=train[train['building_id']==b_id]['building_ave_rate'].iloc[0]
                else: #m_id not in dataing set, check m_count
                    if data.ix[index]['building_count'] in train_b_count_rate.index:
                        b_ave_dict[b_id] = train_b_count_rate.ix[data.ix[index]['building_count']][0]
                    else:
                        b_ave_dict[b_id] = 0.694683
        
                data.set_value(index,'building_ave_rate',b_ave_dict[b_id])
    return data
    
train = clean_data(train,True)
print('data cleaned: ', train.columns)

features_used = ['bathrooms', 'bedrooms', 'price', 
                 'created_month', 'created_day','desc_len', 
                 'feat_num', 'photo_num', 'add_len','loc_index',
                 'manager_count','building_count','manager_ave_rate',
                 'building_ave_rate']
                 
index = list(train.index)

np.random.shuffle(index)
X =train[features_used].ix[index[:40000]]
X_test = train[features_used].ix[index[40000:]]

y = np.array(train['interest_level'].ix[index[:40000]].tolist())
y = np.where(y=='low',np.zeros(len(y)),
         np.where(y=='medium',np.linspace(1,1,len(y)),np.linspace(2,2,len(y))))
y_test = np.array(train['interest_level'].ix[index[40000:]].tolist())
y_test = np.where(y_test=='low',np.zeros(len(y_test)),
         np.where(y_test=='medium',np.linspace(1,1,len(y_test)),np.linspace(2,2,len(y_test))))

dtrain = xgb.DMatrix(data=X,label=y)
dtest = xgb.DMatrix(data=X_test)

param={
    'max_depth' : 4,
    'min_child_weight':1,
    'gamma':0,
    'subsample':0.8,
    'colsample_bytree' : 0.8,
    'objective':'multi:softprob',
    'eval_metric' : 'mlogloss',
    'scale_pos_weight':1,
    'num_class':3
}

evallist  = [(dtest,'eval'), (dtrain,'train')]

print('start training')

gbm = xgb.train(param,dtrain,100,evallist)


y_pred=gbm.predict(dtest)
y_pred=pd.DataFrame(y_pred)
y_pred.set_index(index[40000:],inplace=True)
y_pred.head()
y_pred.to_csv('../input/ypred.csv')
print('Done')






