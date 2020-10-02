#!/usr/bin/env python
# coding: utf-8

# The model uses all historical data for the user.
# The best lb score was ~.26 using only the 'days_since_last_ordered'
# I think by using xgboosts "update" function the model could be updated every time a user places a new order

# In[ ]:



__docformat__ = 'restructedtext en'

import timeit
import time

import os
import threading


import pandas as pd
import numpy as np

import scipy as sci
import sklearn as sk
from sklearn import metrics
import statsmodels.sandbox.stats.runs as statruns
import scipy.stats as sci
from sklearn import preprocessing

import xgboost as xgb


''' inportant link bout the data https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b '''


aisle = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')#Order history of customer
order_products_train = pd.read_csv('../input/order_products__train.csv')
orders = pd.read_csv('../input/orders.csv')#tells to which set (prior, train, test) an order belongs
products = pd.read_csv('../input/products.csv')

'''Find missing values '''
aisle.isnull().values.any()
departments.isnull().values.any()
order_products_prior.isnull().values.any()
order_products_train.isnull().values.any()
orders.isnull().values.any()
products.isnull().values.any()
test=orders[orders.eval_set=='test']
'''Seems like the only NaN is from days_since_prior_order and that is because they are new customers
and this is their first order'''
orders[orders.days_since_prior_order.isnull()]

''' # of unique clients '''
orders.user_id.nunique()

'''merge order_product_prior and orders so information is in one data frame '''
merged_prior=pd.merge(left=order_products_prior,right=orders, on='order_id')
merged_train=pd.merge(left=order_products_train,right=orders, on='order_id')

'''order df so orders are in crohnological order, order_number represents the number
of orders the custerm has placed '''
merged_prior.sort_values(by='order_number', axis=0, inplace=True)
merged_train.sort_values(by='order_number', axis=0, inplace=True)

mergedall=pd.concat([merged_prior,merged_train,test])
merged_prior=mergedall


# The "merged_prior=mergedall`" is like this so I did not have to go back and change it in every spot in the script
# 

# In[ ]:


def load_data_XG(FnlBinDailyData):
    """ Loads the dataset

    :type dataset: Pandas Dataframe
    :param dataset: A dataset with the variables and the labels
    """
    
    Vars=FnlBinDailyData.columns[4:FnlBinDailyData.columns.size-1]#have -1 becauase order_id is last and that is not a feature
    
    D = xgb.DMatrix(FnlBinDailyData[Vars],label=FnlBinDailyData['reordered'].values)
       
    rval = [D]
    
    return D


# In[ ]:


def xgBoost(mydata,test,thresh):
    """ Loads the dataset

    :type mydata: Pandas Dataframe
    :param mydata: The training dataset with the variables and the labels
    
    :type test: Pandas Dataframe
    :param test: The varibles to be used for predicting 
    
    :type thresh: An int
    :param thresh: The threshhold value at which we say the product is in the order
    """
    # prepare dict of params for xgboost to run with
    xgb_params = {
        #'n_trees': 50, 
        'eta': .1,
        'max_depth': 6,
        'subsample': 0.76,
        'objective': 'reg:logistic',
        'eval_metric': 'logloss',
        'lambda': 10,
        'gamma': .7,
        'colsample_bytree':.95,
        'silent': 1,
        'alpha': 2e-05
    }
    
    #for when it breaks we know what user it was on.... becuase it always breaks the first
    #couple of times
    print('xgb user_id ',mydata.user_id[0])
    
    # form DMatrices for Xgboost training
    dtrain = load_data_XG(mydata)#xgb.DMatrix(mydata, mydata.y)
    
    #because in some orders there is less than 5 products
    if(dtrain.get_label().size>15):
        # xgboost, cross-validation
        cv_result = xgb.cv(xgb_params, dtrain,num_boost_round=50, # increase to have better results (~700)
                           nfold=5,early_stopping_rounds=30,
                           show_stdv=False
                          )
           
        num_boost_rounds = len(cv_result)
        print(num_boost_rounds)
        
        
        # train model
        model = xgb.train(dict(xgb_params, silent=0), dtrain,num_boost_round=num_boost_rounds)
    else:
        model = xgb.train(dict(xgb_params, silent=0),dtrain)
    
    # now fixed, correct calculation
    pred=model.predict(dtrain)
    for w in enumerate(pred):
        if(pred[w[0]]>thresh):
            pred[w[0]]=1
        else:
            pred[w[0]]=0    
    
    print('XGB Score ', f1_score(dtrain.get_label(), pred), ' User_id ', mydata.user_id[0])
    
    dtest = xgb.DMatrix(test[model.feature_names])
    y_pred=model.predict(dtest)
    pred=y_pred
    
    for w in enumerate(pred):
        if(pred[w[0]]>thresh):
            pred[w[0]]=1
        else:
            pred[w[0]]=0   
    
    popo=pd.concat([pd.DataFrame(test.product_id.values), pd.DataFrame(pred),pd.DataFrame(test.order_id.values)],axis=1,ignore_index=True)
    popo.rename(columns={0:'product_id',1:'pred',2:'order_id'},inplace = True)
    
    d=dict()
    for row in popo.itertuples():
        if row.pred ==1:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)
    if(len(d)==0):
        d[row.order_id]='None'
        print('None')
    return test.order_id.unique(), d


# In[ ]:


#Sort by most active users in the test set
dfindexer=merged_prior.groupby('user_id').product_id.count().sort_values(ascending=False).index#.intersection(test.user_id.unique())
myseries=pd.Series(index=range(test.user_id.nunique()))
indexcount=np.int(0)
for a in dfindexer:
    if(a in test.user_id.unique()):
        myseries.set_value(indexcount,a,takeable=True)
        indexcount+=1
dfindexer=pd.Index(myseries)

#The code above is slow so I saved the results and just reload them
#myseries.to_csv('dfindexer.csv')
#dfindexer=pd.read_csv('dfindexer.csv',header=-1)
dfindexer=pd.Index(dfindexer[0])
order_id_splits=pd.DataFrame(index=range(np.int((dfindexer.size-1)/10)+1), 
                             columns=['g0','g1','g2','g3','g4',
                                      'g5','g6','g7','g8','g9'])


# In[ ]:


count=np.int(0)
count1=np.int(0)
while count<np.int((dfindexer.size-1)/10):    
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g0'),dfindexer[count1],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g1'),dfindexer[count1+1],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g2'),dfindexer[count1+2],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g3'),dfindexer[count1+3],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g4'),dfindexer[count1+4],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g5'),dfindexer[count1+5],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g6'),dfindexer[count1+6],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g7'),dfindexer[count1+7],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g8'),dfindexer[count1+8],takeable=True)
    order_id_splits.set_value(count,order_id_splits.columns.get_loc('g9'),dfindexer[count1+9],takeable=True)
    count1=count1+10
    count=count+1
 
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g0'),dfindexer[dfindexer.size-10],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g1'),dfindexer[dfindexer.size-9],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g2'),dfindexer[dfindexer.size-8],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g3'),dfindexer[dfindexer.size-7],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g4'),dfindexer[dfindexer.size-6],takeable=True) 
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g5'),dfindexer[dfindexer.size-5],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g6'),dfindexer[dfindexer.size-4],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g7'),dfindexer[dfindexer.size-3],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g8'),dfindexer[dfindexer.size-2],takeable=True)
order_id_splits.set_value(count,order_id_splits.columns.get_loc('g9'),dfindexer[dfindexer.size-1],takeable=True)

grouped = merged_prior.groupby(['user_id','order_id']).product_id.count()
aver_order_size=pd.DataFrame(index=range(merged_prior.user_id.nunique()), columns=['user_id','avg_num_prods_in_cart'])
idcount=np.int(0)


# In[ ]:


#Get the size of the dataframe fro the user_id I am modeling
def getlen(ids):
    count=0
    idcount=0
    for userid in ids:
        d=merged_prior[merged_prior.user_id==userid]
        print('count', count,' idcount ', idcount)
        for e in d.order_number.unique():#iterate over order numbers
            if(e==1):
                daset=d[d.order_number==e].apply(set)
                count=count+len(daset.product_id)
            else:
                daset1=d[d.order_number==e].apply(set)
                count=count+len(daset.product_id)+len(daset.product_id.symmetric_difference(daset1.product_id))
                daset.product_id.update(daset1.product_id)
        
        idcount=idcount+1
    return count


# fastest way to add data to a dataframe
# df.set_value()

# In[ ]:


#Builds the dataframe with all the order history for the user_id we are trying to predict
merged_prior.sort_values(by=['user_id','order_number'], inplace =True)
def buildDf(ids):
    df=pd.DataFrame(index=range(getlen(ids)),columns=['reordered','product_in_order','user_id','order_number','product_id',
                                                      'days_since_last_ordered',
                                                      'order_id'])#
    idcount=0
    dfit=np.int(0)
    we=dict()
    order_id_col=0
    days_since_prior_order_col=1
    reorded_col=2
    num_time_buy_product_col=3
    num_orders_since_last_purchase_col=4
    for userid in ids:
        t0=time.time()
        #could set index to user_id and sort the below??
        b=merged_prior[merged_prior.user_id==userid]
        d=b.order_number.unique()
        print('idcount ', idcount)
        for e in d:#iterate over order numbers
            for c in b[b.order_number==e].itertuples():#iterate over products
                '''has product ever been purchased before
                if yes, then we are reordering the product'''
                if(c.product_id in we):                    
                    '''I do not add becuase for predictin I do not want to say it is in the order becuase I do not really know '''
                    num_times_buy_product=we[c.product_id][len(we[c.product_id])-1][num_time_buy_product_col]
                    '''I know this is a new order and I want to take that into considaration '''
                    num_orders_since_last_purchase=we[c.product_id][len(we[c.product_id])-1][num_orders_since_last_purchase_col]+1
                    '''I know the days since prior order so I  want to add that in '''
                    this_days_since_prior_order=we[c.product_id][len(we[c.product_id])-1][days_since_prior_order_col]+c.days_since_prior_order
                    we[c.product_id].append([e,this_days_since_prior_order,c.reordered,num_times_buy_product,1])
                else:
                    if(c.product_id>0):
                        we[c.product_id]=[[e,0,c.reordered,1,0]]#first time buying product
            
            for key in we:
                #are we reordering a product in this order, yes == 1, no == 0
                if(key in b[b.order_number==e].product_id.values):#'''product is in the order'''
                    if(we[key][len(we[key])-1][reorded_col]>0):#c.days_since_prior_order>0):
                        df.set_value(dfit,df.columns.get_loc('user_id'),c.user_id,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('order_number'),e,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('product_id'),key,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('product_in_order'),1,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('reordered'),we[key][len(we[key])-1][reorded_col],takeable=True)
                        df.set_value(dfit,df.columns.get_loc('days_since_last_ordered'),c.days_since_prior_order,takeable=True)
                        f.set_value(dfit,df.columns.get_loc('order_id'),c.order_id,takeable=True)
                        dfit=dfit+1
                        
                        '''Update we to reflect that the product was in this order '''
                        we[key][len(we[key])-1]=[e,c.days_since_prior_order,we[key][len(we[key])-1][reorded_col],we[key][len(we[key])-1][num_time_buy_product_col]+1,1]
                        
                else:#'''product not in order '''                    
                    if(c.days_since_prior_order>0):#we[key][len(we[key])-1][days_since_prior_order_col]>0):                   
                        #dslo=last[last.product_id==key].days_since_last_ordered.values[0]+c.days_since_prior_order
                        dic=we[key][len(we[key])-1][days_since_prior_order_col]+c.days_since_prior_order
                        num_times_buy_product=we[key][len(we[key])-1][num_time_buy_product_col]
                        num_orders_since_last_purchase=we[key][len(we[key])-1][num_orders_since_last_purchase_col]+1
                        we[key].append([e,dic,0,num_times_buy_product,num_orders_since_last_purchase])
                        df.set_value(dfit,df.columns.get_loc('user_id'),c.user_id,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('order_number'),e,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('product_id'),key,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('product_in_order'),0,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('reordered'),0,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('days_since_last_ordered'),dic,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('order_id'),c.order_id,takeable=True)
                        dfit=dfit+1
                    else:#First order by user
                        df.set_value(dfit,df.columns.get_loc('user_id'),c.user_id,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('order_number'),e,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('product_id'),key,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('product_in_order'),0,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('reordered'),0,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('days_since_last_ordered'),0,takeable=True)
                        df.set_value(dfit,df.columns.get_loc('order_id'),c.order_id,takeable=True)
                        dfit=dfit+1
        
        we.clear()
        idcount=idcount+1
        t1=time.time()
        print('len time ', t1-t0)
    return df


# In[ ]:


#Puts it all together
def xg(splits,filename,thresh):
    
    daindex=pd.Index(merged_prior[merged_prior.eval_set=='test'].order_id.unique(),name='order_id')
    preddf=pd.DataFrame(index=daindex,columns=['products'])
    for split in enumerate(splits):
        mydf=buildDf([split[1]])
        mydf.dropna(inplace=True)
        mydf=mydf.astype(np.int64)
        for a in mydf.user_id.unique():
            fg=mydf[mydf.user_id==a]
            pred_order_id,predvalues=xgBoost(fg[fg.order_number<fg.order_number.max()], fg[fg.order_number==fg.order_number.max()],thresh)
            preddf.set_value(daindex.get_loc(pred_order_id[0]),preddf.columns.get_loc('products') , predvalues.get(pred_order_id[0]), takeable=True)
        print('number of models built ', split[0])
    preddf.dropna(inplace=True) 
    preddf.to_csv(filename)


# In[ ]:


#Calls the function that puts it all together.
#I have divided the dataset into 10 groups. I run two seperate instances of the script
#each instance contains 5 sets of user_id, When everything is done each instance saves
#the predicult results for its 5 groups of user_ids and then I merge the two .csv's by
#hand and then submit

jobNumber=1

filenames=['set0.csv','set1.csv','set2.csv','set3.csv','set4.csv','set5.csv','set6.csv','set7.csv',
           'set8.csv','set9.csv','final0.csv','final1.csv']

if(jobNumber==1):
    mythread0=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g0,'filename':filenames[0],'thresh':.22})
    mythread0.start()
     mythread1=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g1,'filename':filenames[1],'thresh':.22})
     mythread1.start()
     mythread2=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g2,'filename':filenames[2],'thresh':.22})
     mythread2.start()
     mythread3=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g3,'filename':filenames[3],'thresh':.22})
     mythread3.start()
     mythread4=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g4,'filename':filenames[4],'thresh':.22})
     mythread4.start()
     
     mythread0.join()
     mythread1.join()
     mythread2.join()
     mythread3.join()
     mythread4.join()

    csv0=pd.read_csv(filenames[0])
    csv1=pd.read_csv(filenames[1])
    csv2=pd.read_csv(filenames[2])
    csv3=pd.read_csv(filenames[3])
    csv4=pd.read_csv(filenames[4])
    pd.concat([csv0,csv1,csv2,csv3,csv4]).to_csv(filenames[10])
else:
    mythread0=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g5,'filename':filenames[5],'thresh':.22})
    mythread0.start()
    mythread1=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g6,'filename':filenames[6],'thresh':.22})
    mythread1.start()
    mythread2=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g7,'filename':filenames[7],'thresh':.22})
    mythread2.start()
    mythread3=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g8,'filename':filenames[8],'thresh':.22})
    mythread3.start()
    mythread4=threading.Thread(target=xg,kwargs={'splits':order_id_splits.g9,'filename':filenames[9],'thresh':.22})
    mythread4.start()
    
    mythread0.join()
    mythread1.join()
    mythread2.join()
    mythread3.join()
    mythread4.join()

    csv0=pd.read_csv(filenames[5])
    csv1=pd.read_csv(filenames[6])
    csv2=pd.read_csv(filenames[7])
    csv3=pd.read_csv(filenames[8])
    csv4=pd.read_csv(filenames[9])
    pd.concat([csv0,csv1,csv2,csv3,csv4]).to_csv(filenames[11])


# In[ ]:




