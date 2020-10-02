import sys
import gc
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn
import xgboost as xgb
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_colwidth=1000

   
###################################################################################################################
def MAPE(y_true, y_pred): 
    y_pred=y_pred.reshape((y_pred.shape[0],))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100,4)
###################################################################################################################
def GetXGBPreds(trainX,testX,trainY,obj_fun=None,NRounds=1,TH=0,XGBParams={}):
    dtrain = xgb.DMatrix(trainX, label=trainY)
    dtest = xgb.DMatrix(testX)
    xg1 = xgb.train(XGBParams, dtrain, NRounds, verbose_eval=False,obj=obj_fun)
    p1 = xg1.predict(dtest)
    return (p1)
###################################################################################################################
def GetLGBPreds(trainX,testX,trainY,obj_fun=None,NRounds=100,LGParams={}):
    dtrain = lgb.Dataset(trainX, trainY)
    gbm = lgb.train(LGParams, dtrain, num_boost_round=NRounds,fobj=obj_fun)
    p1 = gbm.predict(testX,num_iteration=NRounds)
    return (p1)
################################################
def GeBaggedPreds(trainX,testX,trainY,testY,Learner,LearnerArgs,NIters=5,rowsFraction=1.0,columnsFraction=1.0,doTrace=1,Seed=1):
    pBagg=np.zeros([testX.shape[0]])
    ctr=0;
    print('Bagging '+Learner.__name__+'.....');sys.stdout.flush();
    for ite in range(NIters):
        np.random.seed(seed=ite*Seed); 
        rspl=np.random.choice(trainX.shape[0],size=int(np.round(trainX.shape[0]*rowsFraction)),replace=0)
        np.random.seed(seed=(ite+1)*Seed); 
        cspl=np.random.choice(trainX.shape[1],size=int(np.round(trainX.shape[1]*columnsFraction)),replace=0)
        p1=Learner(trainX=sparse.csr_matrix(trainX[rspl,:][:,cspl]),testX=sparse.csr_matrix(testX[:,cspl]),trainY=trainY[rspl],**LearnerArgs);
        ctr=ctr+1;
        pBagg=pBagg+p1;
        if (doTrace):
            print ("Iter%d of %d, This_Iteration(MAPE): %s, Overall(MAPE): %s"%(ite+1,NIters,MAPE(np.exp(testY),np.exp(p1)),MAPE(np.exp(testY),np.exp(pBagg/ctr))));
            sys.stdout.flush();
    return(pBagg/ctr)           
###################################################################################################################


Final=False #Set this flag to true for final modelling
Sfraction=0.95# Fraction of train data for local validation
FSEED=2018 #Seed number for local validation splits 

print('Reading Data....')
train1=pd.read_csv("../input/train.csv")
test1=pd.read_csv("../input/test.csv")
test1['price']=1000000.0

if (Final):
    TestIDs=test1['ID']
    tt=pd.concat([train1,test1])
    LenTrain=train1.shape[0]   
else:
    tt=train1
    np.random.seed(seed=FSEED); 
    spl=np.random.choice(tt.shape[0],size=tt.shape[0],replace=0);
    tt=tt.loc[spl];
    LenTrain=int((tt.shape[0])*Sfraction)

#Converting locations into radian    
tt['sourceLatitude']=np.radians(tt['sourceLatitude'])
tt['sourceLongitude']=np.radians(tt['sourceLongitude'])
tt['destinationLatitude']=np.radians(tt['destinationLatitude'])
tt['destinationLongitude']=np.radians(tt['destinationLongitude'])

    
tt.fillna(0,inplace=True)
tt=tt.reset_index(drop=True)

'''
Add clustering features to dataset. 
Source and destination coordinates will be labeled using four types of clustering algorithms: KMeans, DBScan, HDBScan and Birch
For HDBScan coordinates has been rounded to 4 decimal places.
'''
print('Clustering....')
from sklearn.cluster import KMeans,DBSCAN,Birch
from hdbscan import HDBSCAN

#DBScan Clustering
print('           DBSCAN....')
x1=tt[['sourceLatitude','sourceLongitude']].reset_index(drop=True)
x2=tt[['destinationLatitude','destinationLongitude']].reset_index(drop=True)
x1.columns=['Lat','Lon'];x2.columns=['Lat','Lon']
points1=pd.concat([x1,x2]).drop_duplicates()
kms_per_radian = 6371.0088
epsilon = 1.5 / kms_per_radian
k1=DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='euclidean')
points1['cluster']=k1.fit(points1).labels_
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['sourceLatitude','sourceLongitude'],right_on=['Lat','Lon']).reset_index(drop=True)
del(tt['Lat']);del(tt['Lon']);
tt.rename(columns ={'cluster': 'source_DBSCAN_Cluster'}, inplace =True);
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['destinationLatitude','destinationLongitude'],right_on=['Lat','Lon']).reset_index(drop=True)
del(tt['Lat']);del(tt['Lon']);
tt.rename(columns ={'cluster': 'dest_DBSCAN_Cluster'}, inplace =True);

#HDBSCAN Clustering
print('           HDBSCAN....')
DP=4
tt['sourceLat2']=np.round(tt['sourceLatitude'],DP);tt['sourceLon2']=np.round(tt['sourceLongitude'],DP);
tt['destinationLat2']=np.round(tt['destinationLatitude'],DP);tt['destinationLon2']=np.round(tt['destinationLongitude'],DP);
x1=tt[['sourceLat2','sourceLon2']].reset_index(drop=True)
x2=tt[['destinationLat2','destinationLon2']].reset_index(drop=True)
x1.columns=['Lat2','Lon2'];x2.columns=['Lat2','Lon2']
points1=(pd.concat([x1,x2]));
points1=points1.drop_duplicates();
points1=points1.reset_index(drop=True)
np.random.seed(seed=82)
k1=HDBSCAN(algorithm='best', alpha=0.25, approx_min_span_tree=True,
        gen_min_span_tree=False, leaf_size=40,
        metric='euclidean', min_cluster_size=5, min_samples=2, p=None)
points1['cluster']=k1.fit(points1).labels_
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['sourceLat2','sourceLon2'],right_on=['Lat2','Lon2']).reset_index(drop=True)
del(tt['Lat2']);del(tt['Lon2']);
tt.rename(columns ={'cluster': 'source_HDBSCAN_Cluster'}, inplace =True);
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['destinationLat2','destinationLon2'],right_on=['Lat2','Lon2']).reset_index(drop=True)
del(tt['Lat2']);del(tt['Lon2']);
tt.rename(columns ={'cluster': 'dest_HDBSCAN_Cluster'}, inplace =True);
del(tt['sourceLat2']);del(tt['sourceLon2']);del(tt['destinationLat2']);del(tt['destinationLon2']);

#Birch clustering
print('           Birch....')
x1=tt[['sourceLatitude','sourceLongitude']].reset_index(drop=True)
x2=tt[['destinationLatitude','destinationLongitude']].reset_index(drop=True)
x1.columns=['Lat','Lon'];x2.columns=['Lat','Lon']
points1=pd.concat([x1,x2]).drop_duplicates()
k1=Birch(threshold=0.005, branching_factor=10, n_clusters=150, compute_labels=True, copy=True)
points1['cluster']=k1.fit(points1).labels_
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['sourceLatitude','sourceLongitude'],right_on=['Lat','Lon']).reset_index(drop=True)
del(tt['Lat']);del(tt['Lon']);
tt.rename(columns ={'cluster': 'source_Birch_Cluster'}, inplace =True);
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['destinationLatitude','destinationLongitude'],right_on=['Lat','Lon']).reset_index(drop=True)
del(tt['Lat']);del(tt['Lon']);
tt.rename(columns ={'cluster': 'dest_Birch_Cluster'}, inplace =True);

#KMeans Clustering
print('           KMeans....')
x1=tt[['sourceLatitude','sourceLongitude']].reset_index(drop=True)
x2=tt[['destinationLatitude','destinationLongitude']].reset_index(drop=True)
x1.columns=['Lat','Lon'];x2.columns=['Lat','Lon']
points1=pd.concat([x1,x2]).drop_duplicates()
k1=KMeans(n_clusters=150, init='k-means++', n_init=10, max_iter=100, tol=0.01, 
       precompute_distances='auto', verbose=0, random_state=1, copy_x=True, n_jobs=1,
       algorithm='auto')
points1['cluster']=k1.fit(points1).labels_
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['sourceLatitude','sourceLongitude'],right_on=['Lat','Lon']).reset_index(drop=True)
del(tt['Lat']);del(tt['Lon']);
tt.rename(columns ={'cluster': 'source_KMeans_Cluster'}, inplace =True);
tt=pd.merge(tt.reset_index(drop=True),points1.reset_index(drop=True),how='left',left_on=['destinationLatitude','destinationLongitude'],right_on=['Lat','Lon']).reset_index(drop=True)
del(tt['Lat']);del(tt['Lon']);
tt.rename(columns ={'cluster': 'dest_KMeans_Cluster'}, inplace =True);

#Creating some numeric features 
tt['F1']=np.log(1.0+(5.0+tt['weight'])*(5000.0+tt['distanceKM'])*(500.0+tt['taxiDurationMin']))
tt['F2']=np.log((tt['taxiDurationMin']+17*tt['weight'])**2)
tt['F3']=(np.log(1.0+tt['distanceKM']/(1.0+tt['taxiDurationMin'])))

#Creating two other categorical features
tt['Source_Destination']=tt['SourceState']+'_'+tt['destinationState']
tt['Vehicle_Type_Option']=tt['vehicleType']+'_'+tt['vehicleOption']

tt['sameSD']=(tt['SourceState']==tt['destinationState']).astype('int')

#Adding more features based on "date"
def doy(sdate):
    x=str(sdate)
    mnt=x[2:4]
    dy=x[4:6]
    d=(int(mnt)-1)*31+int(dy)
    return d
tt['Day_Of_Year']=tt['date'].apply(lambda x:(doy(x)))
tt['Week_Of_Year']=tt['Day_Of_Year'].apply(lambda x:round(x/7))
tt['Month_Of_Year']=tt['date'].astype('str').apply(lambda x:int(x[2:4]))
del(tt['date'])

tt=tt.reset_index(drop=True)
tt['distanceKM']=tt['distanceKM'].fillna(0)
tt.fillna('NV',inplace=True)

#Encoding Categorical Features
cats=['vehicleType','vehicleOption','SourceState','destinationState'
     ,'Source_Destination','Vehicle_Type_Option'
     ,'source_DBSCAN_Cluster','dest_DBSCAN_Cluster'
     ,'source_HDBSCAN_Cluster','dest_HDBSCAN_Cluster'
     ,'source_Birch_Cluster','dest_Birch_Cluster'
     ,'source_KMeans_Cluster','dest_KMeans_Cluster'
     ]
ttCat=tt[cats]
ttCat[cats].fillna('NV',inplace=True) 
for cat in cats:
    del(tt[cat]);
    ttCat[cat]=ttCat[cat].astype('category')
    ttCat[cat]=ttCat[cat].cat.codes;      
encoder=OneHotEncoder();
ttCat=encoder.fit_transform(ttCat);

Y=tt['price']
del(tt['price']);
del(tt['ID']);
tt=tt.reset_index(drop=True)


st=StandardScaler()
tt=st.fit_transform(tt)

tt=sparse.csr_matrix(tt)
ttX=sparse.hstack([tt,ttCat])
ttX=sparse.csc_matrix(ttX)

trainX=ttX[:LenTrain];
testX=ttX[LenTrain:];	
trainY=Y[:LenTrain];
testY=Y[LenTrain:];	



trainY=trainY.reset_index(drop=True)
testY=testY.reset_index(drop=True)

#Applying Ridge Regression algorithm. Predictions on test and train data will be added to dataset.
print('LR(Ridge).....');sys.stdout.flush();
ss=sklearn.linear_model.Ridge(alpha=5.0, fit_intercept=1, normalize=0,
                           copy_X=True, max_iter=None, tol=0.001,
                           solver='auto', random_state=1)
ss.fit(trainX,np.log(trainY*0.975))
ptst_LR=np.exp(ss.predict(testX))
print(MAPE(testY,ptst_LR))
ptrn_LR=np.exp(ss.predict(trainX))
print(MAPE(trainY,ptrn_LR))
ptrn_LR=ptrn_LR.reshape((ptrn_LR.shape[0],1))
ptst_LR=ptst_LR.reshape((ptst_LR.shape[0],1))
trainX=sparse.hstack([trainX,ptrn_LR])
testX=sparse.hstack([testX,ptst_LR])
trainX=sparse.csr_matrix(trainX)
testX=sparse.csr_matrix(testX)

print('.....................................................')

'''
Applying 3 Bagged Gradient Boosting Models: 
	XGBoost using gbtree method
	LightGBM using goss method
	LighGBM using dart method
Each model uses a different "mape" objective function.
 '''
 
############################################################################### 
def mape_obj1(preds, dtrain):
    labels = dtrain.get_label()
    grad = (preds-labels)/(0.25+labels*np.abs(preds-labels))
    hess = 0.1*np.ones(len(preds));
    return grad,hess
XGLearner=GetXGBPreds    
XGParams={'NRounds':1500,'obj_fun':mape_obj1,'XGBParams':{'objective' : 'reg:linear',
    "booster":"gbtree",'eta': 0.05,'max_depth': 15,'silent': 1,'min_child_weight': 0.01,
    'subsample': 1.0,'colsample_bytree': 0.8,'colsample_bylevel': 0.05,"alpha": 0.0,
    "lambda": 0.5,'seed': 1,"gamma":0.0,"max_delta_step":0,'nthread':4,'min_split_gain':0.0} }       
predXGB=GeBaggedPreds(trainX,testX,np.log(trainY*0.975),np.log(testY),XGLearner,XGParams,
                  NIters=40,rowsFraction=1.0,columnsFraction=1.0,doTrace=1,Seed=2)
predXGB=np.exp(predXGB)
print(MAPE(testY,predXGB))



def mape_obj2(preds, dtrain):
    labels = dtrain.get_label()
    grad = (preds-labels)/(0.5+labels*np.abs(preds-labels))
    hess = 0.2*np.ones(len(preds));
    return grad,hess
LGLearner=GetLGBPreds     
LGParams0={"NRounds":2500,'obj_fun':mape_obj2,"LGParams":{'task': 'train','boosting_type': 'goss'
                  ,'objective': 'regression','metric': 'mape','num_leaves': 50,
                  'learning_rate': 0.1,'feature_fraction': 0.2,
                  'top_rate':0.2,'other_rate':0.1,'max_bin':250,'num_threads':4,
                    'verbose': 1 ,'lambda_l1':0.0,'lambda_l2':0.0,'min_split_gain':0.0,
            		'min_data_in_leaf':3,'max_depth':30,'min_sum_hessian_in_leaf':1e-3}}        
predLGB_GOSS=GeBaggedPreds(trainX,testX,(np.log(trainY*0.975)),(np.log(testY)),LGLearner,LGParams0,
                  NIters=40,rowsFraction=1.0,
                  columnsFraction=1.0,doTrace=1,Seed=37)
predLGB_GOSS=np.exp(predLGB_GOSS)
print(MAPE(testY,(predLGB_GOSS)))


def mape_obj3(preds, dtrain):
    labels = dtrain.get_label()
    grad = (preds-labels)/(1.0+labels*np.abs(preds-labels))
    hess = 0.0025*np.ones(len(preds));
    return grad,hess
LGLearner=GetLGBPreds     
LGParams0={"NRounds":400,'obj_fun':mape_obj3,"LGParams":{'task': 'train','boosting_type': 'dart'
                  ,'objective': 'regression','metric': 'mape','num_leaves': 150,
                  'learning_rate': 0.007,'feature_fraction': 0.2,
                  'drop_rate': 0.1,'max_drop': 100,'skip_drop':0.5,'xgboost_dart_mode':False,
                  'uniform_drop':True,'max_bin':2500,'num_threads':4,
                    'verbose': 1 ,'lambda_l1':0.0,'lambda_l2':0.0,'min_split_gain':0.0,
            		'min_data_in_leaf':1,'max_depth':60,'min_sum_hessian_in_leaf':1e-3}}        
predLGB_DART=GeBaggedPreds(trainX,testX,(np.log(trainY*0.975)),(np.log(testY)),LGLearner,LGParams0,
                  NIters=80,rowsFraction=1.0,
                  columnsFraction=1.0,doTrace=1,Seed=37)
predLGB_DART=np.exp(predLGB_DART)
print(MAPE(testY,(predLGB_DART)))
print(MAPE(testY,((predXGB+predLGB_GOSS+2*predLGB_DART)/4)))

 
if (Final):
    print('Creating Submission.....');sys.stdout.flush();
    pf=np.round((predXGB+predLGB_GOSS+2*predLGB_DART)/4).astype('int')
    sm=pd.DataFrame(TestIDs)
    sm['price']=pf
    sm.to_csv('submission04.csv',index=False)
    


