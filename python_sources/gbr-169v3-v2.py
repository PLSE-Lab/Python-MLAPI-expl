#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error



def rmsle(y, y_pred):
        assert len(y) == len(y_pred)
        terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
        return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
    

def fix_target(frame, key, target, new_target_name="target"):
    import numpy as np

    corrections = 0
    group_keys = frame[ key].values.tolist()
    target = frame[target].values.tolist()

    for i in range(1, len(group_keys) - 1):
        previous_group = group_keys[i - 1]
        current_group = group_keys[i]

        previous_value = target[i - 1]
        current_value = target[i]
        if current_group == previous_group:
                if current_value<previous_value:
                    current_value=previous_value
                    target[i] =current_value


        target[i] =max(0,target[i] )#correct negative values

    frame[new_target_name] = np.array(target)
    
    
def rate(frame, key, target, new_target_name="rate"):
    import numpy as np


    corrections = 0
    group_keys = frame[ key].values.tolist()
    target = frame[target].values.tolist()
    rate=[1.0 for k in range (len(target))]

    for i in range(1, len(group_keys) - 1):
        previous_group = group_keys[i - 1]
        current_group = group_keys[i]

        previous_value = target[i - 1]
        current_value = target[i]
         
        if current_group == previous_group:
                if previous_value!=0.0:
                     rate[i]=current_value/previous_value

                 
        rate[i] =max(1,rate[i] )#correct negative values

    frame[new_target_name] = np.array(rate)
    
def get_data_by_key(dataframe, key, key_value, fields=None):
    mini_frame=dataframe[dataframe[key]==key_value]
    if not fields is None:                
        mini_frame=mini_frame[fields].values
        
    return mini_frame

directory="/kaggle/input/covid19-global-forecasting-week-3/"
model_directory="/kaggle/input/modelv3-dir/model"
geo_dir=None#"/kaggle/input/covid19-forecasting-metadata/"
extra_stable_columns=None#["lat","lon","population","area","density"] #"continent"
group_by_columns=None #["Country_Region"]#["continent"]

if not group_by_columns is None and "continent" in group_by_columns:
    assert not geo_dir is None

train=pd.read_csv(directory + "train.csv", parse_dates=["Date"] , engine="python")
test=pd.read_csv(directory + "test.csv", parse_dates=["Date"], engine="python")

train["key"]=train[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)
test["key"]=test[["Province_State","Country_Region"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)
if not geo_dir is None:
    region_metadata=pd.read_csv(geo_dir+"region_metadata.csv")
    train=pd.merge(train,region_metadata, how="left", left_on=["Province_State","Country_Region"], right_on=["Province_State","Country_Region"] )
    test=pd.merge(test,region_metadata, how="left", left_on=["Province_State","Country_Region"], right_on=["Province_State","Country_Region"] )
    train.to_csv(directory + "train_plus_geo.csv", index=False)

    
    
group_names=None

if  not group_by_columns is None and len(group_by_columns)>0:
    group_names=[]
    for group in group_by_columns:
        groupss=train[["Date", group,target1,target2]]
        grp=groupss.groupby(["Date", group], as_index=False).sum()
        grp.columns=["Date", group,group +"_" + target1,group +"_" + target2 ]
        train=pd.merge(train,grp, how="left", left_on=["Date", group,group], right_on=["Date", group,group] )
        #group_names+=[group +"_" + target1,group +"_" + target2]
        for gr in [group +"_" + target1,group +"_" + target2]:
            rate(train, key, gr, new_target_name="rate_" +gr ) 
            group_names+=["rate_" +gr]
        train.to_csv(directory + "train_plus_groups.csv", index=False)
        
        
    
#last day in train
max_train_date=train["Date"].max()
max_test_date=test["Date"].max()
horizon=  (max_test_date-max_train_date).days
print ("horizon", int(horizon))
print ("max_train_date", (max_train_date))
print ("max_test_date", (max_test_date))

#test_new=pd.merge(test,train, how="left", left_on=["key","Date"], right_on=["key","Date"] )
#train.to_csv(directory + "transfomed.csv")

target1="ConfirmedCases"
target2="Fatalities"

key="key"


# In[ ]:


fix_target(train, key, target1, new_target_name=target1)
fix_target(train, key, target2, new_target_name=target2)

rate(train, key, target1, new_target_name="rate_" +target1 )
rate(train, key, target2, new_target_name="rate_" +target2 )
unique_keys=train[key].unique()
print(len(unique_keys))


train


# In[ ]:



def get_lags(rate_array, current_index, size=20):
    lag_confirmed_rate=[-1 for k in range(size)]
    for j in range (0, size):
        if current_index-j>=0:
            lag_confirmed_rate[j]=rate_array[current_index-j]
        else :
            break
    return lag_confirmed_rate

def days_ago_thresold_hit(full_array, indx, thresold):
        days_ago_confirmed_count_10=-1
        if full_array[indx]>thresold: # if currently the count of confirmed is more than 10
            for j in range (indx,-1,-1):
                entered=False
                if full_array[j]<=thresold:
                    days_ago_confirmed_count_10=abs(j-indx)
                    entered=True
                    break
                if entered==False:
                    days_ago_confirmed_count_10=100 #this value would we don;t know it cross 0      
        return days_ago_confirmed_count_10 
    
    
def ewma_vectorized(data, alpha):
    sums=sum([ (alpha**(k+1))*data[k] for  k in range(len(data)) ])
    counts=sum([ (alpha**(k+1)) for  k in range(len(data)) ])
    return sums/counts

def generate_ma_std_window(rate_array, current_index, size=20, window=3):
    ma_rate_confirmed=[-1 for k in range(size)]
    std_rate_confirmed=[-1 for k in range(size)] 
    
    for j in range (0, size):
        if current_index-j>=0:
            ma_rate_confirmed[j]=np.mean(rate_array[max(0,current_index-j-window+1 ):current_index-j+1])
            std_rate_confirmed[j]=np.std(rate_array[max(0,current_index-j-window+1 ):current_index-j+1])           
        else :
            break
    return ma_rate_confirmed, std_rate_confirmed

def generate_ewma_window(rate_array, current_index, size=20, window=3, alpha=0.05):
    ewma_rate_confirmed=[-1 for k in range(size)]

    
    for j in range (0, size):
        if current_index-j>=0:
            ewma_rate_confirmed[j]=ewma_vectorized(rate_array[max(0,current_index-j-window+1 ):current_index-j+1, ], alpha)           
        else :
            break
    
    #print(ewma_rate_confirmed)
    return ewma_rate_confirmed


def get_target(rate_col, indx, horizon=33, average=3, use_hard_rule=False):
    target_values=[-1 for k in range(horizon)]
    cou=0
    for j in range(indx+1, indx+1+horizon):
        if j<len(rate_col):
            if average==1:
                target_values[cou]=rate_col[j]
            else :
                if use_hard_rule and j +average <=len(rate_col) :
                     target_values[cou]=np.mean(rate_col[j:j +average])
                else :
                    target_values[cou]=np.mean(rate_col[j:min(len(rate_col),j +average)])
                   
            cou+=1
        else :
            break
    return target_values


def dereive_features(frame, confirmed, fatalities, rate_confirmed, rate_fatalities, 
                     horizon ,size=20, windows=[3,7], days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10], 
                    extra_data=None, groups_data=None, windows_group=[3,7], size_group=20,
                    days_back_confimed_group=[1,10,100]):
    targets=[]
    if not extra_data is None:
        assert len(extra_stable_columns)==extra_data.shape[1]
        
    if not groups_data is None:
        assert len(group_names)==groups_data.shape[1]        
        
    names=["lag_confirmed_rate" + str(k+1) for k in range (size)]
    for day in days_back_confimed:
        names+=["days_ago_confirmed_count_" + str(day) ]
    for window in windows:        
        names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]
        names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 
        names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         
        
        
    names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]
    for day in days_back_fatalities:
        names+=["days_ago_fatalitiescount_" + str(day) ]    
    for window in windows:        
        names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]
        names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  
        names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]        
    names+=["confirmed_level"]
    names+=["fatalities_level"]   
    if not extra_data is None: 
        names+=[k for k in extra_stable_columns]
    if not groups_data is None:  
         for gg in range (groups_data.shape[1]):
             names+=["lag_rate_group_"+ str(gg+1) + "_" + str(k+1) for k in range (size_group)]    
             for day in days_back_confimed_group:
                names+=["days_ago_grooupcount_" + str(gg+1) + "_" + str(day) ]             
             for window in windows_group:        
                names+=["ma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]
                names+=["std_group_" + str(gg+1)+ "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]  
                #names+=["ewma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size)]  

            
    names+=["confirmed_plus" + str(k+1) for k in range (horizon)]    
    names+=["fatalities_plus" + str(k+1) for k in range (horizon)]  
    
    #names+=["current_confirmed"]
    #names+=["current_fatalities"]    
    
    features=[]
    for i in range (len(confirmed)):
        row_features=[]
        #####################lag_confirmed_rate       
        lag_confirmed_rate=get_lags(rate_confirmed, i, size=size)
        row_features+=lag_confirmed_rate
        #####################days_ago_confirmed_count_10
        for day in days_back_confimed:
            days_ago_confirmed_count_10=days_ago_thresold_hit(confirmed, i, day)               
            row_features+=[days_ago_confirmed_count_10] 
        #####################ma_rate_confirmed       
        #####################std_rate_confirmed 
        for window in windows:
            ma3_rate_confirmed,std3_rate_confirmed= generate_ma_std_window(rate_confirmed, i, size=size, window=window)
            row_features+= ma3_rate_confirmed   
            row_features+= std3_rate_confirmed          
            ewma3_rate_confirmed=generate_ewma_window(rate_confirmed, i, size=size, window=window, alpha=0.05)
            row_features+= ewma3_rate_confirmed              
        #####################lag_fatalities_rate   
        lag_fatalities_rate=get_lags(rate_fatalities, i, size=size)
        row_features+=lag_fatalities_rate
        #####################days_ago_confirmed_count_10
        for day in days_back_fatalities:
            days_ago_fatalitiescount_2=days_ago_thresold_hit(fatalities, i, day)               
            row_features+=[days_ago_fatalitiescount_2]     
        #####################ma_rate_fatalities       
        #####################std_rate_fatalities 
        for window in windows:        
            ma3_rate_fatalities,std3_rate_fatalities= generate_ma_std_window(rate_fatalities, i, size=size, window=window)
            row_features+= ma3_rate_fatalities   
            row_features+= std3_rate_fatalities  
            ewma3_rate_fatalities=generate_ewma_window(rate_fatalities, i, size=size, window=window, alpha=0.05)
            row_features+= ewma3_rate_fatalities                  
        ##################confirmed_level
        confirmed_level=0
        
        """
        if confirmed[i]>0 and confirmed[i]<1000:
            confirmed_level= confirmed[i]
        else :
            confirmed_level=2000
        """   
        confirmed_level= confirmed[i]
        row_features+=[confirmed_level]
        ##################fatalities_is_level
        fatalities_is_level=0
        """
        if fatalities[i]>0 and fatalities[i]<100:
            fatalities_is_level= fatalities[i]
        else :
            fatalities_is_level=200            
        """
        fatalities_is_level= fatalities[i]
        
        row_features+=[fatalities_is_level] 
        
        if not extra_data is None:    
            row_features+=extra_data[i].tolist()
            
        if not groups_data is None:  
          for gg in range (groups_data.shape[1]): 
             ## lags per group
             this_group=groups_data[:,gg].tolist()
             lag_group_rate=get_lags(this_group, i, size=size_group)
             row_features+=lag_group_rate           
             #####################days_ago_confirmed_count_10
             for day in days_back_confimed_group:
                days_ago_groupcount_2=days_ago_thresold_hit(this_group, i, day)               
                row_features+=[days_ago_groupcount_2]     
             #####################ma_rate_fatalities       
             #####################std_rate_fatalities 
             for window in windows_group:        
                ma3_rate_group,std3_rate_group= generate_ma_std_window(this_group, i, size=size_group, window=window)
                row_features+= ma3_rate_group   
                row_features+= std3_rate_group             
            
            
        #######################confirmed_plus target
        confirmed_plus=get_target(rate_confirmed, i, horizon=horizon)
        row_features+= confirmed_plus          
        #######################fatalities_plus target
        fatalities_plus=get_target(rate_fatalities, i, horizon=horizon)
        row_features+= fatalities_plus 
        ##################current_confirmed
        #row_features+=[confirmed[i]]
        ##################current_fatalities
        #row_features+=[fatalities[i]]        
        
          

        
        features.append(row_features)
        
    new_frame=pd.DataFrame(data=features, columns=names).reset_index(drop=True)
    frame=frame.reset_index(drop=True)
    frame=pd.concat([frame, new_frame], axis=1)
    #print(frame.shape)
    return frame
    
    
def feature_engineering_for_single_key(frame, group, key, horizon=33, size=20, windows=[3,7], 
                                       days_back_confimed=[1,10,100], days_back_fatalities=[1,2,10],
                                      extra_stable_=None, group_nams=None,windows_group=[3,7], 
                                       size_group=20, days_back_confimed_group=[1,10,100]):
    
    mini_frame=get_data_by_key(frame, group, key, fields=None)
    
    mini_frame_with_features=dereive_features(mini_frame, mini_frame["ConfirmedCases"].values,
                                              mini_frame["Fatalities"].values, mini_frame["rate_ConfirmedCases"].values, 
                                               mini_frame["rate_Fatalities"].values, horizon ,size=size, windows=windows,
                                              days_back_confimed=days_back_confimed, days_back_fatalities=days_back_fatalities,
                                              extra_data=mini_frame[extra_stable_].values if not extra_stable_ is None else None,
                                              groups_data=mini_frame[group_nams].values if not group_nams is None else None,
                                              windows_group=windows_group, size_group=size_group, 
                                              days_back_confimed_group=days_back_confimed_group)
    #print (mini_frame_with_features.shape[0])
    return mini_frame_with_features


# In[ ]:


from tqdm import tqdm
train_frame=[]
size=10
windows=[3]
days_back_confimed=[1,5,10,20,50,100,250,500,1000]
days_back_fatalities=[1,2,5,10,20,50]

size_group=10
windows_group=[3,5]
days_back_confimed_group=[1,10,100]


#print (len(train['key'].unique()))
for unique_k in tqdm(unique_keys):
    mini_frame=feature_engineering_for_single_key(train, key, unique_k, horizon=horizon, size=size, 
                                                  windows=windows, days_back_confimed=days_back_confimed,
                                                  days_back_fatalities=days_back_fatalities,
                                                  extra_stable_=extra_stable_columns if extra_stable_columns is not None and len(extra_stable_columns)>0 else None,
                                     group_nams=group_names,windows_group=windows_group, 
                                     size_group=size_group, days_back_confimed_group=days_back_confimed_group
                                                 ).reset_index(drop=True) 
    #print (mini_frame.shape[0])
    train_frame.append(mini_frame)
    
train_frame = pd.concat(train_frame, axis=0).reset_index(drop=True)
#train_frame.to_csv(directory +"all" + ".csv", index=False)
new_unique_keys=train_frame['key'].unique()
for kee in new_unique_keys:
    if kee not in unique_keys:
        print (kee , " is not there ")


# In[ ]:


import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.externals import joblib

def predict(xtest,input_name=None):
   #print (type(yt))
   # create array object to hold predictions 
  
   baggedpred=np.array([ 0.0 for d in range(0, xtest.shape[0])]) 
   model=  joblib.load( input_name) 
   preds=model.predict(xtest)               
   baggedpred+=preds

   return baggedpred


# In[ ]:



names=["lag_confirmed_rate" + str(k+1) for k in range (size)]
for day in days_back_confimed:
    names+=["days_ago_confirmed_count_" + str(day) ]
for window in windows:        
    names+=["ma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]
    names+=["std" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)] 
    names+=["ewma" + str(window) + "_rate_confirmed" + str(k+1) for k in range (size)]         


names+=["lag_fatalities_rate" + str(k+1) for k in range (size)]
for day in days_back_fatalities:
    names+=["days_ago_fatalitiescount_" + str(day) ]    
for window in windows:        
    names+=["ma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]
    names+=["std" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]  
    names+=["ewma" + str(window) + "_rate_fatalities" + str(k+1) for k in range (size)]        
names+=["confirmed_level"]
names+=["fatalities_level"]   
if not extra_stable_columns is None and len(extra_stable_columns)>0: 
    names+=[k for k in extra_stable_columns]  
    
if not group_names is None:  
     for gg in range (len(group_names)):
         names+=["lag_rate_group_"+ str(gg+1) + "_" + str(k+1) for k in range (size_group)]    
         for day in days_back_confimed_group:
            names+=["days_ago_grooupcount_" + str(gg+1) + "_" + str(day) ]             
         for window in windows_group:        
            names+=["ma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]
            names+=["std_group_" + str(gg+1)+ "_" + str(window) + "_rate_" + str(k+1) for k in range (size_group)]  
            #names+=["ewma_group_" + str(gg+1) + "_" + str(window) + "_rate_" + str(k+1) for k in range (size)]      
      


# In[ ]:



def decay_4_first_10_then_1_f(array):
    arr=[1.0 for k in range(len(array))]
    for j in range(len(array)):
        if j<10:
            arr[j]=1. + (max(1,array[j])-1.)/4.
        else :
            arr[j]=1.
    return arr
	
def decay_16_first_10_then_1_f(array):
    arr=[1.0 for k in range(len(array))]
    for j in range(len(array)):
        if j<10:
            arr[j]=1. + (max(1,array[j])-1.)/16.
        else :
            arr[j]=1.
    return arr	
            
def decay_2_f(array):
    arr=[1.0 for k in range(len(array))]    
    for j in range(len(array)):
            arr[j]=1. + (max(1,array[j])-1.)/2.
    return arr 

def decay_4_f(array):
    arr=[1.0 for k in range(len(array))]    
    for j in range(len(array)):
            arr[j]=1. + (max(1,array[j])-1.)/4.
    return arr 	
	
def acceleratorx2_f(array):
    arr=[1.0 for k in range(len(array))]    
    for j in range(len(array)):
            arr[j]=1. + (max(1,array[j])-1.)*2.
    return arr 



def decay_1_5_f(array):
    arr=[1.0 for k in range(len(array))]    
    for j in range(len(array)):
            arr[j]=1. + (max(1,array[j])-1.)/1.5
    return arr            
         
         
def stay_same_f(array):
    arr=[1.0 for k in range(len(array))]      
    for j in range(len(array)):
        arr[j]=1.
    return arr   

def decay_2_last_12_linear_inter_f(array):
    arr=[1.0 for k in range(len(array))]
    for j in range(len(array)):
        arr[j]=1. + (max(1,array[j])-1.)/2.
    arr12= (max(1,arr[-12])-1.)/12. 

    for j in range(0, 12):
        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))
    return arr

def decay_4_last_12_linear_inter_f(array):
    arr=[1.0 for k in range(len(array))]
    for j in range(len(array)):
        arr[j]=1. + (max(1,array[j])-1.)/4.
    arr12= (max(1,arr[-12])-1.)/12. 

    for j in range(0, 12):
        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))
    return arr

def linear_last_12_f(array):
    arr=[1.0 for k in range(len(array))]
    for j in range(len(array)):
        arr[j]=max(1,array[j])
    arr12= (max(1,arr[-12])-1.)/12. 
    
    for j in range(0, 12):
        arr[len(arr)-12 +j]= max(1, 1 + ( (arr12*12) - (j+1)*arr12 ))
    return arr
    
decay_4_first_10_then_1 =[ "Heilongjiang_China","Liaoning_China","Shanghai_China"]#, "Hong Kong_China"

decay_4_first_10_then_1_fatality=[]

decay_16_first_10_then_1 =["Beijing_China","Fujian_China","Guangdong_China","Shandong_China","Sichuan_China","Zhejiang_China"]
decay_16_first_10_then_1_fatality=[]

decay_4=["nan_Bhutan","nan_Burundi","nan_Cabo Verde","Prince Edward Island_Canada",
"nan_Central African Republic","Inner Mongolia_China","nan_Maldives"]
decay_4_fatality=["nan_Congo (Kinshasa)"]

decay_2 =["nan_Congo (Kinshasa)","Faroe Islands_Denmark","nan_Eritrea","French Guiana_France","nan_Korea, South","nan_MS Zaandam"]
decay_2_fatality=[]

stay_same=["nan_Diamond Princess","nan_Timor-Leste"]
		   
stay_same_fatality=["Beijing_China","Fujian_China","Guangdong_China","Shandong_China",
"Sichuan_China","Zhejiang_China", "Heilongjiang_China","Liaoning_China","Shanghai_China"]#

normal=[]
normal_fatality=["nan_Korea, South","New York_US"]

decay_4_last_12_linear_inter =[ "Greenland_Denmark","nan_Dominica","nan_Equatorial Guinea","nan_Eswatini","New Caledonia_France",
"Saint Barthelemy_France","St Martin_France","nan_Gambia","nan_Grenada","nan_Holy See","nan_Mauritania","nan_Namibia","nan_Nicaragua"
,"nan_Papua New Guinea","nan_Saint Lucia","nan_Saint Vincent and the Grenadines","nan_Seychelles","nan_Sierra Leone","nan_Somalia","nan_Suriname",
"Anguilla_United Kingdom","British Virgin Islands_United Kingdom","Montserrat_United Kingdom","Turks and Caicos Islands_United Kingdom","nan_Zimbabwe"
                              , "Hong Kong_China","Curacao_Netherlands"]

decay_4_last_12_linear_inter_fatality=[]

decay_2_last_12_linear_inter =[ "nan_Chad",
"nan_Congo (Brazzaville)","nan_Fiji","French Polynesia_France","nan_Gabon",
"nan_Guyana","nan_Laos","nan_Nepal","Sint Maarten_Netherlands",
"nan_Saint Kitts and Nevis","nan_Sudan","nan_Syria","nan_Tanzania",
"Bermuda_United Kingdom","Cayman Islands_United Kingdom","nan_Zambia","Northwest Territories_Canada","Yukon_Canada"
,"nan_Mongolia","nan_Uganda"]
decay_2_last_12_linear_inter_fatality=[]

acceleratorx2=[]
acceleratorx2_fatality=[]


warm_st=['nan_Angola','nan_Antigua and Barbuda','Northern Territory_Australia','nan_Bahamas',
'nan_Bangladesh','nan_Belize','nan_Benin','nan_Botswana','nan_Burundi','nan_Cabo Verde','nan_Cameroon',
'nan_Central African Republic','nan_Chad','Hong Kong_China',"nan_Cote d'Ivoire",'nan_Cuba','Greenland_Denmark',
'nan_Dominica','nan_Equatorial Guinea','nan_Eritrea','nan_Eswatini','nan_Fiji','French Polynesia_France','New Caledonia_France',
'Saint Barthelemy_France','St Martin_France','nan_Gabon','nan_Gambia','nan_Grenada','nan_Guyana','nan_Haiti','nan_Holy See',
'nan_Honduras','nan_Ireland','nan_Korea, South','nan_Laos','nan_Liberia','nan_Libya','nan_Maldives','nan_Mali',
'nan_Mauritania','nan_Mauritius','nan_Mongolia','nan_Mozambique','nan_Namibia','nan_Nepal','Aruba_Netherlands',
'nan_Nicaragua','nan_Niger','nan_Papua New Guinea','nan_Saint Kitts and Nevis','nan_Saint Lucia',
'nan_Saint Vincent and the Grenadines','nan_Seychelles','nan_Sierra Leone','nan_Somalia',
'nan_Spain','nan_Sudan','nan_Suriname','nan_Syria','nan_Tanzania','nan_Togo','nan_Uganda','Anguilla_United Kingdom',
'Bermuda_United Kingdom','British Virgin Islands_United Kingdom','Channel Islands_United Kingdom',
'Gibraltar_United Kingdom','Isle of Man_United Kingdom','Montserrat_United Kingdom','nan_United Kingdom',
'Turks and Caicos Islands_United Kingdom','nan_Uzbekistan','nan_Zimbabwe',
]


decay_1_5 =["nan_Angola" ,"nan_Antigua and Barbuda","Montana_US","Nebraska_US","nan_Bangladesh","Illinois_US"
,"Northern Territory_Australia","nan_Bahamas","nan_Bahrain","nan_Barbados" ,"nan_Belize","nan_Benin",
	"nan_Botswana","nan_Brunei","Manitoba_Canada","New Brunswick_Canada","Saskatchewan_Canada",
	"nan_Cote d'Ivoire","nan_France","nan_Guinea-Bissau","nan_Haiti","nan_Italy","nan_Libya","nan_Malta","nan_Mauritius",
	"Aruba_Netherlands","nan_Niger","nan_Spain","nan_Togo","Guam_US","Iowa_US","Idaho_US","Connecticut_US","California_US","New York_US","Virgin Islands_US",
	"Channel Islands_United Kingdom","Gibraltar_United Kingdom","Isle of Man_United Kingdom","nan_United Kingdom",'nan_Burma']

decay_1_5_fatality=["nan_Cameroon","nan_Mali","nan_Cuba","Delaware_US","District of Columbia_US",
"Kansas_US","Louisiana_US","Michigan_US","New Mexico_US","Ohio_US","Oklahoma_US","Pennsylvania_US","Puerto Rico_US","Rhode Island_US",
"South Dakota_US" ,"Tennessee_US","Texas_US","Vermont_US","Virginia_US","West Virginia_US","nan_Uzbekistan"]

linear_last_12=["nan_Honduras","nan_Ireland","Colorado_US","nan_Liberia","nan_Mozambique"]
linear_last_12_fatality=[]


tr_frame=train_frame

features_train=tr_frame[names].values   

standard_confirmed_train=tr_frame["ConfirmedCases"].values
standard_fatalities_train=tr_frame["Fatalities"].values
current_confirmed_train=tr_frame["ConfirmedCases"].values

     

features_cv=[]
name_cv=[]
standard_confirmed_cv=[]
standard_fatalities_cv=[]
names_=tr_frame["key"].values
training_horizon=int(features_train.shape[0]/len(unique_keys)) 
print("training horizon = ",training_horizon)
for dd in range(training_horizon-1,features_train.shape[0],training_horizon):
    features_cv.append(features_train[dd])
    name_cv.append(names_[dd])
    standard_confirmed_cv.append(standard_confirmed_train[dd])
    standard_fatalities_cv.append(standard_fatalities_train[dd])
    print (name_cv[-1], standard_confirmed_cv[-1], standard_fatalities_cv[-1])
    
 

features_cv=np.array(features_cv)
preds_confirmed_cv=np.zeros((features_cv.shape[0],horizon))
preds_confirmed_standard_cv=np.zeros((features_cv.shape[0],horizon))

preds_fatalities_cv=np.zeros((features_cv.shape[0],horizon))
preds_fatalities_standard_cv=np.zeros((features_cv.shape[0],horizon))

overal_rmsle_metric_confirmed=0.0

for j in range (preds_confirmed_cv.shape[1]):

    this_features_cv=features_cv                          

    preds=predict(features_cv, input_name=model_directory +"confirmed"+ str(j))
    preds_confirmed_cv[:,j]=preds
    print (" modelling confirmed, case %d, , original cv %d and after %d "%(j,this_features_cv.shape[0],this_features_cv.shape[0])) 

predictions=[] 
for ii in range (preds_confirmed_cv.shape[0]):
    current_prediction=standard_confirmed_cv[ii]
    if current_prediction==0 :
        current_prediction=0.1   
    this_preds=preds_confirmed_cv[ii].tolist()
    name=name_cv[ii]
    reserve=this_preds[0]
    #overrides

	
    if name in normal:
        this_preds=this_preds	
	
    elif name in decay_4_first_10_then_1:
        this_preds=decay_4_first_10_then_1_f(this_preds)
		
    elif name in decay_16_first_10_then_1:
        this_preds=decay_16_first_10_then_1_f(this_preds)		
		
    elif name in decay_4_last_12_linear_inter:
        this_preds=decay_4_last_12_linear_inter_f(this_preds)		        
          
    elif name in decay_4:
        this_preds=decay_4_f(this_preds)

    
    elif name in decay_2:
        this_preds=decay_2_f(this_preds)
        
    elif name in decay_2_last_12_linear_inter:
        this_preds=decay_2_last_12_linear_inter_f(this_preds)
        
    elif name in decay_1_5:
        this_preds=decay_1_5_f(this_preds)        
        
    elif name in linear_last_12:
        this_preds=linear_last_12_f(this_preds)
        
    elif name in acceleratorx2:
        this_preds=acceleratorx2_f(this_preds)         

        
    elif name in stay_same or  "China" in name:
        this_preds=stay_same_f(this_preds)      

    if name in warm_st:
        this_preds[0]=reserve
    for j in range (preds_confirmed_cv.shape[1]):
                current_prediction*=max(1,this_preds[j])
                preds_confirmed_standard_cv[ii][j]=current_prediction


for j in range (preds_confirmed_cv.shape[1]):

    this_features_cv=features_cv
                             
    preds=predict(features_cv, input_name=model_directory +"fatal"+ str(j))
    preds_fatalities_cv[:,j]=preds
    print (" modelling fatalities, case %d, original cv %d and after %d "%( j,this_features_cv.shape[0],this_features_cv.shape[0])) 

predictions=[]
for ii in range (preds_fatalities_cv.shape[0]):
    current_prediction=standard_fatalities_cv[ii]
    if current_prediction==0 and standard_confirmed_cv[ii]>400:
        current_prediction=0.1
        
    this_preds=preds_fatalities_cv[ii].tolist()
    name=name_cv[ii]
    reserve=this_preds[0]
    #overrides
   
    ####fatality special
	
    if name in normal_fatality:
        this_preds=this_preds	 	
	
    elif name in decay_4_first_10_then_1_fatality:
        this_preds=decay_4_first_10_then_1_f(this_preds) 
		
    elif name in decay_16_first_10_then_1_fatality:
        this_preds=decay_16_first_10_then_1_f(this_preds)
        
    elif name in decay_4_last_12_linear_inter_fatality:
        this_preds=decay_4_last_12_linear_inter_f(this_preds)		            

    elif name in decay_4_fatality:
        this_preds=decay_4_f(this_preds)		
		
    elif name in decay_2_fatality:
        this_preds=decay_2_f(this_preds)        

    elif name in decay_2_last_12_linear_inter_fatality:
        this_preds=decay_2_last_12_linear_inter_f(this_preds)
        
    elif name in decay_1_5_fatality:
        this_preds=decay_1_5_f(this_preds)   	
 
    elif name in linear_last_12_fatality:
        this_preds=linear_last_12_f(this_preds) 
         
    elif name in acceleratorx2_fatality:
        this_preds=acceleratorx2_f(this_preds)
        
    elif name in stay_same_fatality:     
        this_preds=stay_same_f(this_preds) 
        
     
    ####general   
    elif name in normal:
        this_preds=this_preds	   
	
    elif name in decay_4_first_10_then_1:
        this_preds=decay_4_first_10_then_1_f(this_preds)
		
    elif name in decay_16_first_10_then_1:
        this_preds=decay_16_first_10_then_1_f(this_preds)		
        
    elif name in decay_4_last_12_linear_inter:
        this_preds=decay_4_last_12_linear_inter_f(this_preds)		               

    elif name in decay_4:
        this_preds=decay_4_f(this_preds)        
		
    elif name in decay_2:
        this_preds=decay_2_f(this_preds)
        
    elif name in decay_2_last_12_linear_inter:
        this_preds=decay_2_last_12_linear_inter_f(this_preds)
        
    elif name in decay_1_5:
        this_preds=decay_1_5_f(this_preds)        
        
    elif name in linear_last_12:
        this_preds=linear_last_12_f(this_preds) 
        
    elif name in acceleratorx2:
        this_preds=acceleratorx2_f(this_preds)                 
        
    elif name in stay_same or  "China" in name:
        this_preds=stay_same_f(this_preds)         
       
    if name in warm_st:
        this_preds[0]=reserve    
    
    for j in range (preds_fatalities_cv.shape[1]):
                if current_prediction==0 and  (preds_confirmed_standard_cv[ii][j]>400 or "Malta" in name or "Somalia" in name):
                    current_prediction=1.
                if j==0 and "nan_Antigua and Barbuda" in name:
                    current_prediction=2.
                if j==0 and 'nan_Burma' in name:
                    current_prediction=3.                    
                current_prediction*=max(1,this_preds[j])
                preds_fatalities_standard_cv[ii][j]=current_prediction


# In[ ]:


key_to_confirmed_rate={}
key_to_fatality_rate={}
key_to_confirmed={}
key_to_fatality={}
print(len(features_cv), len(name_cv),len(standard_confirmed_cv),len(standard_fatalities_cv)) 
print(preds_confirmed_cv.shape,preds_confirmed_standard_cv.shape,preds_fatalities_cv.shape,preds_fatalities_standard_cv.shape) 

for j in range (len(name_cv)):
    
    key_to_confirmed_rate[name_cv[j]]=preds_confirmed_cv[j,:].tolist()
    #print(key_to_confirmed_rate[name_cv[j]])
    key_to_fatality_rate[name_cv[j]]=preds_fatalities_cv[j,:].tolist()
    key_to_confirmed[name_cv[j]]  =preds_confirmed_standard_cv[j,:].tolist()  
    key_to_fatality[name_cv[j]]=preds_fatalities_standard_cv[j,:].tolist()  


# In[ ]:


train_new=train[["Date","ConfirmedCases","Fatalities","key","rate_ConfirmedCases","rate_Fatalities"]]

test_new=pd.merge(test,train_new, how="left", left_on=["key","Date"], right_on=["key","Date"] ).reset_index(drop=True)
test_new


# In[ ]:


def fillin_columns(frame,key_column, original_name, training_horizon, test_horizon, unique_values, key_to_values):
    keys=frame[key_column].values
    original_values=frame[original_name].values.tolist()
    print(len(keys), len(original_values), training_horizon ,test_horizon,len(key_to_values))
    
    for j in range(unique_values):
        current_index=(j * (training_horizon +test_horizon )) +training_horizon 
        current_key=keys[current_index]
        values=key_to_values[current_key]
        co=0
        for g in range(current_index, current_index + test_horizon):
            original_values[g]=values[co]
            co+=1
    
    frame[original_name]=original_values
 

all_days=int(test_new.shape[0]/len(unique_keys))

tr_horizon=all_days-horizon
print(all_days,tr_horizon, horizon )

fillin_columns(test_new,"key", 'ConfirmedCases', tr_horizon, horizon, len(unique_keys), key_to_confirmed)    
fillin_columns(test_new,"key", 'Fatalities', tr_horizon, horizon, len(unique_keys), key_to_fatality)   
submission=test_new[["ForecastId","ConfirmedCases","Fatalities"]]

submission.to_csv( "submission.csv", index=False)


# In[ ]:




