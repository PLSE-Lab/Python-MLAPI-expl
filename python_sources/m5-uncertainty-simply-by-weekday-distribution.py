#!/usr/bin/env python
# coding: utf-8

# # Very simple idea.
# 
# 1. Calculate distribution by weekday from last 2 months.
# 2. And calculate percentile 0.5%,2.5%,16.5%,25%,50%,75%,83.5%,97.5%,99.5%.
# 
# # But complication is sample submission file's data structure.

# In[ ]:


#DataFrame of sample submission file
#Percentile Start    End  Aggligation
#0.005         0       0  Total
#              1       3  state
#              4      13  store
#             14      16  cat
#             17      23  dept
#             14      32  state, cat
#             33      53  state, dept
#             54      83  store, cat
#             84     153  store, dept
#            154    3203  item
#           3203   12349  item, state
#          12350   42839  item, store
#0.025     42840   42840  Total
#          42841   42843  state
#          42844   42853  store
#          42854   42856  cat
#          42857   42863  dept
#          42864   42872  state, cat
#          42873   42893  state, dept
#          42894   42923  store, cat
#          42924   42993  store, dept
#          42994   46042  item
#          46043   55189  item, state
#          55190   85679  item, store
#
#  Continue 0.165 , 0.25 , ,,, 0.995 , same procedure of validation and evalunation.
#
#0.995    728280  728280  Total
#         728281  728283  state
#         728284  728293  store
#         728294  728296  cat
#         728297  728303  dept
#         728304  728312  state, cat
#         728313  728333  state, dept
#         728334  728363  store, cat
#         728364  728433  store, dept
#         728434  731482  item
#         731483  740629  item, state
#         740630  771119  item, store         


# And one more attention is that permutation of 'item, store' is not descending order.
# 
# So, when I use 'groupby', it become different order between result of 'groupby' and submission file's order. 
# 

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


validation = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sales_train_validation.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/calendar.csv')
sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv')
prices = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# In[ ]:


ss=sample_submission
ss_array=np.zeros((771120,29))


# In[ ]:


#Order of sample_submission is changed to same order of "sort_values". 
#(such that descending order)

for k in range(2):
    for i in range(9):
        ss_i_1=ss.iloc[385560*k+42840*i+1:385560*k+42840*i+4,:]
        ss_i_1_s=ss_i_1.sort_values(["id"])
        ss_i_1_s_r=ss_i_1_s.reset_index(drop=True)
        ss_i_1_s_r.index=ss_i_1_s_r.index+1+42840*i+385560*k
        ss.iloc[385560*k+42840*i+1:385560*k+42840*i+4,:]=ss_i_1_s
        
        ss_i_2=ss.iloc[385560*k+42840*i+4:385560*k+42840*i+14,:]
        ss_i_2_s=ss_i_2.sort_values(["id"])
        ss_i_2_s_r=ss_i_2_s.reset_index(drop=True)
        ss_i_2_s_r.index=ss_i_2_s_r.index+4+42840*i+385560*k
        ss.iloc[385560*k+42840*i+4:385560*k+42840*i+14,:]=ss_i_2_s
        
        ss_i_3=ss.iloc[385560*k+42840*i+14:385560*k+42840*i+17,:]
        ss_i_3_s=ss_i_3.sort_values(["id"])
        ss_i_3_s_r=ss_i_3_s.reset_index(drop=True)
        ss_i_3_s_r.index=ss_i_3_s_r.index+14+42840*i+385560*k
        ss.iloc[385560*k+42840*i+14:385560*k+42840*i+17,:]=ss_i_3_s
        
        ss_i_4=ss.iloc[385560*k+42840*i+17:385560*k+42840*i+24,:]
        ss_i_4_s=ss_i_4.sort_values(["id"])
        ss_i_4_s_r=ss_i_4_s.reset_index(drop=True)
        ss_i_4_s_r.index=ss_i_4_s_r.index+17+42840*i+385560*k
        ss.iloc[385560*k+42840*i+17:385560*k+42840*i+24,:]=ss_i_4_s
        
        ss_i_5=ss.iloc[385560*k+42840*i+24:385560*k+42840*i+33,:]
        ss_i_5_s=ss_i_5.sort_values(["id"])
        ss_i_5_s_r=ss_i_5_s.reset_index(drop=True)
        ss_i_5_s_r.index=ss_i_5_s_r.index+24+42840*i+385560*k
        ss.iloc[385560*k+42840*i+24:385560*k+42840*i+33,:]=ss_i_5_s
        
        ss_i_6=ss.iloc[385560*k+42840*i+33:385560*k+42840*i+54,:]
        ss_i_6_s=ss_i_6.sort_values(["id"])
        ss_i_6_s_r=ss_i_6_s.reset_index(drop=True)
        ss_i_6_s_r.index=ss_i_6_s_r.index+33+42840*i+385560*k
        ss.iloc[385560*k+42840*i+33:385560*k+42840*i+54,:]=ss_i_6_s
        
        ss_i_7=ss.iloc[385560*k+42840*i+54:385560*k+42840*i+84,:]
        ss_i_7_s=ss_i_7.sort_values(["id"])
        ss_i_7_s_r=ss_i_7_s.reset_index(drop=True)
        ss_i_7_s_r.index=ss_i_7_s_r.index+54+42840*i+385560*k
        ss.iloc[385560*k+42840*i+54:385560*k+42840*i+84,:]=ss_i_7_s
        
        ss_i_8=ss.iloc[385560*k+42840*i+84:385560*k+42840*i+154,:]
        ss_i_8_s=ss_i_8.sort_values(["id"])
        ss_i_8_s_r=ss_i_8_s.reset_index(drop=True)
        ss_i_8_s_r.index=ss_i_8_s_r.index+84+42840*i+385560*k
        ss.iloc[385560*k+42840*i+84:385560*k+42840*i+154,:]=ss_i_8_s
        
        ss_i_9=ss.iloc[385560*k+42840*i+154:385560*k+42840*i+3203,:]
        ss_i_9_s=ss_i_9.sort_values(["id"])
        ss_i_9_s_r=ss_i_9_s.reset_index(drop=True)
        ss_i_9_s_r.index=ss_i_9_s_r.index+154+42840*i+385560*k
        ss.iloc[385560*k+42840*i+154:385560*k+42840*i+3203,:]=ss_i_9_s
        
        ss_i_10=ss.iloc[385560*k+42840*i+3203:385560*k+42840*i+12350,:]
        ss_i_10_s=ss_i_10.sort_values(["id"])
        ss_i_10_s_r=ss_i_10_s.reset_index(drop=True)
        ss_i_10_s_r.index=ss_i_10_s_r.index+3203+42840*i+385560*k
        ss.iloc[385560*k+42840*i+3203:385560*k+42840*i+12350,:]=ss_i_10_s
        
        ss_i_11=ss.iloc[385560*k+42840*i+12350:385560*k+42840*i+42840,:]
        ss_i_11_s=ss_i_11.sort_values(["id"])
        ss_i_11_s_r=ss_i_11_s.reset_index(drop=True)
        ss_i_11_s_r.index=ss_i_11_s_r.index+12350+42840*i+385560*k
        ss.iloc[385560*k+42840*i+12350:385560*k+42840*i+42840,:]=ss_i_11_s_r


# In[ ]:


per=[0.5,2.5,16.5,25,50,75,83.5,97.5,99.5]


# In[ ]:


#Transform to nparray (To calculate fast)

validation=validation.T
validation=validation.reset_index()
validation=validation.T
validation_array=validation.values


# In[ ]:


#Use only last 2 month data

validtion_array_del=np.delete(validation_array,np.s_[6:1854],1)


# In[ ]:


#Return to DataFrame

validation_del=pd.DataFrame(validtion_array_del)
array_list=validation_del[0:1].values
validation_del.columns=array_list[0,:]
validation=validation_del.drop(0)
validation=validation.reset_index()
validation=validation.drop(['index'],axis=1)


# In[ ]:


validation


# In[ ]:


#Calculate TOTAL

index=[0]
unc_all_w=(np.zeros((7,9)))
allsales = pd.DataFrame(index=index,columns=[])

for i in range(1848,1913):
    allsales[f"d{i+1}"]=validation[f"d_{i+1}"].sum()
    
allsales = allsales.T
allsales=allsales.reset_index()
allsales=allsales.rename(columns={0:'allitem'})

calendar_wday=calendar['wday'].iloc[1848:1913].reset_index()
allsales['wday']=calendar_wday['wday']


for k in range(0,9):
    for i in range(0,7):
        unc_all_w[i,k] = np.percentile(allsales.query('wday == {}'.format(i+1)).allitem,q=per[k])


# In[ ]:



for j in range(0,9):
    for i in range(0,4):
        ss_array[j*42840,i*7+1]=unc_all_w[2,j]
        ss_array[j*42840,i*7+2]=unc_all_w[3,j]
        ss_array[j*42840,i*7+3]=unc_all_w[4,j]
        ss_array[j*42840,i*7+4]=unc_all_w[5,j]
        ss_array[j*42840,i*7+5]=unc_all_w[6,j]
        ss_array[j*42840,i*7+6]=unc_all_w[0,j]
        ss_array[j*42840,i*7+7]=unc_all_w[1,j]
    


# In[ ]:


#Calculate aggregation sales

def calculate_sales(series, select_id, drop_list, previous_end):
   
    unc_w=(np.zeros((7,9)))
    
    sales = pd.DataFrame(index=index,columns=[])
    
    sales = validation.groupby(select_id).sum() #select_id
        
    sales = sales.T
    sales=sales.reset_index()
    sales=sales.drop(drop_list,axis=0) #drop_list
    sales=sales.reset_index(drop=True)
    
    sales.columns=[i for i in range(series+1)]
    
    
    calendar_wday=calendar['wday'].iloc[1848:1913].reset_index()
    sales['wday']=calendar_wday['wday']
    
    
    
    #Calculate wday=1 (Monday)
    sales_1=sales.query('wday == [1]')
    sales_1=sales_1.drop([0],axis=1)
    sales_1_val=sales_1.values
    
    unc_w1=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w1[i,k] = np.percentile(sales_1_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+6]=unc_w1[l,r]       #previous_end  
            
    
    #Calculate wday=2 (Tuesday)
    sales_2=sales.query('wday == [2]')
    sales_2=sales_2.drop([0],axis=1)
    sales_2_val=sales_2.values
    
    unc_w2=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w2[i,k] = np.percentile(sales_2_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+7]=unc_w2[l,r]   #previous_end
    
    #Calculate wday=3 (Wednesday)
    sales_3=sales.query('wday == [3]')
    sales_3=sales_3.drop([0],axis=1)
    sales_3_val=sales_3.values
    
    unc_w3=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w3[i,k] = np.percentile(sales_3_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+1]=unc_w3[l,r]       #previous_end   
    
                
    #Calculate wday=4 (Thursday)
    sales_4=sales.query('wday == [4]')
    sales_4=sales_4.drop([0],axis=1)
    sales_4_val=sales_4.values
    
    unc_w4=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w4[i,k] = np.percentile(sales_4_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+2]=unc_w4[l,r]  #previous_end
                
                
    #Calculate wday=5 (Friday)
    sales_5=sales.query('wday == [5]')
    sales_5=sales_5.drop([0],axis=1)
    sales_5_val=sales_5.values
    
    unc_w5=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w5[i,k] = np.percentile(sales_5_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+3]=unc_w5[l,r]  #previous_end
                
                
    #Calculate wday=6 (Saturday)
    sales_6=sales.query('wday == [6]')
    sales_6=sales_6.drop([0],axis=1)
    sales_6_val=sales_6.values
    
    unc_w6=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w6[i,k] = np.percentile(sales_6_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+4]=unc_w6[l,r]     #previous_end        
    
                
    #Calculate wday=7 (Sunday)
    sales_7=sales.query('wday == [7]')
    sales_7=sales_7.drop([0],axis=1)
    sales_7_val=sales_7.values
    
    unc_w7=np.zeros((series,9))
    for k in range(0,9):
        for i in range(0,series):
            unc_w7[i,k] = np.percentile(sales_7_val[:,i],q=per[k])
    
    
    for l in range(0,series):
        for r in range(0,9):
            for n in range(0,4):
                ss_array[r*42840+l+previous_end,n*7+5]=unc_w7[l,r]  #previous_end
            


# In[ ]:


calculate_sales(3,['state_id'],[0,1,2,3,4],1)
calculate_sales(10,['store_id'],[0,1,2,3,4],4)
calculate_sales(3,['cat_id'],[0,1,2,3,4],14)
calculate_sales(7,['dept_id'],[0,1,2,3,4],17)
calculate_sales(9,['state_id','cat_id'],[0,1,2,3],24)
calculate_sales(21,['state_id','dept_id'],[0,1,2,3],33)
calculate_sales(30,['store_id','cat_id'],[0,1,2,3],54)
calculate_sales(70,['store_id','dept_id'],[0,1,2,3],84)
calculate_sales(3049,['item_id'],[0,1,2,3,4],154)
calculate_sales(9147,['item_id','state_id'],[0,1,2,3],3203)
calculate_sales(30490,['item_id','store_id'],[0,1,2,3],12350)


# In[ ]:


ss_array.shape


# In[ ]:


ss_array[385560:771120,:]=ss_array[0:385560,:]
ss_array_df=pd.DataFrame(ss_array)
ss_array_df[0]=ss['id']
ss_array_df.columns=['id','F1','F2','F3','F4','F5','F6','F7','F8',
                   'F9','F10','F11','F12','F13','F14','F15','F16',
                   'F17','F18','F19','F20','F21','F22','F23','F24',
                   'F25','F26','F27','F28']


# In[ ]:


ss_array_df.to_csv("submission.csv", index=False)


# # I will try not only percentile but Poisson, Gamma, Norm, etc.
