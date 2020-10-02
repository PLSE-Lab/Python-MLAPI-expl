#!/usr/bin/env python
# coding: utf-8

# This kernel for leakage of stie15 cornell [http://portal.emcs.cornell.edu/GannettHealthCenter] .
# 
# Thanks my teammate douhan li[http://www.facebook.com/lidouhan].He get me a lot help, Scraping a lot of data.
# 

# **1. So Far, This competition leakage 6 site.**
#     1. site0[ucl], 
#     2. site1[ucl], 
#     3. site2[asu], 
#     4. site3[WDC], [https://github.com/buds-lab/island-of-misfit-buildings/tree/master/data/processed](http://) [Dmitry Labazkin](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116773)
#     5. site4[Berkeley], 
#     6. site15[Cornell]]
# **2. and this data contain some fack data.**
#     1. site3 this data ,you can get from github,and just compare used building_id and year_build. but it's not very useful for this .because the 2016 meter_reading not very match. if you have interest, you can get this data ,used corr cal.
#     2. site4 this data, also have some mismatch, it's you can see https://www.kaggle.com/serengil/ucb-data-leakage-site-4-45-buildigs, 
#     
# **Now let find the Cornell building energy**
#     1. I used this data to submit. but it's not let me get a good imrove. so i think maybe this data in privte LB.
#     2. Cornell data, is very interest. because maybe organizer just want get more mismatch in this competition. organizer mix the building_id and meter_type. if you careful read my code, you can find building and meter is not match for the training data/
#     3. you can use this to scrape the data from cornell web.[http://portal.emcs.cornell.edu/GannettHealthCenter?cmd=download&s=1d&b=1574571600&e=1574658000](http://)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
import os
import gc
import sys
import glob


# In[ ]:


from pathlib import Path
root = Path('../input/ashrae-feather-format-for-fast-loading')
train_pd = pd.read_feather(root/'train.feather')
all_data = None


# # common data

# In[ ]:


def data_preprocess_site15(tempe_df,building_name):
    column_name = ['timestamp','Electric','ChilledWater','Steam']
    for column in column_name:
        if column not in tempe_df.columns:
            print(column)
            tempe_df[column] = np.nan
    #print(tempe_df.columns)
    tempe_df['timestamp'] = pd.to_datetime(tempe_df['timestamp'],unit='ms', origin='unix')
    tempe_df['label'] = building_name
    tempe_df[0] = tempe_df['Electric']
    tempe_df[1] = tempe_df['ChilledWater']# * 3.516
    tempe_df[2] = tempe_df['Steam']
    tempe_df = tempe_df.drop(['Electric', 'ChilledWater', 'Steam'],axis=1)
    tempe_df = tempe_df.join(tempe_df[[0,1,2]].stack().reset_index(level=1).rename(columns={"level_1": "meter", 0: "meter_reading"}))
    tempe_df = tempe_df.drop([0,1,2], axis=1)
    #tempe_df['meter'] = tempe_df['meter'].apply(int)
    #tempe_df = tempe_df.query('tempe_df[meter_reading.isna() == False]')
    tempe_df = tempe_df[tempe_df.meter_reading.isna() == False]
    tempe_df['meter'] = tempe_df['meter'].apply(int)
    return tempe_df


# In[ ]:


#for common data
file_path = '../input/cornell/site15/'
for dirname, _, filenames in os.walk(file_path):
    for filename in tqdm(filenames):
        filename = os.path.join(dirname, filename)
        #print(filename)
        name = filename.replace(file_path,"")
        name = name.replace("result_cornell_","")
        name = name.replace(".csv","")
        tempe_df = pd.read_csv(filename)
        tempe_df = data_preprocess_site15(tempe_df,name)
        
        all_data = pd.concat([all_data,tempe_df])
        del tempe_df
        gc.collect()
        
all_data = all_data.reset_index(drop=True)
#all_data.to_csv('../input/fetch_data/site15_tempe_all.csv',float_format='%.4f',index=None)


# # split data

# In[ ]:


def data_preprocess_site15_split(tempe_df,building_name,meter_type):

    tempe_df.columns = ['timestamp',meter_type]

    tempe_df['timestamp'] = pd.to_datetime(tempe_df['timestamp'],unit='ms', origin='unix')
    tempe_df['label'] = building_name
    return tempe_df


# In[ ]:


#for common data
file_path = '../input/cornell/cornell_split/'
all_data = None
name_preprocess = ''
process_cnt = 0
for dirname, _, filenames in os.walk(file_path):
    for filename in tqdm(filenames):
        filename = os.path.join(dirname, filename)
        #print(filename)
        name = filename.replace(file_path,"")
        name = name.replace("result_cornell_","")
        name = name.replace(".csv","")
        name, meter_type = name.split('_')

        if name_preprocess != name:
            find_str = file_path + "result_cornell_" + name +"_" + '*'
            lst = glob.glob(find_str)
            df_local = None
            for i, header in enumerate(lst):
                process_cnt += 1
                name_preprocess = name
                str_filename = header.replace("\\","/")
                fd_p = str_filename.replace("../input/cornell/cornell_split/result_cornell_{}_".format(name),"")
                meter_type = fd_p.replace(".csv","")
                print(name, meter_type)
                print(str_filename)
                tempe_df = pd.read_csv(str_filename,header=None)
                tempe_df = data_preprocess_site15_split(tempe_df,name,meter_type)
                if i == 0:
                    df_local = tempe_df
                else:
                    df_local = df_local.merge(tempe_df, on=['timestamp','label'],how='left')
                del tempe_df

            column_name = ['timestamp','Electric','ChilledWater','Steam','label']
            for column in column_name:
                if column not in df_local.columns:
                    print(column)
                    df_local[column] = np.nan

            df_local[0] = df_local['Electric']
            df_local[1] = df_local['ChilledWater']# * 3.516
            df_local[2] = df_local['Steam']
            df_local = df_local.drop(['Electric', 'ChilledWater', 'Steam'],axis=1)
            df_local = df_local.join(df_local[[0,1,2]].stack().reset_index(level=1).rename(columns={"level_1": "meter", 0: "meter_reading"}))
            df_local = df_local.drop([0,1,2], axis=1)
            df_local = df_local[df_local.meter_reading.isna() == False]
            df_local['meter'] = df_local['meter'].apply(int)

            all_data = pd.concat([all_data,df_local])
            del df_local
            gc.collect()
print('processed cnt = {}'.format(process_cnt))


# In[ ]:


all_data


# In[ ]:


final_building_scrap_pd = all_data


# In[ ]:


sit15_compare_df = pd.DataFrame(columns = ['building_id', 'label','meter','score'])


# In[ ]:


for j in range(4):
    df_sit2 = train_pd[(train_pd.building_id >= 1325)]

    test1 = pd.DataFrame()
    test1['building_id'] = df_sit2['building_id']
    test1['meter_reading'] = df_sit2['meter_reading']
    test1['timestamp'] = df_sit2['timestamp']
    test1['meter'] = df_sit2['meter']
    test2 = pd.DataFrame()
    test2['building_id'] = final_building_scrap_pd['label']
    test2['meter_reading'] = final_building_scrap_pd['meter_reading']
    test2['timestamp'] = final_building_scrap_pd['timestamp']
    test2['meter'] = final_building_scrap_pd['meter']
    test2 = pd.concat([test1,test2])
    test2 = test2[test2['timestamp'] <= '2017-01-01 1:00:00']
    test2 = test2[test2['meter'] == j].sort_values('timestamp')
    cr = test2.pivot_table(index = 'timestamp', columns='building_id', values = 'meter_reading', aggfunc=np.mean)

    data_cr = cr.corr(method='spearman')

    m=0

    for i in range(1325,1450):
        #print(i)
        if i not in test2['building_id'].unique():
            #print("dont have {} in meter:{}".format(i,j))
            m += 1
        else:
            data_f = data_cr.replace(1,0)[i].idxmax()
            #print(i, data_f,data_cr.replace(1,0)[i].max() ,j, train_pd[train_pd['building_id'] == i]['meter'].unique())#data_f[i])
            if data_cr.replace(1,0)[i].max() > 0.99:
                print(i, data_f,data_cr.replace(1,0)[i].max() ,j, train_pd[train_pd['building_id'] == i]['meter'].unique())#data_f[i])
                sit15_compare_df = sit15_compare_df.append({'building_id':i,'label':data_f, 'meter':j,'score':data_cr.replace(1,0)[i].max()},ignore_index=True)
    print(m)


# 1. if you want to get more data,you can let the threshold down than 0.95. 
# 2. if you want to get all corr score. just drop the threshold.

# In[ ]:


sit15_compare_df


# In[ ]:


all_data_t = all_data.merge(sit15_compare_df, on=['label', 'meter'], how='left')
all_data_t = all_data_t[all_data_t['building_id'].isna() == False]
all_data_t = all_data_t.drop_duplicates()
all_data_t = all_data_t.reset_index(drop = True)
del all_data_t['label'], final_building_scrap_pd, all_data
gc.collect()


# In[ ]:


all_data_t = all_data_t[(all_data_t['timestamp'] >= '2017/01/01 0:00:00') & (all_data_t['timestamp'] <'2019/01/01 0:00:00')]


# In[ ]:


all_data_t.to_csv("site15_leakage.csv", index=None)
all_data_t


# In[ ]:


def script_plot(final_building_scrap_pd, train_pd, meter, b_id,start_time = '2016-1-1 1:00:00', end_time = '2019-1-1 1:00:00'): 
    scrap = final_building_scrap_pd[(final_building_scrap_pd.meter == meter) & (final_building_scrap_pd.timestamp < '2017-01-01 00:00:00') & 
                             (final_building_scrap_pd.building_id == b_id)].reset_index().set_index('timestamp').sort_index()["meter_reading"].values
    hist = train_pd.query("meter == %d & building_id == %d" % (meter,b_id)).set_index('timestamp').sort_index()["meter_reading"].values

    if len(scrap) != len(hist):
        print("\n**** Building id = %d, not same length scrap = %d train = %d ****" %(b_id, len(scrap), len(hist)))
        validated_ids.append((b_id, len(scrap), len(hist)))
    else:
        diff = np.nansum(hist - scrap)
        print("Building id = %d, Diff = %d" %(b_id, diff))
        validated_ids.append((b_id, len(scrap), len(hist)))

    fig, ax = plt.subplots(figsize=(20, 4))
    d = final_building_scrap_pd[(final_building_scrap_pd.meter == meter )& (final_building_scrap_pd.building_id == b_id)&
    (final_building_scrap_pd.timestamp >= start_time)&(final_building_scrap_pd.timestamp <= end_time)].set_index('timestamp').plot(
    kind='line', y=["meter_reading"], ax=ax, color='tab:blue', linestyle='-', linewidth=0.5)
    d = train_pd[(train_pd.meter == meter) & (train_pd.building_id == b_id)&
    (train_pd.timestamp >= start_time)&(train_pd.timestamp <= end_time)].set_index('timestamp').plot(
    kind='line', y="meter_reading",alpha=0.5,  ax=ax, color='tab:orange', linewidth=1.0, title="meter: %d building_id: %d" % (meter, b_id))
    plt.show()


# In[ ]:


validated_ids = []
for i in tqdm(range(len(sit15_compare_df))):
    temp_df = sit15_compare_df.loc[i]
    building_id = temp_df['building_id']
    meter = temp_df['meter']
    label = temp_df['label']
    if len(all_data_t[all_data_t.building_id == building_id]) != 0:
        print(building_id , label, meter)
        script_plot(all_data_t, train_pd, meter, building_id)
        

