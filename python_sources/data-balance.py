#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def data_balance(train_df):
    split_df = train_df["ImageId_ClassId"].str.split("_", expand = True)
# add new columns to train_df
    train_df['Image'] = split_df[0]
    train_df['Label'] = split_df[1]
    train_df['EncodedPixels'] = train_df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)
    
#datasets for different defects
    defect1 = train_df[(train_df['Label'] == '1') & (train_df['EncodedPixels'] == 1)]['Image']#datasets for defect1, 897
    defect2 = train_df[(train_df['Label'] == '2') & (train_df['EncodedPixels'] == 1)]['Image']#datasets for defect2, 247
    defect3 = train_df[(train_df['Label'] == '3') & (train_df['EncodedPixels'] == 1)]['Image']#datasets for defect3, 5150
    defect4 = train_df[(train_df['Label'] == '4') & (train_df['EncodedPixels'] == 1)]['Image']#datasets for defect4, 801 
    no_defect = train_df[(train_df['EncodedPixels'] == 0)]['Image']

    data_wihout_defect3 = pd.concat([defect1, defect2,defect4])#datasets without defect3, 1945 (= 897 + 247 +801)

    train_df_all = pd.DataFrame()#create empty dataframe
    for i in range(5):
        defect3_sub = defect3[1030 * i : 1030 * (i + 1)] # one fifth of defect3, equated to 1030 (=5150/5) 
        train_df = pd.concat([data_wihout_defect3, defect3_sub],ignore_index = True)
        train_df_all = pd.concat([train_df_all, train_df], axis = 1) # (datasets without defect3) + (one fifth of defect3), equated to 2975 (=1945 + 1030) 

    #print (train_df_all) 
    return train_df_all # includes 5 colums, correspongding to 5 csvs

#defect1_num = defect1.count()
#defect2_num = defect2.count()
#defect3_num = defect3.count()
#defect4_num = defect4.count()
#no_defect_num = no_defect.count()
#    path=''+str(i)+'.csv'
#    train_df.to_csv(path)


# In[ ]:


train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
train_df_balance = data_balance(train_df)
print(train_df_balance)

