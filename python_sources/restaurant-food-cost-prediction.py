#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
train = pd.read_excel('../input/food-cost/Data_Train.xlsx')
test = pd.read_excel('../input/food-cost/Data_Test.xlsx')
submission = pd.read_excel('../input/food-cost-submission/Sample_submission.xlsx')


# In[ ]:


train["source"] = "train"
test["source"] = "test"


# In[ ]:


df = pd.concat([train,test])


# Having null values in CITY,LOCALITY,RATING,VOTES
# 

# In[ ]:


df.info()


# In[ ]:


#Investigating the entire dataset first
df.duplicated().sum()


# In[ ]:


df= df.drop_duplicates()


# In[ ]:


df.isna().sum()


# In[ ]:


# Data exploration for CITY
# CITY has 147 null values
#combining City and locality
df['Location']=df['CITY']+' '+df['LOCALITY']
df.drop(columns=['CITY','LOCALITY'])


# In[ ]:


df.dropna(subset=['Location'],inplace=True)


# In[ ]:



from fuzzywuzzy import process
 
names_array=[]
def match_names(wrong_names,correct_names):
    for row in wrong_names:
        x=process.extractOne(row, correct_names)
        if x[1]<60:
            names_array.append('Others')
        else:
            names_array.append(x[0])
    return names_array
  
#Wrong country names dataset

correct_names=['Bangalore','Thane',
'Hyderabad','Andheri',
'Delhi', 'Kerala',
'Chennai', 'Bandra',
'Mumbai', 'Telangana',
'Kochi', 
'Noida', 
'Gurgaon', 'Ernakulam',
'Faridabad', 'Ghaziabad',
'Secunderabad' ]
name_match=match_names(df.Location,correct_names)    

print(len(names_array))
df['Location']=names_array


# In[ ]:


cuisines_list=[]
for row in df['CUISINES']:
    cuisines_list.append(list(row.split(',')))

df['CUISINES']=cuisines_list


# In[ ]:


df['CUISINES'].isna().sum()


# In[ ]:


df_cuisines=df['CUISINES'].apply(lambda x: pd.Series(1, x))


# In[ ]:


title_list=[]
for row in df['TITLE']:
    title_list.append(list(row.split(',')))
df['TITLE']=title_list


# In[ ]:


df_title=df['TITLE'].apply(lambda x: pd.Series(1, x))


# In[ ]:


df_title.head()


# In[ ]:


# cleaning time - pending


# In[ ]:


df[df['RATING'].isna()]


# In[ ]:


df["RATING"] = df.groupby("CITY").RATING.transform(lambda x : x.fillna(x.mode()[0]))


# In[ ]:


df['RATING']=df['RATING'].str.extract('(\d+)').astype(float)


# In[ ]:


df['VOTES'].isna().sum()


# In[ ]:


df.VOTES.fillna('0',inplace=True)
df['VOTES']=df['VOTES'].str.extract('(\d+)').astype(float)


# In[ ]:


df.drop(columns='CITY',inplace=True)
df.drop(columns='LOCALITY',inplace=True)
df.drop(columns='CUISINES',inplace=True)


# In[ ]:


#df.drop(columns='CUISINES',inplace=True)
#df.drop(columns='CUISINES++',inplace=True)
#df.drop(columns='Location++',inplace=True)
#df.drop(columns='TITLE++',inplace=True)


# In[ ]:


df_City=pd.get_dummies(df['Location'])
df.drop(columns='Location',inplace=True)
df_City.head()


# In[ ]:


df = pd.concat([df,df_City,df_cuisines,df_title], axis=1)


# In[ ]:


df.drop(columns='TITLE',inplace=True)


# In[ ]:


df_column_category = df.select_dtypes(exclude=np.number).columns
df_column_category


# In[ ]:


#df.drop(columns='City found',inplace=True)
df.drop(columns='TIME',inplace=True)


# In[ ]:


df.fillna(0,inplace=True)


# In[ ]:



train_final = df[df.source=="train"]
test_final = df[df.source=="test"]


# In[ ]:


train_final.shape


# In[ ]:


train_final.drop(columns=["source"],inplace=True)


# In[ ]:


test_final.drop(columns=["source",'COST'],inplace=True)


# In[ ]:


train_X = train_final.drop(columns=["COST",'RESTAURANT_ID'])


# In[ ]:


train_Y = train_final["COST"]


# In[ ]:


test_X = test_final.drop(columns=["RESTAURANT_ID"])


# In[ ]:


train_X.fillna(0,inplace=True)
train_X.isna().sum()


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_Y)
dtrain_predictions = model.predict(train_X)


# In[ ]:


from sklearn.model_selection import cross_val_score
a = cross_val_score(model, train_X, train_Y, cv=5, scoring='neg_mean_squared_error')


# In[ ]:


#Print model report:
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error, r2_score
print("\nModel Report")
print("RMSE : %.4g" % np.sqrt(mean_squared_error(train_Y.values, dtrain_predictions)))
    
#Predict on testing data:
test_X.fillna(0,inplace=True)
test_final["res_linear"] =  model.predict(test_X)


# In[ ]:


print('r2 train',r2_score(train_Y,dtrain_predictions))
#print('r2 test',r2_score(test_y,test_predict))


# In[ ]:


Linear_submission = test_final[["RESTAURANT_ID","res_linear"]]


# In[ ]:


Linear_submission.head(20)

