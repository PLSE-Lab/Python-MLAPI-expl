#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# <h2 id="problem-statement">Data</h2>
# <p><strong>Train Data</strong></p>
# <table border="1" cellspacing="0"><colgroup width="200"></colgroup><colgroup width="472"></colgroup>
# <tbody>
# <tr>
# <td align="left" valign="bottom" height="20"><strong>Variable</strong></td>
# <td align="left" valign="bottom"><strong>Definition</strong></td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">session_id</td>
# <td align="left" valign="bottom">Unique ID for a session</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">DateTime</td>
# <td align="left" valign="bottom">Timestamp</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">user_id</td>
# <td align="left" valign="bottom">Unique ID for user</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">product</td>
# <td align="left" valign="bottom">Product ID</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">campaign_id</td>
# <td align="left" valign="bottom">Unique ID for ad campaign</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">webpage_id</td>
# <td align="left" valign="bottom">Webpage ID at which the ad is displayed</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">product_category_1</td>
# <td align="left" valign="bottom">Product category 1 (Ordered)</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">product_category_2</td>
# <td align="left" valign="bottom">Product category 2</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">user_group_id</td>
# <td align="left" valign="bottom">Customer segmentation ID</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">gender</td>
# <td align="left" valign="bottom">Gender of the user</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">age_level</td>
# <td align="left" valign="bottom">Age level of the user</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">user_depth</td>
# <td align="left" valign="bottom">Interaction level of user with the web platform (1 - low, 2 - medium, 3 - High)</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">city_development_index</td>
# <td align="left" valign="bottom">Scaled development index of the residence city</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">var_1</td>
# <td align="left" valign="bottom">Anonymised session feature</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="20">is_click</td>
# <td align="left" valign="bottom">0 - no click, 1 - click</td>
# </tr>
# </tbody>
# </table>
# <p>&nbsp;</p>
# <p>&nbsp;</p>
# <p><strong>Historical User logs</strong></p>
# <table border="1" cellspacing="0"><colgroup width="200"></colgroup><colgroup width="472"></colgroup>
# <tbody>
# <tr>
# <td align="left" valign="bottom" height="19"><strong>Variable</strong></td>
# <td align="left" valign="bottom"><strong>Definition</strong></td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="19">DateTime</td>
# <td align="left" valign="bottom">Timestamp</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="19">user_id</td>
# <td align="left" valign="bottom">Unique ID for the user</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="19">product</td>
# <td align="left" valign="bottom">Product ID</td>
# </tr>
# <tr>
# <td align="left" valign="bottom" height="19">action</td>
# <td align="left" valign="bottom">view/interest (view - viewed the product page, interest - registered interest for the product)</td>
# </tr>
# </tbody>
# </table>

# In[ ]:


train = pd.read_csv("../input/train.csv")
history = pd.read_csv("../input/historical_user_logs.csv", nrows= 40_00_000)
display(train.head())
display(history.head())


# In[ ]:


def datatypes_insight(data):
    display(data.dtypes.to_frame())
    data.dtypes.value_counts().plot(kind="barh")


# In[ ]:


datatypes_insight(train)


# In[ ]:


datatypes_insight(history)


# In[ ]:


train.shape,history.shape


# In[ ]:


df_train = train.merge(history, on = "user_id")


# In[ ]:


y = df_train['is_click']
df_train.drop(['is_click'], axis = 1, inplace = True)


# In[ ]:


df_train.isnull().sum().plot(kind="barh")


# In[ ]:


df_train.columns


# In[ ]:


df_train.drop(['session_id','user_id','product_category_2'], axis=1, inplace=True)
# yourdf.drop(['columnheading1', 'columnheading2'], axis=1, inplace=True)


# In[ ]:


df_train['DateTime_x'] = pd.to_datetime(df_train['DateTime_x'])
df_train['DateTime_y'] = pd.to_datetime(df_train['DateTime_y'])
df_train['DateTime_x'] = pd.to_datetime(df_train['DateTime_x'],format='%d-%m-%Y %H:%M')
df_train['DateTime_y'] = pd.to_datetime(df_train['DateTime_y'],format = '%d-%m-%y %H:%M')


# In[ ]:


df_train['date_x']= df_train['DateTime_x'].dt.date                    
df_train['time_x']= df_train['DateTime_x'].dt.time                   
df_train['year_x']= df_train['DateTime_x'].dt.year                   
df_train['month_x']= df_train['DateTime_x'].dt.month                  
df_train['day_x']= df_train['DateTime_x'].dt.day                    
df_train['hour_x']= df_train['DateTime_x'].dt.hour                   
df_train['minute_x']= df_train['DateTime_x'].dt.minute                 
df_train['second_x']= df_train['DateTime_x'].dt.second                 
df_train['microsecond_x']= df_train['DateTime_x'].dt.microsecond            
df_train['nanosecond_x']= df_train['DateTime_x'].dt.nanosecond             
df_train['week_x']= df_train['DateTime_x'].dt.week                   
df_train['weekofyear_x']= df_train['DateTime_x'].dt.weekofyear             
df_train['dayofweek_x']= df_train['DateTime_x'].dt.dayofweek              
df_train['weekday_x']= df_train['DateTime_x'].dt.weekday                
df_train['dayofyear_x']= df_train['DateTime_x'].dt.dayofyear              
df_train['quarter_x']= df_train['DateTime_x'].dt.quarter                
df_train['is_month_start_x']= df_train['DateTime_x'].dt.is_month_start         
df_train['is_month_end_x']= df_train['DateTime_x'].dt.is_month_end           
df_train['is_quarter_start_x']= df_train['DateTime_x'].dt.is_quarter_start       
df_train['is_quarter_end_x']= df_train['DateTime_x'].dt.is_quarter_end         
df_train['is_year_start_x']= df_train['DateTime_x'].dt.is_year_start          
df_train['is_year_end_x']= df_train['DateTime_x'].dt.is_year_end            
df_train['is_leap_year_x']= df_train['DateTime_x'].dt.is_leap_year           
df_train['daysinmonth_x']= df_train['DateTime_x'].dt.daysinmonth            
df_train['days_in_month_x']= df_train['DateTime_x'].dt.days_in_month          
df_train['tz_x']= df_train['DateTime_x'].dt.tz                     
df_train['freq_x']= df_train['DateTime_x'].dt.freq

df_train['date_y']= df_train['DateTime_y'].dt.date                    
df_train['time_y']= df_train['DateTime_y'].dt.time                   
df_train['year_y']= df_train['DateTime_y'].dt.year                   
df_train['month_y']= df_train['DateTime_y'].dt.month                  
df_train['day_y']= df_train['DateTime_y'].dt.day                    
df_train['hour_y']= df_train['DateTime_y'].dt.hour                   
df_train['minute_y']= df_train['DateTime_y'].dt.minute                 
df_train['second_y']= df_train['DateTime_y'].dt.second                 
df_train['microsecond_y']= df_train['DateTime_y'].dt.microsecond            
df_train['nanosecond_y']= df_train['DateTime_y'].dt.nanosecond             
df_train['week_y']= df_train['DateTime_y'].dt.week                   
df_train['weekofyear_y']= df_train['DateTime_y'].dt.weekofyear             
df_train['dayofweek_y']= df_train['DateTime_y'].dt.dayofweek              
df_train['weekday_y']= df_train['DateTime_y'].dt.weekday                
df_train['dayofyear_y']= df_train['DateTime_y'].dt.dayofyear              
df_train['quarter_y']= df_train['DateTime_y'].dt.quarter                
df_train['is_month_start_y']= df_train['DateTime_y'].dt.is_month_start         
df_train['is_month_end_y']= df_train['DateTime_y'].dt.is_month_end           
df_train['is_quarter_start_y']= df_train['DateTime_y'].dt.is_quarter_start       
df_train['is_quarter_end_y']= df_train['DateTime_y'].dt.is_quarter_end         
df_train['is_year_start_y']= df_train['DateTime_y'].dt.is_year_start          
df_train['is_year_end_y']= df_train['DateTime_y'].dt.is_year_end            
df_train['is_leap_year_y']= df_train['DateTime_y'].dt.is_leap_year           
df_train['daysinmonth_y']= df_train['DateTime_y'].dt.daysinmonth            
df_train['days_in_month_y']= df_train['DateTime_y'].dt.days_in_month          
df_train['tz_y']= df_train['DateTime_y'].dt.tz                     
df_train['freq_y']= df_train['DateTime_y'].dt.freq
df_train.shape


# In[ ]:


def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

df_train = mis_impute(df_train)
# df_train.isnull().sum()


# In[ ]:


df_train.shape


# In[ ]:


y.value_counts().plot(kind="bar")


# In[ ]:


x = df_train
x.shape,y.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[ ]:


get_ipython().run_line_magic('time', '')
from sklearn.naive_bayes import GaussianNB

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(X_train, y_train)
preds = gnb.predict(X_test)
print(preds)
plt.figure(figsize = (20,8))
plt.plot(preds)
plt.plot(y_test)
plt.show()


# In[ ]:




