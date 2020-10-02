#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime
import numpy as np
import pytz


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from tqdm import tqdm
tqdm.pandas()


# In[ ]:


df = pd.read_csv('../input/logistics-shopee-code-league/delivery_orders_march.csv')


# In[ ]:


matrix_dict = {
    "metro manila" : {
        "metro manila" : 3,
        "luzon" : 5,
        "visayas": 7,
        "mindanao": 7
        },
    "luzon" : {
        "metro manila" : 5,
        "luzon" : 5,
        "visayas": 7,
        "mindanao": 7
        },
    "visayas" : {
        "metro manila" : 7,
        "luzon" : 7,
        "visayas": 7,
        "mindanao": 7
        },
    "mindanao" : {
        "metro manila" : 7,
        "luzon" : 7,
        "visayas": 7,
        "mindanao": 7
        }   
}


# In[ ]:


from pandas.tseries.offsets import CustomBusinessDay

bday = CustomBusinessDay(weekmask='Mon Tue Wed Thu Fri Sat', holidays=['2020-03-08','2020-03-25','2020-03-30','2020-03-31'])


# In[ ]:


def calculate_diff_1(x):
    return len(pd.date_range(start=datetime.datetime.fromtimestamp(x['pick'],tz=pytz.timezone("Singapore")).date(), end=datetime.datetime.fromtimestamp(x['1st_deliver_attempt'],tz=pytz.timezone("Singapore")).date(), freq= bday)) -1


def calculate_diff_2(x):
    try:
        return len(pd.date_range(start=datetime.datetime.fromtimestamp(x['1st_deliver_attempt'],tz=pytz.timezone("Singapore")).date(), end=datetime.datetime.fromtimestamp(x['2nd_deliver_attempt'],tz=pytz.timezone("Singapore")).date(), freq= bday)) -1
    except:
        return 0


# In[ ]:


def determine_if_late(x):
    if x.second_count > 3:
        return 1
    elif x.first_count > get_sla(x):
        return 1
    else:
        return 0


# In[ ]:


def get_sla(x):
    for buyer_location in matrix_dict.keys():
        if np.char.endswith(x.buyeraddress.lower(), buyer_location):
            for seller_location in matrix_dict.keys():
                if np.char.endswith(x.selleraddress.lower(), seller_location):
                    return matrix_dict[buyer_location][seller_location]
    print("Error!")


# In[ ]:


def checkiflate(sample_df):
    sample_df['first_count'] = sample_df.progress_apply(lambda x: calculate_diff_1(x),axis=1)
    sample_df['second_count'] = sample_df.progress_apply(lambda x: calculate_diff_2(x),axis=1)
    sample_df['is_late'] = sample_df.progress_apply(lambda x: determine_if_late(x),axis=1)
    return sample_df


# In[ ]:


new_df = checkiflate(df)


# In[ ]:


submission = new_df[['orderid','is_late']]


# In[ ]:


submission.to_csv('submission.csv',index=False)

