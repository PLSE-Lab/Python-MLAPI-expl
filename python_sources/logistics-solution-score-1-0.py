#!/usr/bin/env python
# coding: utf-8

# # Shopee Code League 2020 - Logistics Solution Score 1.0
# ***9min runtime***

# In[ ]:


import numpy as np
import pandas as pd
import time
import pytz

from collections import defaultdict
from datetime import datetime, timedelta, date


pd.set_option('display.max_colwidth', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# brutally imported the matrix
sla_mat = pd.read_excel('/kaggle/input/logistics-shopee-code-league/SLA_matrix.xlsx',index_col=1, skiprows=[0], nrows=4)
sla_mat = sla_mat.drop(columns=['Unnamed: 0'])

mat = defaultdict(lambda: defaultdict(int)) #defaultdict of defaultdict of integers
for x, y in sla_mat.iteritems():
    for ii, jj in y.iteritems():
        mat[x][ii] = timedelta(days= int(jj[0]))


# In[ ]:


df = pd.read_csv('/kaggle/input/logistics-shopee-code-league/delivery_orders_march.csv')

# preprocess the address to match the dictionary's keys
df['buyeraddress'] = df['buyeraddress'].map(lambda x : x.split()[-1].title() if x.split()[-1].lower()!='manila' else ' '.join(x.split()[-2:]).title())
df['selleraddress'] = df['selleraddress'].map(lambda x : x.split()[-1].title() if x.split()[-1].lower()!='manila' else ' '.join(x.split()[-2:]).title())
df['2nd_deliver_attempt'] = df['2nd_deliver_attempt'].fillna(0)
df


# In[ ]:


ph = ['2020-03-25','2020-03-30','2020-03-31'] # removed '2020-03-08' as it's Sunday~
ph = [datetime.strptime(x, '%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0) for x in ph]
sunday = 6
islate = []

def calHolidays(date1, date2):
    date1_ord = date1.toordinal()
    date2_ord = date2.toordinal()
    cnt = 0

    for d_ord in range(date1_ord, date2_ord):
        d = date.fromordinal(d_ord)
        if (d.weekday() == sunday):
            cnt += 1
            
    cnt += sum([date1<=x<=date2 for x in ph])
    return timedelta(days=cnt)

for index, cols in df.iterrows():    
    late = 0
    pick = datetime.fromtimestamp(cols['pick']).replace(hour=0, minute=0, second=0, microsecond=0)
    first_att = datetime.fromtimestamp(cols['1st_deliver_attempt']).replace(hour=0, minute=0, second=0, microsecond=0)
    sec_att = datetime.fromtimestamp(cols['2nd_deliver_attempt']).replace(hour=0, minute=0, second=0, microsecond=0) if cols['2nd_deliver_attempt'] else 0
    
    days = first_att - pick - calHolidays(pick,first_att)
    if days > mat[cols['selleraddress']][cols['buyeraddress']]:
        late = 1

    if not late and sec_att:
        interval = sec_att - first_att - calHolidays(first_att,sec_att)
        if interval > timedelta(days=3):
            late = 1
                
    islate.append((cols['orderid'],late))
    
res = pd.DataFrame(islate, columns=('orderid','is_late'))


# In[ ]:


res.to_csv('result.csv', index = False)
print(res)

