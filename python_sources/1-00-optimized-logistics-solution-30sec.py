#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


filepath = '/kaggle/input/open-shopee-code-league-logistic/delivery_orders_march.csv'

WORKDAYS = '1111110'
HOLIDAYS = ['2020-03-08','2020-03-25', '2020-03-30', '2020-03-31']

GMT8_OFFSET = 3600 * 8
DURATION_1DAY = 3600 * 24

def mat_to_dict(mat):
    n = len(mat)
    return {i*n+j: mat[i][j] for i in range(n) for j in range(n)}

sla_matrix_1st_attempt = [
    [3, 5, 7, 7],
    [5, 5, 7, 7],
    [7, 7, 7, 7],
    [7, 7, 7, 7],
]
sla_matrix_2nd_attempt = [
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
]
locations = ["Metro Manila", "Luzon", "Visayas", "Mindanao"]
locations = [loc.lower() for loc in locations]
min_length = min(map(len, locations))
trunc_location_to_index = {loc[-min_length:]: i for i, loc in enumerate(locations)}


# In[ ]:


get_ipython().run_cell_magic('time', '', "dtype = {\n    'orderid': np.int64,\n    'pick': np.int64,\n    '1st_deliver_attempt': np.int64,\n    '2nd_deliver_attempt': np.float64,\n    'buyeraddress': np.object,\n    'selleraddress': np.object,\n}\ndf = pd.read_csv(filepath, dtype=dtype)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# convert address to index\ndf['buyeraddress'] = df['buyeraddress'].apply(lambda s: s[-min_length:].lower()).map(trunc_location_to_index)\ndf['selleraddress'] = df['selleraddress'].apply(lambda s: s[-min_length:].lower()).map(trunc_location_to_index)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# convert unix datetime(seconds)stamps to unix datetime(date)stamps\ndt_columns = ['pick', '1st_deliver_attempt', '2nd_deliver_attempt']\ndf[dt_columns[-1]] = df['2nd_deliver_attempt'].fillna(0).astype(np.int64)\ndf[dt_columns] = (df[dt_columns] + GMT8_OFFSET) // DURATION_1DAY")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# compute number of working days between time intervals\nt1 = df['pick'].values.astype('datetime64[D]')\nt2 = df['1st_deliver_attempt'].values.astype('datetime64[D]')\nt3 = df['2nd_deliver_attempt'].values.astype('datetime64[D]')\ndf['num_days1'] = np.busday_count(t1, t2, weekmask=WORKDAYS, holidays=HOLIDAYS)\ndf['num_days2'] = np.busday_count(t2, t3, weekmask=WORKDAYS, holidays=HOLIDAYS)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# compute sla based on addresses\nto_from = df['buyeraddress']*4 + df['selleraddress']\ndf['sla1'] = to_from.map(mat_to_dict(sla_matrix_1st_attempt))\ndf['sla2'] = to_from.map(mat_to_dict(sla_matrix_2nd_attempt))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# compute if deliver is late\ndf['is_late'] = (df['num_days1'] > df['sla1']) | (df['num_days2'] > df['sla2'])\ndf['is_late'] = df['is_late'].astype(int)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# write to file\ndf[['orderid', 'is_late']].to_csv('submission.csv', index=False)")

