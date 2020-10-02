#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pandas.io.sql as psql
import sqlite3
from sklearn import linear_model
from IPython.display import display, HTML
from datetime import datetime as dt
import time
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import re
import json
import inspect
import requests
import codecs
import platform
import numpy as np
from matplotlib import ticker
get_ipython().run_line_magic('matplotlib', 'inline')
clf = linear_model.LinearRegression()
# http://www.customs.go.jp/toukei/info/tsdl_e.htm
show_tables = "select tbl_name from sqlite_master where type = 'table'"
desc = "PRAGMA table_info([{table}])"

conn_check   =     sqlite3.connect('../input/japantradestatistics2/check_data_1979.db')
check_df = pd.read_sql('select * from check_data',conn_check)

conn_yf_1997   =     sqlite3.connect('../input/japan-trade-statistics/y_1997.db')
yf_1997_df = pd.read_sql('select * from year_from_1997',conn_yf_1997)

#check_df(year,month)-> c_df (year)
sql="""
select Year,exp_imp,sum(Value) as Value
from  check_data
group by Year,exp_imp
"""[1:-1]
c_df = pd.read_sql(sql,conn_check)

#y_1997 check
sql="""
select Year,exp_imp,sum(Value) as Value
from  year_from_1997
group by Year,exp_imp
"""[1:-1]
b = pd.read_sql(sql,conn_yf_1997)

xdf = pd.merge(b,c_df,on=['Year','exp_imp'])
xdf['diff'] = xdf['Value_x']-xdf['Value_y']
xdf


# In[ ]:


"""
memo

conn_ym_2018   = \
    sqlite3.connect('../input/japan-trade-statistics/ym_2018.db')
ym_2018_df = pd.read_sql('select * from ym_2018',conn_ym_2018)

conn_ym_2019   = \
    sqlite3.connect('../input/japan-trade-statistics/ym_2019.db')
ym_2019_df = pd.read_sql('select * from ym_2019',conn_ym_2019)

conn_code   = \
    sqlite3.connect('../input/japantradestatistics2/trade_meta_data.db')
#code_df = pd.read_sql('select * from trade_meta_data',conn_code)
# code 


code_list = ['country_jpn',
             'hs2_jpn',
             'hs2_eng',
             'hs4_eng',
             'hs6_jpn',
             'hs6_eng',
             'hs9_jpn',
             'hs9_eng',
             'country_eng',
             'custom',
             'hs4_jpn']

for t in code_list:
    text = t + "_df=pd.read_sql('select * from " + t + "',conn_code)"
    exec(text)



conn_custom_2018= \
    sqlite3.connect('../input/custom-2016/custom_2018.db')
conn_custom_2019= \
    sqlite3.connect('../input/custom-2016/custom_2019.db')

custom_2018_df = pd.read_sql('select * from custom_2018',conn_custom_2018)
custom_2019_df = pd.read_sql('select * from custom_2019',conn_custom_2019)
conn_customf_2012= \
    sqlite3.connect('../input/custom-2016/from_2012.db')
customf_2012_df = pd.read_sql('select * from from_2012',conn_customf_2012)

class dashboard:
    def __init__(self):
        self.cwd = os.getcwd()
        print(dt.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(platform.system())
%time ds = dashboard()


"""
''



