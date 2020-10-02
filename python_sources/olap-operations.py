#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from subprocess import check_output
print(check_output(['ls','/kaggle/input']).decode('utf8'))


# # get the data of axisbank.csv from https://www.kaggle.com/rohanrao/nifty50-stock-market-data and add year month and day columns from the date column?

# In[ ]:


import pandas as pd
abanks=pd.read_csv(r'/kaggle/input/AXISBANK.csv')
abanks.head(5)


# In[ ]:


pdate=pd.to_datetime(abanks['Date'])


# In[ ]:


abanks['year']=pdate.dt.year
abanks['month']=pdate.dt.month
abanks['Day']=pdate.dt.day


# In[ ]:


abanks.head(5)


# ## 1) pip install atoti,start session and create cube to the data?

# In[ ]:


get_ipython().system('pip install atoti')


# In[ ]:


import atoti as tt
session = tt.create_session()


# In[ ]:


bankstore = session.read_pandas(abanks,store_name='banker')


# In[ ]:


cube = session.create_cube(bankstore)


# ## 2) add hireracy of date into cube and also delete unwanted dimensions(or manual creation of cube)?

# In[ ]:


h = cube.hierarchies
lvl = cube.levels
m = cube.measures
h


# In[ ]:


del h['Date']
del h['Day']
del h['Volume']
del h['year']


# In[ ]:


del h['month']


# In[ ]:


cube.hierarchies["Date"] = [
    bankstore["year"],
    bankstore["month"],
    bankstore["Day"],
    bankstore["Date"]
]


# ## 3) aggregate the mean of close column and number of rows of the data?

# In[ ]:


cube.query(m['contributors.COUNT'],m['Close.MEAN'])


# ## 4) roll up to year,month and day with sum of turnovers?

# In[ ]:


cube.query(m['Turnover.SUM'],levels=lvl['year'])


# In[ ]:


cube.query(m['Turnover.SUM'],levels=lvl['month'])


# In[ ]:


cube.query(m['Turnover.SUM'],levels=lvl['Day'])


# ## 5) drill down to symbol column and print the open sum?

# In[ ]:


cube.query(m['Open.SUM'],levels=lvl['Symbol'])


# ## 6) slice the data with only months of 2003 with vmap sum and count for each day of month?

# In[ ]:


cube.query(m["contributors.COUNT"],m["VWAP.SUM"],levels=[lvl['year'],lvl['month']],condition=lvl['year']=='2003')


# In[ ]:





# In[ ]:




