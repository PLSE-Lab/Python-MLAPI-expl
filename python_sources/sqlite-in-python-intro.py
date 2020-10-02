#!/usr/bin/env python
# coding: utf-8

# 1. Import libraries

# In[ ]:


import sqlite3
import pandas as pd


# 2. create connection

# In[ ]:


conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')


# 3. Create dataframe

# In[ ]:


df = pd.read_sql(
                       """
                       
                        SELECT *
                        from fires 
                        
                       """, con=conn)


# 4. OR run queries

# In[ ]:


pd.read_sql("""

SELECT *
FROM fires
LIMIT 100

""",con = conn)


# In[ ]:


pd.read_sql("""

SELECT *
FROM fires
WHERE STATE = 'CA'

""",con = conn)


# In[ ]:


pd.read_sql("""

SELECT SOURCE_REPORTING_UNIT_NAME,count(*) as [count]
FROM fires
GROUP BY SOURCE_REPORTING_UNIT_NAME

""",con = conn)


# In[ ]:




