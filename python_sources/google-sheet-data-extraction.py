#!/usr/bin/env python
# coding: utf-8

# # installing libraries

# In[ ]:


get_ipython().system('pip install gspread oauth2client')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

import gspread
from oauth2client.service_account import ServiceAccountCredentials


# In[ ]:


scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("/kaggle/input/google form-c0a03acfc626.json", scope)
client = gspread.authorize(creds)


# In[ ]:


sheet = client.open('googleform').sheet1


# In[ ]:


data = sheet.get_all_records()


# In[ ]:


data


# # Saving file in pandas dataframe

# In[ ]:


import pandas as ps
import numpy as np


# In[ ]:


df_location = pd.DataFrame()


# In[ ]:


name = []
from_target = []
to = []

for values in data:
    name.append(values['name'])
    from_target.append(values['from'])
    to.append(values['to'])


# In[ ]:


df_location['Name'] = name
df_location['Start location'] = from_target
df_location['Destination location'] = to


# In[ ]:


df_location


# # Get only row or column value or specific cell value

# In[ ]:


data


# **get data of certain row**

# In[ ]:


sheet.row_values(3)


# **get data of certain column**

# In[ ]:


sheet.col_values(3)


# **get data of certain cell**

# In[ ]:


sheet.cell(2,1).value


# # Data Manipulation

# **insert new data**

# In[ ]:


insert_row = ["karan", "delhi", "ktm"]
sheet.insert_row(insert_row, 4)


# In[ ]:


data = sheet.get_all_records()
data


# **delete a row of data**

# In[ ]:


sheet.delete_row(4)


# In[ ]:


data = sheet.get_all_records()
data


# **update a cell.**

# In[ ]:


sheet.update_cell(3,2,"gorkha")


# In[ ]:


data = sheet.get_all_records()
data


# # Reference:
# https://www.youtube.com/watch?v=cnPlKLEGR7E
