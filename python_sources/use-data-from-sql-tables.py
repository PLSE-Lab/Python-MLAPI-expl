#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Useful manual here: https://datatofish.com/how-to-connect-python-to-sql-server-using-pyodbc/
# You will need installed and configured ODBC driver on your computer!


# In[ ]:


import pandas as pd
import pyodbc


# In[ ]:


# Before launching, change the server name and table name to your!
# You must also specify the username and password for connecting to the database in the parameters, if necessary.


# In[ ]:


Server = 'Server_Name\SQL_Domain_Name'
DB = 'Data Science'
connection = pyodbc.connect('DRIVER={SQL Server}; SERVER=' + Server + '; DATABASE=' + DB + '; Trusted_Connection=yes')


# In[ ]:


query_SQLTable = "SELECT * FROM SQLTable_Name"
SQLTable = pd.read_sql(query_SQLTable, connection)
SQLTable.head()


# In[ ]:


# Perhaps this code will not run on Kaggle due to the specialty of its work,
# then you will need to download this code and run it on your computer.

