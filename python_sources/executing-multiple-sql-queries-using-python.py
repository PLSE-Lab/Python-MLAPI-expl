#!/usr/bin/env python
# coding: utf-8

# **Python Script for Running multiple SQL queries in a single go to hit the database at once**

# **Importing required libraries**

# In[ ]:


import numpy as np 
import pandas as pd 
import sqlite3
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **Creating SQL Connection To Database,if database doesnt exist it will create one.**
# **Used Try and Except to avoid any exceptions**

# In[ ]:


def sql_connection(databaseName):
    try:
        connection=sqlite3.connect(databaseName)
        print('Connection Successfull!!!')
    except:
        print(Error)
    finally:
        #connection.close()
        return connection
        
connection=sql_connection('mydatabase.db')


# **Creating Cursor Object to run SQLite commands**

# In[ ]:


cursor=connection.cursor()


# **Creating SQL Queries Script**
# * To execute all the DDL and DML commands like create,insert,etc use cursor.executescript() to run multiple queries

# In[ ]:


cursor.executescript("""
CREATE TABLE employee( 
firstname text, 
lastname text, 
age integer
);
CREATE TABLE Book( 
title text, 
author text 
);
INSERT INTO 
Book(title, author) 
VALUES ( 
'Dan Clarke''s GFG Detective Agency', 
'Sean Simpsons' 
);
INSERT INTO 
Employee(firstname,lastname,age)
VALUES(
'John',
'Doe',
27
);
""")


# **To run a single SQL query use cursor.execute('query')**
# * To save the contents to database use connection.commit()
# * Executing DQL i.e SELECT statements

# In[ ]:


connection.commit()


# In[ ]:


sql_select_query='SELECT * FROM Employee'
cursor.execute(sql_select_query)


# * For SELECT statements result is returned as a list
# * To view use cursor.fetchall()

# In[ ]:


result=cursor.fetchall()
print(result)

