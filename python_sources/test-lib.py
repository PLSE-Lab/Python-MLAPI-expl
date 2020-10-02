#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
'''
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT p.Name Sender,
       e.MetadataSubject Subject
FROM Emails e
INNER JOIN Persons p ON e.SenderPersonId=p.Id
LIMIT 10""", con)
print(sample)
'''
# You can read a CSV file like this
emails = pd.read_csv("../input/Emails.csv")
persons=pd.read_csv("../input/Persons.csv")
emailReceivers=pd.read_csv("../input/EmailReceivers.csv")



#print(emails.head())

# It's yours to take from here!


# In[ ]:


#Drop Rows with MetadataTo and MetadataFrom in Emails database NaN or empty 
import numpy as np
emails = emails[pd.notnull(emails['MetadataTo'])]
emails=emails[pd.notnull(emails['MetadataFrom'])]
emails.head()


# In[ ]:


#get the Sender given the SenderPersonId
def getSender(SenderId):
    print(persons.loc[persons['Id'] == SenderId])


# In[ ]:


head(persons)


