#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import cassandra
import re
import os
import glob
import numpy as np
import json
import csv


# In[ ]:


print(os.getcwd())

filepath = os.getcwd() + '/event_data'

for root, dirs, files in os.walk(filepath):   
    file_path_list = glob.glob(os.path.join(root,'*'))


# In[ ]:


full_data_rows_list = [] 
     
for f in file_path_list: 
    with open(f, 'r', encoding = 'utf8', newline='') as csvfile:  
        csvreader = csv.reader(csvfile) 
        next(csvreader)
                
        for line in csvreader:
            full_data_rows_list.append(line) 
            
csv.register_dialect('myDialect', quoti
full_data_rows_list = [] 
     
for f in file_path_list: 
    with open(f, 'r', encoding = 'utf8', newline='') as csvfile:  
        csvreader = csv.reader(csvfile) 
        next(csvreader)
        
        for line in csvreader:
            full_data_rows_list.append(line) 
            

csv.register_dialect('myDialect', quoting=csv.QUOTE_ALL, skipinitialspace=True)

with open('event_datafile_new.csv', 'w', encoding = 'utf8', newline='') as f:
    writer = csv.writer(f, dialect='myDialect')
    writer.writerow(['artist','firstName','gender','itemInSession','lastName','length',\
                'level','location','sessionId','song','userId'])
    for row in full_data_rows_list:
        if (row[0] == ''):
            continue
        writer.writerow((row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[12], row[13], row[16]))ng=csv.QUOTE_ALL, skipinitialspace=True)

with open('event_datafile_new.csv', 'w', encoding = 'utf8', newline='') as f:
    writer = csv.writer(f, dialect='myDialect')
    writer.writerow(['artist','firstName','gender','itemInSession','lastName','length',                'level','location','sessionId','song','userId'])
    for row in full_data_rows_list:
        if (row[0] == ''):
            continue
        writer.writerow((row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[12], row[13], row[16]))

with open('event_datafile_new.csv', 'r', encoding = 'utf8') as f:
    print(sum(1 for line in f))


# In[ ]:


from cassandra.cluster import Cluster

try:
    cluster = Cluster()
    session = cluster.connect()
except Exception as e:
    print(e)

try:
    session.execute("""
    CREATE KEYSPACE IF NOT EXISTS test
    WITH REPLICATION = 
    { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }"""
    )
except Exception as e:
    print(e)

try:
    session.set_keyspace('test')
except Execption as e:
    print(e)


# In[ ]:


#First Query

query = "CREATE TABLE IF NOT EXISTS event_datafile1 "
query = query + "(sessionId int, itemInSession int, artist text, song text, length float, PRIMARY KEY(sessionId, itemInSession))"
try:
    session.execute(query)
except Exception as e:
    print(e)
    
file = 'event_datafile_new.csv'

with open(file, encoding = 'utf8') as f:
    csvreader = csv.reader(f)
    next(csvreader) 
    for line in csvreader:
        query = "INSERT INTO event_datafile1 (sessionId, itemInSession, artist, song, length)"
        query = query + " VALUES (%s, %s, %s, %s, %s)"
        session.execute(query, (int(line[8]), int(line[3]), line[0], line[9], float(line[5])))
        
query = "SELECT artist, song, length FROM event_datafile1 WHERE sessionId=338 AND itemInSession=4"
try:
    session.execute(query)
except Exception as e:
    print(e)


# In[ ]:


query = "DROP TABLE IF EXISTS event_datafile1"
try:
    session.execute(query)
except Exception as e:
    print(e)


# In[ ]:


session.shutdown()
cluster.shutdown()


# In[ ]:




