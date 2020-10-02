#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Step by Step tutorial to configure & use MongoDB Atlas with Jupyter Notebook

# ## STEP 1: Register to MongoDB cloud services
# ### [Link to MongoDB Atlas registration page](https://cloud.mongodb.com/user#/atlas/register/accountProfile)
# 
# ![](https://i.imgur.com/GvLXQnh.png)

# ## After successful registration, you will get a nice welcome message
# ![](https://i.imgur.com/PkfXbrO.png)
# 
# ## Just Close it!

# ## STEP 2: Create new cluster
# ### Second step is to create a cluster. You will see a page like this where you need to choose few things.
# ![](https://i.imgur.com/0FSHL97.png)
# 
# ### Good thing is that first cluster is free forever.
# ![](https://i.imgur.com/cbliRzq.png)
# 
# ### Default setting are good enough to work with however you have choices like Cloud Service provider and it's region, however, you can choose only those regions where free tier is available. Try to choose regions where latency is low. You may also rename your cluster.
# 
# ![](https://i.imgur.com/yyyUMdj.png)
# 
# ### Click on create cluster below
# ![](https://i.imgur.com/T4uRjik.png)

# ### After creating cluster you will see this page for a while
# ![](https://i.imgur.com/ejQoiNI.png)
# 
# ### Once clusters are deployed, you will see a page like this
# ![](https://i.imgur.com/KASYKLL.png)

# 
# ## STEP 3: Create Database User
# 
# ### User created in this step with password will be used for authentication of your client to read and write data from/in database. For this step, click on security tab as shown below:
# ![](https://i.imgur.com/lzu1rsT.png)
# 
# ### Click on Add New User Tab
# ![](https://i.imgur.com/KSIDJ8F.png)
# 
# ### Fill Details in the form
# ![](https://i.imgur.com/axYirJz.png)
# 
# ### You can choose user previlages also. You can make more than one user if you need, with different privilages.

# ## STEP 4: WhiteList your IP
# ### Click on IP Whitelist tab
# ![](https://i.imgur.com/Bzk2olW.png)
# 
# ### Click on Add IP Address button
# ![](https://i.imgur.com/2z1knBv.png)
# 
# ### You will be allowed to access database only by whitelisted IP's. If you have static IP, you need to specify that IP as shown below. If you have dynamic IP (for most cases), just click on 'Allow Access from Anywhere' button and click on 'Confirm' Button
# ![](https://i.imgur.com/jHI28Bl.png)
# 
# ### Once it is done, Now you are ready to connect to your cluster from notebook.

# ## STEP 5: Connecting to Cluster via Jupyter Notebook
# ### Click on Connect button as shown below
# ![](https://i.imgur.com/Cs4RlYr.png)
# 
# ### After clicking on the Connect button, following window will open
# ![](https://i.imgur.com/Fp33jwH.png)
# 
# ### In this window you need to select highlighted option i.e. 'Connect Your Application'.
# ### After Clicking on this option following window will open.
# ![](https://i.imgur.com/QXwPhSd.png)
# 
# ### Now, first you should check your driver details to avoid any compatibility issues. Link is given below.
# ### [Check Compatibility](https://docs.mongodb.com/ecosystem/drivers/driver-compatibility-reference/#python-driver-compatibility)
# 
# ### For most of the time 'standard connection string' works. A standard connection string looks like below:
# ### -----------------------------------------------------------------------------------------------------------------------------------------------------
# #### mongodb://my_user_1:[PASSWORD]>/@mycluster-shard-00-00-+++++.mongodb.net:27017,mycluster-shard-00-01-+++++.mongodb.net:27017,mycluster-shard-00-02-+++++.mongodb.net:27017/test?ssl=true&replicaSet=MyCluster-shard-0&authSource=admin&retryWrites=true
# ### -----------------------------------------------------------------------------------------------------------------------------------------------------

# ### Username is already present in string. You just have to change [PASSWORD] with the password made for user in step 3. Now, we are good to go for programming. 

# ### In current tutorial, 'pymongo' library is used to work with MongoDB. This notebok will show you two data insertion methods into DB, making query, updating the database based on query etc. 
# 
# ### Apart from MongoDB, few other things are also illustrated like reading data, making dictionary etc.

# In[ ]:


# Importing libraries

import pymongo
import hashlib
import csv
import prettytable


# > #  Reading data from the csv file

# In[ ]:


# Function to configure reader
def reset_reader():

    f = open('../input/cereal.csv')
    reader = csv.reader(f)

    # Runs for first time
    header = next(reader)

    # Changing name of 'name' column to '_id' which will be used as unique ID for MongoDB
    header[0] = '_id'
    
    return reader, header


# In[ ]:


# Function to convert datatypes of certain items specific to 'cereal.csv' data.
def changedtype(d):
    ftype = range(3,len(header))
    for i in ftype:
        d[i] = float(d[i])
    return d


# ## Function to create md5 hash on one of the most intuitive column in the data

# In[ ]:


# Function to convert string to md5 hash
def change_to_hash(mystring):
    hash_object = hashlib.md5(mystring.encode())
    return hash_object.hexdigest()


# ## Defining Mongo Client, Creating Database and Creating Collection

# In[ ]:


# Making client for pymongo (connection to server)
client = pymongo.MongoClient("mongodb://guest_user_1:wpHI8MjmkWbL9aKr@mongocluster1-shard-00-00-kvch7.mongodb.net:27017,mongocluster1-shard-00-01-kvch7.mongodb.net:27017,mongocluster1-shard-00-02-kvch7.mongodb.net:27017/test?ssl=true&replicaSet=MongoCluster1-shard-0&authSource=admin&retryWrites=true")

# Defining new database, if database already exist (like in this case), that database will be selected.
# 'cereals_and_the_collection' is the name of database. Cereal_db is object.
cereal_db = client['cereals_and_the_collection']

# Defining new collection for cereal_db. Just like database, if collection already exist,will be selected.
# 'Cereal_Data_1' is the name of collection and cereal_collection is object to call.
cereal_collection = cereal_db['Cereal_Data_1']


# ## Pushing data into DB
# ### Data is actually pushed into a collection, not exactly into DB. Each database may contain more than one collection.
# 
# ### Data is pushed in DB by two methods, each row at once or all data at once. Both methods are demonstrated below.
# 
# #### Note: For current user of DB, data writing is prohibited so if you want to run commands below, you need to make your seperate account on MongoDB Atlas Cloud. However, query can be made and executed on already existing database.

# ## Given reference contain various examples of MongoDB operations.
# --------------------------
# ### [MongoDB Tutorials](http://www.w3schools.com/python/python_mongodb_getstarted.asp)
# --------------------------

# ### Pushing each row at a time into DB

# In[ ]:


# reader, header = reset_reader()
# nam = [] # To store original name of product
# hashid = [] # To store hash ids
# flag = 1
# while flag == 1:
#     try:
#         data = next(reader)
#         nam.append(data[0])
#         # Converting first column to corresponding hash value
#         data[0] = change_to_hash(data[0])
        
#         hashid.append(data[0])
        
#         # Changing datatypes of columns
#         data = changedtype(data)
        
#         # Making dictionary to insert into database
#         myData = dict(zip(header,data))
        
#         # Pusing single row in database
#         x = cereal_collection.insert_one(myData)
#     except:
#         break


# ### Pushing all data at once into DB in a new collection

# In[ ]:


# # Resetting reader to work again from start
# reader, header = reset_reader()

# # Defining a new collection in DB
# cereal_collection1 = cereal_db['Cereal_Data_1']

# # Now pushing data at once into DB Collection
# myData_list = [] # A list containing collection of dictionaries to be pushed into DB

# flag = 1
# while flag == 1:
#     try:
#         data = next(reader)
        
#         # Converting first column to corresponding hash value
#         data[0] = change_to_hash(data[0])
        
#         # Changing datatypes of columns
#         data = changedtype(data)
        
#         # Making dictionary to insert into database
#         myData = dict(zip(header,data))
        
#         # Appending dictionary into list
#         myData_list.append(myData)
        
#     except:
#         break
        
# # Pushing data into DB
# x = cereal_collection1.insert_many(myData_list)
# print('{} records were inserted into DB.'.format(len(x.inserted_ids)))


# In[ ]:


# # dictionary to retrieve name of product from hash id
# id_to_nam_dict = dict(zip(hashid,nam))

# # Pushing 'id_to_nam_dict' dictionary into a new collection
# ID_2_NAME = cereal_db['Cereal_ID2NAME']

# # inserting records
# x = ID_2_NAME.insert_one(id_to_nam_dict)


# ## Query to retrieve all the entries from DB where the cups value == 0.33

# In[ ]:


query = {'cups':{'$eq' : 0.33}}
q_data = cereal_collection.find(query)

r,header = reset_reader()
PT = prettytable.PrettyTable()
PT.field_names = header
for i in q_data:
    ilist = []
    for key in header:
        ilist.append(i.get(key))
    PT.add_row(ilist)

print(PT)


# ### Here in queries, these '$eq' and other same notations are Comparison Expression Operators.
# ### [Here](https://docs.mongodb.com/manual/meta/aggregation-quick-reference/#comparison-expression-operators) is the reference.
# 

# ## From the DB update the cups value to 0.75 where the cup value is 0.67

# In[ ]:


query = {'cups':{'$eq' : 0.67}}
newvalues = { "$set": { 'cups': 0.75 } }
cereal_collection.update_many(query,newvalues)


# In[ ]:


query = {'cups':{'$eq' : 0.75}}
q_data = cereal_collection.find(query)

PT = prettytable.PrettyTable()
PT.field_names = header

for i in q_data:
    ilist = []
    for key in header:
        ilist.append(i.get(key))
    PT.add_row(ilist)

print(PT)


# ## Print count of entries where sodium >150 using db query.

# In[ ]:


query = { 'sodium' : { '$gt' : 150 } }
cereal_collection.count_documents(query)


# ## Print count of entries where fat == 1 and shelf == 3

# In[ ]:


query = { '$and' : [{ 'fat' : { '$eq' : 1 } },{ 'shelf' : { '$eq' : 3 } }]}

cereal_collection.count_documents(query)


# In[ ]:




