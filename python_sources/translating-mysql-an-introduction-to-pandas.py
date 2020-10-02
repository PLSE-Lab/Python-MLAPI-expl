#!/usr/bin/env python
# coding: utf-8

# While getting started with Machine Learning, one of the biggest challenges I faced was understanding how to use pandas. Also, since I was (and still am) a complete beginner to Python, understanding pandas data structures became more difficult.
# 
# While furthering into data science, I felt data needs to be very efficiently handled and manipulated using the pandas framework. Since I knew some basics of MySQL, I thought of learning pandas by translating some of the useful and easy commands in MySQL into pandas Series and DataFrames.
# 
# This kernel can be used as an introduction to the pandas framework for someone who has some knowledge of MySQL commands. Also, it can be used as a refresher to some of the basic scripts for pandas in case someone wants to revise.
# 
# Since the kernel is very basic and I am a complete beginner too, I am expecting quite a number of errors here. Apologies.
# Also, there could be several alternatives for the scripts that I have used for the translation. All suggestions and feedback are most welcome.

#  <h1>Getting started</h1>
# 
# The first requirement is data. All we requires is a simple dataset with few columns required for some basic operations. The most trivial dataset I found was an employee table, the one we all have used in schools.
# I found the dataset used for this kernel at: http://eforexcel.com/wp/downloads-16-sample-csv-files-data-sets-for-testing/
# 
# Also, you may need to install MySQL server on your computer to run the commands in the kernel. For running python commands, you can fork this kernel.
# 
# Just to preview the data, I will be importing this table into a DataFrame. Further explanation for the code will be provided later.

# In[ ]:


import pandas as pd
import numpy as np

ds = pd.read_csv('../input/ds1.csv')
ds.head(5)


# As we can see in the above table, we have 11 columns and the index is based on position. In the next section we will be creating a table in MySQL with the columns specified above.

# <h1>Importing Data</h1>
# 
# Since the data that we have obtained is a CSV file, we can directly import the files into our tables.
# 
# > __MySQL:__
# - Before importing, make sure the the following utility is turned on. MySQL has a security feature that does not allow local files to be imported.
# Check the status of the local file import with the following command:
#         mysql> show global variables like "%infile%";
# 		+---------------+-------+
# 		| Variable_name | Value |
# 		+---------------+-------+
# 		| local_infile  | OFF   |
# 		+---------------+-------+
# 		1 row in set (0.00 sec)
# - Turn on local import with the following command:
# 		mysql> set global local_infile = 'ON';
# - Check status with the above code and make sure it is turned ON
# - MySQL has a utility to import CSVs. Before running the following command, make sure you have created the table in your MySQL database. Also, Note that the filename should be the same as the table name. Run the following commands to create your table:
#         mysql> create database my_db;
#         Query OK, 1 row affected (0.05 sec)
#         mysql> use my_db;
#         Database changed
#         mysql> create table ds (Emp_ID int(8), Name_Prefix varchar(6), First_Name varchar(20), Middle_Initial varchar(4), Last_Name varchar(20), Gender varchar(10), E_Mail varchar(30), Father_Name varchar(40), Mother_Name varchar(40), Mother_Maiden_Name varchar(40), Salary int(15));
#         Query OK, 0 rows affected (0.14 sec)
# Once the table has been created, run the following command in command prompt. Note that this command should not be run in MySQL console.
#         mysqlimport --ignore-lines=1 --fields-terminated-by=, --local -u root -p tester C:\<your local file location>\ds.csv
# 
# >  __Python__:
# - Loading data from a CSV in pandas is easy. All you have to do is run the following command and all the rows along with the column headers will be inserted into a pandas DataFrame.

# In[ ]:


ds = pd.read_csv('../input/ds1.csv')


# > *If you are too anxious and want to look into the data right now, just type 'ds' into the python console. This will display all the data. *

# <h1>Viewing Columns</h1>
# 
# Once the table has been created and values have been inserted, we would like to gain some information regarding the columns.
# 
# > __MySQL:__
#         
#     mysql> describe ds;
#    <img src="https://i.imgur.com/H9xgfVu.png"/>
# 
# > __Python__
# - The following attribute for a DataFrame lists the columns associated:

# In[ ]:


ds.columns


# > However, a describe() method is present for DataFrame which is really helpful for numerical data. It gives few of the really important features about the numerical columns of the table.

# In[ ]:


ds.describe()


# <h1>Peeking into data</h1>
# 
# Let us now look at some ways where we could have a quick look on some of the data within the tables.
# 
# <h2>See all records</h2>
# 
# > __MySQL__
#     
#     mysql> select * from ds; 
# <img src="https://i.imgur.com/9bpBFqk.png"/>
# <img src="https://i.imgur.com/du2o4cP.png"/>
# 
# 
# <br><br>
# > __Python__
# - Just the DataFrame name is all you need!

# In[ ]:


ds

# Or you can use the '[]' notation to display all columns. Uncomment the below line to see results.
# ds[:]


# > - Two very important and interesting features of DataFrame are __iloc__ and __loc__.
# - Basically, there are two sections for iloc/loc. We specify the criteria/conditions for selecting rows on the left side of comma and the criteria for columns on the right side.
# <br><br>
# - With __iloc__, data retrieval and slicing becomes easy. iloc uses integer-based postions for retrieving rows. Hence, you can easily retrieve the 5th row or the last 5 rows.
# - __loc__  uses labels (instead of positions as copmared to iloc) for retrieving rows/columns. If we designate a column as an index of our choice, slicing using loc becomes convineient from the data perspective.

# In[ ]:


# Any of the following scripts will show all data, since we are selecting all rows and all columns

ds.iloc[:,:]

ds.loc[:,:]


# <h2>See specific records from all columns</h2>
# __See first 10 records__
# 
# > __MySQL__
# 
#     mysql> select * from ds limit 10;
#     
#    <img src="https://i.imgur.com/6Z3pxk5.png">

# > __Python__
# - There are multiple ways in which first few rows can be fetched in python. Following are some of them:

# In[ ]:


# Using head() function
ds.head(10)

# Using the [] notation - [] notation is used to either select a range of rows (as in the following script) or to select a specific column
# ds[:10] 
ds[0:10]

# Using iloc - Select rows from 0 to 10
# ds.iloc[:10]
ds.iloc[0:10]


# <h2>See a sample of 10 random records</h2>
# 
# __MySQL__
# 
# >        select * from ds order by rand() limit 10;
# <img src="https://i.imgur.com/fzu4nZz.png">

# >__Python__
# 

# In[ ]:


ds.sample(10)


# <h2>See records from one or more columns</h2>
# 
# >__MySQL__
# 
#         mysql> select E_Mail from ds limit 10;
#         
# <img src="https://i.imgur.com/LfnNj0e.png"/>
# 
#         mysql> select E_Mail, Gender from ds limit 10;
#         
# <img src="https://i.imgur.com/J4uwE4u.png"/>

# >__Python__
# - An important point to note is that '[]' notation is used to retrieve a lower dimension of the data structure. For example, if we have our DataFrame as ds, we can retrieve a lower dimension data structure, i.e., Series using the '[]' notation against the DataFrame.
# 

# In[ ]:


# Retrieving a single column

# Returns a Series for first 10 E Mail addresses
ds['E Mail'][:10]

# Returns a DataFrame for first 10 E Mail addresses
ds[['E Mail']][:10]

# Using iloc 
# Note the method used to retrieve the columns position of 'E Mail'.
# Since the second parameter will only accept positions of the columns, the method was used.
# Returns a DataFrame
ds.iloc[:10,[ds.columns.get_loc('E Mail')]]

# Using loc
# The below method will only work if the index has been set to the positional values of the rows.
# In case it is something else, uncomment the following line to reset the index to positional values.

# ds.reset_index(drop=True, inplace=True)
ds.loc[:10,['E Mail']]


# In[ ]:


# Retrieving multiple columns

# Using iloc - Select the 6th and 5th columns which are 'E Mail' and 'Gender' respectively
ds.iloc[:10,[6,5]]

# Using loc - Select the columns 'E Mail' and 'Gender' by mentioning the label names explicitly
ds.loc[:10,['E Mail', 'Gender']]

# Conversely, we can extract the all the rows by pulling out the required columns from the DataFrame and then select the first 10 rows.
ds[['E Mail', 'Gender']][:10]


# <h1>Add/Remove row(s)</h1>
# 
# >__MySQL__<br><br>
# __Add one row__
#     
#     mysql> insert into ds values(633433, 'Dr.', 'Spider', 'H', 'Man', 'M', 'spider.man@gmail.com', 'Richard Parker', 'Mary Parker', 'Mary', 500000);
#     
# <img src="https://i.imgur.com/XpEOnxQ.png">
# 
# >__Add multiple rows__
#     
#     mysql> insert into ds values
#     (633433, 'Dr.', 'Spider', 'H', 'Man', 'M', 'spider.man@gmail.com', 'Richard Parker', 'Mary Parker', 'Mary', 500000), 
#     (633434, 'Dr.', 'Iron', 'H', 'Man', 'M', 'iron.man@gmail.com', 'Howard Stark', 'Maria Stark', 'Maria', 7500000);
#     
# <img src="https://i.imgur.com/2nIUXAN.png">
# 
# >__Remove one row__
# - Rows can be removed based on a condition. Hence, one or more rows can be deleted based on the condition satisfied. To remove one row, select a column that is unique to the rows.
#         
#         mysql> delete from ds where Emp_ID=633433;
# <img src="https://i.imgur.com/V2Dlkth.png">
#     
# >__Remove multiple rows__
# - Select a condition where the which will apply to rows of your choice.
# 
#         mysql> delete from ds where Name_Prefix="Dr.";
# <img src="https://i.imgur.com/9Uyes1B.png">

# >__Python__<br>
# __Add one row__
# - One simple way to do this is to create a DataFrame for your row and then use append/concat function to add this row. This will also apply to addition of multiple rows.

# In[ ]:


insert_val = pd.DataFrame([['633433', 'Dr.', 'Spider', 'H', 'Man', 'M', 'spider.man@gmail.com', 'Richard Parker', 'Mary Parker', 'Mary', 500000]], columns=ds.columns)
ds = ds.append(insert_val, ignore_index=True)

# The parameter ignore_index=True ensures that the indexing is done automatically. If you have an indexing already defined, make it False.


# >__Add multiple rows__

# In[ ]:


insert_vals = pd.DataFrame([['633433', 'Dr.', 'Spider', 'H', 'Man', 'M', 'spider.man@gmail.com', 'Richard Parker', 'Mary Parker', 'Mary', 500000], [633434, 'Dr.', 'Iron', 'H', 'Man', 'M', 'iron.man@gmail.com', 'Howard Stark', 'Maria Stark', 'Maria', 7500000]], columns=ds.columns)
ds = ds.append(insert_vals, ignore_index=True)


# >We have listed only one way of adding a row. However, there could be multiple ways in which rows can be appended. They will be gradually added to the kernel.

# >__Remove rows__<br><br>
# Rows can be removed from DataFrame very easily if the index of the row to be deleted is known. In case there is only one row to be deleted, put the index of the row in the square brackets. Else, add all the indices of the rows to be deleted.

# In[ ]:


ds.drop([0,1], axis=0, inplace=True)

# axis=0 is for rows, axis=1 is for columns,
# inplace=True to make the change permanent


# >__Conditional deletion__
# - As we saw in the example, we need to provide the indices of the rows to be deleted to use drop. Hence we can use this fact to generate a list of indices for which the condition is true.

# In[ ]:


ds.drop(ds[ds['Middle Initial']=='A'].index, axis=0)


# <h1>Data Manipulation</h1>
# 
# <h2>Updating Values</h2>
# 
# >__MySQL__
# 
#         mysql> update ds set Gender='Male' where Gender='M';
# <img src="https://i.imgur.com/CT82lnV.png">
# 
# >__Python__
# - Using loc, retrieve only the column required along with the condition imposed and set the value required.

# In[ ]:


ds.loc[ds['Gender']=='M',['Gender']]='Male'


# > - For updating in multiple columns, use the following script:

# In[ ]:


ds.loc[ds['Gender']=='Male',['Gender', 'Name Prefix']]='M','Doctor'


# <h2>Mathematical operations</h2>
# 
# <h3>Count</h3>
# - To get a count of the columns/data in the table is important and is required very often.
# 
# 
# >__MySQL__
#     
#     mysql> select count(*) from ds;
# <img src="https://i.imgur.com/aXB890n.png">
# 
# >__Python__

# In[ ]:


ds.count()


# <h3>Numerical Functions and Operations</h3>
# - Following are very few mathematical functions that are provided by both the languages. There exist numerous other functions and operations which one can read about as per interest. Following examples are just for the reader to know the basics.
# 
# >__MySQL__
#         
#         mysql> select sum(Salary) from ds;
# <img src="https://i.imgur.com/SVM40VK.png">
# 
#         mysql> select Salary+25 from ds;
# <img src="https://i.imgur.com/abnJc4X.png">
# <img src="https://i.imgur.com/vrC8d1A.png">
# 
#         mysql> select avg(Salary) as average from ds;
# <img src="https://i.imgur.com/14we8pb.png">
# 

# >__Python__
# 

# In[ ]:


ds['Salary'].astype(int).sum()


# In[ ]:


ds['Salary'].astype(int) + 25


# In[ ]:


average = ds['Salary'].astype(int).mean()
average


# <h1> Finding Distinct Values </h1>
# - MySQL and pandas may have different function names for selecting distinct values but both have efficient ways of retrieving distinct data.
# 
# >__MySQL__
# 
#     mysql>select count(distinct(Gender)) from ds;
# <img src="https://i.imgur.com/LZC2BTk.png">
# 
# >__Python__
# 

# In[ ]:


ds['Gender'].unique()


# > - Following is an interesting translation where we display the distinct items along with their respective counts. 
# 
#         mysql> select Gender,count(Gender) from ds group by Gender;
# <img src="https://i.imgur.com/uytGa7C.png">

# In[ ]:


ds['Gender'].value_counts()


# <h1>Sorting</h1>
# 
# >__MySQL__
# 
#     select * from ds order by Emp_ID desc;
# <img src="https://i.imgur.com/FFUKG2m.png">
# <img src="https://i.imgur.com/oYgBslr.png">
# 
# >__Python__

# In[ ]:


ds.sort_values('Middle Initial', ascending=True)


# This kernel is a work in progress and I shall be keeping it updated with all the knowledge and analogies that I can find.
# I understand that many more features, such as joins, etc. could have been added and will try my best to add them in the next version. So stay tuned! :)
# 
# All suggestions/feedback are most welcome! :)
