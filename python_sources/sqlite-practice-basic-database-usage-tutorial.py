#!/usr/bin/env python
# coding: utf-8

# > <h1><b>TABLE OF CONTENTS</b></h1>
# <ul>
#     <a href='#1'><li>1.What is Database?</li></a>
# </ul>
# <ul>
#     <a href='#2'><li>2.What is SQL?</li></a>
# </ul>
# <ul>
#     <a href='#3'><li>3.Basics of SQL Programming</li></a>
#         <ul>
#              <a href='#4'><li>3.0.Creating a Database</li></a>
#              <a href='#5'><li>3.1.Selecting from Table</li></a>
#              <a href='#6'><li>3.2.Updating a Table</li></a>
#              <a href='#7'><li>3.3.Deleting from Table</li></a>
#              <a href='#8'><li>3.4.Qualified Table</li></a>
#         </ul>
# </ul>
# <ul>
#     <a href='#9'><li>4.Extra Commands to Program SQL</li></a>
#         <ul>
#              <a href='#10'><li>4.0.Rollback</li></a>
#              <a href='#11'><li>4.1.Order By</li></a>
#              <a href='#12'><li>4.2.Select Distinct</li></a>
#              <a href='#13'><li>4.3.Limit</li></a>
#              <a href='#14'><li>4.4.Between</li></a>
#              <a href='#15'><li>4.5.In</li></a>
#             <a href='#16'><li>4.6.Like</li></a>
#             <a href='#17'><li>4.7.Glob</li></a>
#             <a href='#18'><li>4.8.Group By</li></a>
#             <a href='#19'><li>4.9.Having</li></a>
#             <a href='#20'><li>4.10.Union</li></a>
#             <a href='#21'><li>4.11.Except</li></a>
#             <a href='#22'><li>4.12.Case</li></a>
#             <a href='#23'><li>4.13.Replace</li></a>
#         </ul>
# </ul>
# <ul>
#     <a href='#24'><li>5.Conclusion</li></a>
#     <a href='#25'><li>6.References</li></a>
# </ul>

# <p id='1'><h2><b>1.What is Database?</b></h2></p>

# Database is a systematic collection of data. Databases support storage and  manipulation of data. Databases make data management easy. Let's discuss few examples.
# 
# An online telephone directory would definitely use database to store data pertaining to people, phone numbers, other contact details, etc.
# 
# Your electricity service provider is obviously using a database to manage billing , client related issues, to handle fault data, etc.
# 
# Let's also consider the facebook. It needs to store, manipulate and present data related to members, their friends, member activities, messages, advertisements and lot more.
# 
# We can provide countless number of examples for usage of databases .

# <p id='2'><h2><b>2.What is SQL?</b></h2></p>

# Structured Query language (SQL) pronounced as "S-Q-L" or sometimes as "See-Quel"is actually the standard language for dealing with Relational Databases.
# 
# SQL programming can be effectively used to insert, search, update, delete database records.
# 
# That doesn't mean SQL cannot do things beyond that.
# 
# In fact it can do lot of things including, but not limited to, optimizing and maintenance of databases. 
# 
# Relational databases like MySQL Database, Oracle, Ms SQL server, Sybase, etc uses SQL ! How to use sql syntaxes?
# 
#  SQL syntaxes used in these databases are almost similar, except the fact that some are using few different syntaxes and even proprietary SQL syntaxes.
#  
#  In this kernel we will use SQLite.

# <p id='3'><h2><b>3.Basics of SQL Programming</b></h2></p>

# <p id='4'><h3><b>3.0.Creating a Database</b></h3></p>
# In this part of Basics of SQL Programming, we will learn how to create(connect to) a database, using of cursor, creating a table in database, inserting values to table, saving and closing a database.

# Importing libraries.

# In[ ]:


import bq_helper
import numpy as np
import pandas as pd
import sqlite3 as sql 


# Here we will create a database named 'bookshelf.sqlite' sqlite is an extension.  
# * .connect() method to connect a database or if the database is not exist then the method creates a new database,  
# * .cursor() method to process on a database.

# In[ ]:


db = sql.connect("bookshelf.sqlite")
cursor = db.cursor()


# * .execute() method to do actions on SQL.
# * CREATE TABLE IF NOT EXISTS "name of table" (column1,column2,...)  here we use IF NOT EXISTS not to get an error.

# In[ ]:


cursor.execute("PRAGMA busy_timeout = 30000")
cursor.execute("CREATE TABLE IF NOT EXISTS book_information (name,author,readornot,rate)")


# * INSERT INTO "name of table" VALUES ("value of column1","value of column2",...),
# * When we execute cursor.execute(book1) then book1 is saved to table,
# * .commit() method to save the information on memory to database,
# * .close() method to close database. If we don't close, maybe we can get some errors when we use database again.

# In[ ]:


book1 = "INSERT INTO book_information VALUES ('Crime and Punishment','Dostoyevski','Yes','*****')"
book2 = "INSERT INTO book_information VALUES ('White Fang','Jack London','Yes','***')"
cursor.execute(book1)
cursor.execute(book2)
db.commit()
db.close()


# <p id='5'><h3><b>3.1.Selecting from Table</b></h3></p>

# We import os to look if there is a folder or not.  
# os.path.exists("database name") 

# In[ ]:


import os 

database = "bookshelf.sqlite"
folder_exists = os.path.exists(database)


# In[ ]:


db = sql.connect("bookshelf.sqlite")
cursor = db.cursor()


# * SELECT * FROM "name of table" :  here " star** " means all in table,
# * .fetchall() method to read everything in the table and to write to a list that we defined.

# In[ ]:


cursor.execute("SELECT * FROM book_information")
books = cursor.fetchall() # books is a list.
print(books)
for i in books:
    print(i)
    #for k in i:
    #    print(k,end=" ")
    #print("")


# Here we add one more row and save it to the database.

# In[ ]:


cursor.execute("INSERT INTO book_information VALUES ('Greek Mythology','Anna and Louie','Yes','****')")
db.commit()


# In[ ]:


cursor.execute("SELECT * FROM book_information")
books = cursor.fetchall()
print(books)
for i in books:
    print(i)


# <p id='6'><h3><b>3.2.Updating a Table</b></h3></p>

# * UPDATE "name of table" SET "the value we want to change to" WHERE "selecting the value we want to change".

# In[ ]:


cursor.execute("UPDATE book_information SET rate='****' WHERE rate='***'") # where rate is 3 stars
                                                                        # makes them 4 stars
db.commit()
cursor.execute("UPDATE book_information SET readornot='No' WHERE rate ='****'")#if rate equals 3 stars 
                                                                        #make read status no.
db.commit()
cursor.execute("SELECT * FROM book_information")
books = cursor.fetchall()
for i in books:
    print(i)


# <p id='7'><h3><b>3.3.Deleting from Table</b></h3></p>

# * DELETE FROM "name of table" WHERE "the value we want to delete"

# In[ ]:


cursor.execute("DELETE FROM book_information WHERE rate='****'")
db.commit()
cursor.execute("SELECT * FROM book_information")
books = cursor.fetchall()
for i in books:
    print(i)


# <p id='8'><h3><b>3.4.Qualified Table</b></h3></p>

# I will explain you with names of columns. For example ; 
# * book_id is an id that contains integer numbers and these integer numbers increment automatically for all the other columns.

# In[ ]:


cursor.execute("CREATE TABLE IF NOT EXISTS special (book_id INTEGER PRIMARY KEY  AUTOINCREMENT,book_name,author,readornot,rate)")
cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('Greek Mythology','Anna and Louie','Yes','****')")
cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('White Fang','Jack London','Yes','***')")
db.commit()
cursor.execute("SELECT * FROM special")
books = cursor.fetchall()
for i in books:
    print(i)

db.close()


# <p id='9'><h2><b>4.Extra Commands to Program SQL</b></h2></p>

# <p id='10'><h3><b>4.0.Rollback</b></h3></p>
# * .rollback() method is used to rolls back any changes to the database since the last call to commit.

# In[ ]:


db = sql.connect("bookshelf.sqlite")
cursor = db.cursor()
db.rollback()
cursor.execute("SELECT * FROM book_information")
cursor.fetchall()


# <p id='11'><h3><b>4.1.Order By</b></h3></p>
# * ORDER BY "name of sorted by value" DESC/ASC 
# * DESC : Descending orders.
# * ASC : Ascending orders (Default).

# In[ ]:


cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('Greek Mythology','Anna and Louie','Yes','****')")
cursor.execute("INSERT INTO special (book_name,author,readornot,rate) VALUES ('White Fang','Jack London','Yes','***')")
cursor.execute("SELECT * FROM special ORDER BY book_id DESC")
cursor.fetchall()


# <p id='12'><h3><b>4.2.Select Distinct</b></h3></p>
# * Removes the duplicate rows in the result set.

# In[ ]:


cursor.execute("SELECT DISTINCT book_name,author,readornot,rate FROM special")
cursor.fetchall()


# <p id='13'><h3><b>4.3.Limit</b></h3></p>
# * Limit method, defines that how many rows you want to select.

# In[ ]:


cursor.execute("SELECT book_name,author,readornot,rate FROM special LIMIT 3")
cursor.fetchall()


# <p id='14'><h3><b>4.4.Between</b></h3></p>
# * Between is a logical operator that tests whether a value is in range of values.

# In[ ]:


cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE book_id BETWEEN 1 AND 4 ")
cursor.fetchall()
#as you see it contains 1 and 4 


# <p id='15'><h3><b>4.5.In</b></h3></p>
# * Depending on value of the thing that depend on WHERE method In selects that rows.
# * If we use IN() with a null set so it turns nothing.

# In[ ]:


#cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE book_id IN ()")
# Nothing :)
cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE book_id IN (1,2,3,4,5,6,7)")
cursor.fetchall()


# <p id='16'><h3><b>4.6.Like</b></h3></p>
# * There are two ways to construct a pattern using % (percent sign) and _ (underscore) wildcards:
# 
# * The (percent sign) % wildcard matches any sequence of zero or more characters.
# * The (underscore) _ wildcard matches any single character.

# In[ ]:


cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE author LIKE 'Anna%'")
cursor.fetchall()


# <p id='17'><h3><b>4.7.Glob</b></h3></p>
# * The asterisk (*) wildcard matches any number of characters.
# * The question mark (?) wildcard matches exactly one character.
# * Start with "sthg*" end with "*sthg" starts "?nna*" any characters goes with "nna" 
# * Which one contains a > '*a*' which one contains 1 to 9 '*[1-9]*' doesnt contains any numbers '*[^1-9]*'
# * Whose name ends with number '*[1-9]'

# In[ ]:


cursor.execute("SELECT book_name,author,readornot,rate FROM special WHERE author GLOB '*A*'")
cursor.fetchall()


# <p id='18'><h3><b>4.8.Group By</b></h3></p>
# * Groups the columns so we can select max book id depending on unique book names

# In[ ]:


cursor.execute("SELECT MAX(book_id) book_name,author,readornot,rate FROM special GROUP BY book_name")
cursor.fetchall()


# <p id='19'><h3><b>4.9.Having</b></h3></p>
# * Filters the groups
# * You often use the HAVING clause with the GROUP BY clause. The GROUP BY clause groups a set of rows into a set of summary rows or groups. Then the HAVING clause filters groups based on specified conditions.
# * If you use a HAVING clause without the GROUP BY clause, the HAVING clause behaves like a WHERE clause.

# In[ ]:


cursor.execute("SELECT book_id,book_name,author,readornot,rate FROM special GROUP BY book_id HAVING book_id=3")
cursor.fetchall()


# <p id='20'><h3><b>4.10.Union</b></h3></p>
# * If we use union don't write same numbers but if union all then it writes all the numbers.

# In[ ]:


cursor.execute("SELECT book_id FROM special UNION ALL SELECT book_id FROM special")
cursor.fetchall()


# <p id='21'><h3><b>4.11.Except</b></h3></p>
# * As you can see left side has 1 2 (3 4 5) the other side (3 4 5 6 7 8..)
# * So output is 1 and 2.

# In[ ]:


cursor.execute("SELECT book_id FROM special WHERE book_id GLOB '[1-5]' EXCEPT SELECT book_id FROM special WHERE book_id GLOB '[3-8]'")
cursor.fetchall()


# <p id='22'><h3><b>4.12.Case</b></h3></p>
# * It looks "readornot" column, if the value is "Yes" then it write "Read" for this row in the new column that we created "ReadedoRnot".
# * I think it is more complicated then the others and maybe I couldn't explain well. So if you have a question you can ask me.

# In[ ]:


cursor.execute("SELECT * ,CASE readornot WHEN 'Yes' THEN 'Read' ELSE 'NotRead' END 'ReadedoRnot' FROM special")
cursor.fetchall()


# <p id='23'><h3><b>4.13.Replace</b></h3></p>
# * Adding new row to table.

# In[ ]:


cursor.execute("REPLACE INTO special (book_name,author,readornot,rate) VALUES ('ABC','ABCD','No','*' )")
cursor.execute("SELECT * FROM special")
cursor.fetchall()


# <p id='24'><h2><b>5.Conclusion</b></h2></p>
# In this kernel we have looked : 
# * What a database and SQL is,
# * How we create, select, delete and update a table,
# * Extra methods to program SQL.
# * If you like, please upvote! 

# <p id='25'><h2><b>6.References</b></h2></p>
# 
# https://www.guru99.com/introduction-to-database-sql.html#3
# 
# http://www.sqlitetutorial.net
# 
# https://docs.python.org/3.4/library/sqlite3.html
# 
# https://www.youtube.com/watch?v=nb0zQ-WEeV8 
