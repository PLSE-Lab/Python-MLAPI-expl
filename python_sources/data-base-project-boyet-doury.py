#!/usr/bin/env python
# coding: utf-8

# # DATA BASE - PROJECT
# BOYET Benjamin (21304320)
# 
# DOURY Antoine (21302660)

# > # Table of contents
# > ### Loading datasets
# > ## PART 1 : Design and implementation of the database
# > ### 1.1 - Design of the database
# > ### 1.2 - Implementing the database
# > #### 1.2.1 - Create empty datatables
# > #### 1.2.2 Insert data in datatables
# > ## PART 2 : Analyses of the database using SQL
# > ### 2.1 - RAW INFORMATION
# > #### 2.1.1 - Total numbers of rows per datatable
# > #### 2.1.2 - Describing the content of database through queries
# > ### 2.2 ADVANCED INFORMATION
# > #### 2.2.1 Statistics
# > #### 2.2.2 - Finding sets of interesting users
# > #### 2.2.3 Representative news propagation
# 
# 

# ## Loading datasets

# The very first step to this project is to gather all the data needed and that we have at our disposal. From the Kaggle competition, we load the dataset and create Panda dataframe with the function *panda.read_csv*. From the 3 *txt* files at our disposal, we create the dataframe *relation_users* from the *UserUser.txt* file, the dataframe labels_training from the *labels_training.txt* file and finally the dataframe *news_users* from the *newsUser.txt* file. 

# In[ ]:


import numpy as np 
import pandas as pd 
import os

data_dir = '../input/data_competition/data_competition'

relation_users = pd.read_csv(data_dir + "/UserUser.txt", sep="\t", header=None)
relation_users.columns = ["follower", "followed"]

labels_training = pd.read_csv(data_dir + "/labels_training.txt", sep=",")
labels_training.columns = ["news", "label"]

news_users = pd.read_csv(data_dir + "/newsUser.txt", sep="\t", header=None)
news_users.columns = ["news", "user", "times"]


# # PART 1 : Design and implementation of the database

# ## 1.1 - Design of the database
# 
# * **From the project...**
# 
# The goal in this project is to implement a database from the datasets given by the Kaggle competition "Fake News Detection" and to analyze it through the news propagation. We want to use SQL queries to know the inside of the dataset. The dataset gathers news, with an ID, its title and its text. We also have the label, real of fake, for a majority of the news, but not for all, predict the label of those is the objective of the Web Mining Project. We also have informations about users that have shared those news. We have the information when an user has shared a news and how many times. And finally, we have the informations about if an user follows an other user such that we can construct a web.
# 
# Our objectives here is to construct the database gathering all those informations and be able to interrogate the database to find useful informations on the users that are the most actives, the ones with the biggest audience and also the distribution of the label type of the news that those mains users shared, which can show us how those main users behave.
# 
# 
# * **...to a database**
# 
# For the implementation of the database, we respected the steps of the subject of the project. Following the entity-relationship diagram presented below, we created 5 tabless :
# * a table *news* gathering the ID, the title and the text of each news with the news ID as the primary key.
# * a table *news_labels* gathering the ID of each news and its label, that can be real (0), fake (1) or unlaballed (unlabelled)
# * a table *users* with the ID of each user
# * a table *news_propagation* gathering the ID of a news, the ID of a user that has shared this news and the number of times that is shared it
# * a table *followers* giving for an user ID, the ID of the user that he follows
# 
# The exact relational schema is as follows :
# 
# * **NEWS**(<u>newsID</u>, newsTitle, newsText)
# * **NEWS_LABELS**(<u>newsID#</u>, label)
# * **NEWS_PROPAGATION**(<u>newsID#</u>, <u>userID#</u>, propagCount)
# * **USERS**(<u>userID</u>)
# * **FOLLOWERS**(<u>userID#</u>, <u>userID_followed#</u>)
# 
# We can see that the primary key in **news** is the ID of the news. For **users**, it is the ID of the user. And the other keys are then foreign primary keys, which is the case for the table **news_labels** and its primary key the ID of the news that comes from the table **news** or for the table **followers** were there is 2 primary keys, the ID of the user and the ID of the followed that come both from the **users** table.
# 
# Finally, below you can find the entity-relationship model of this database.
# 
# 
# 
# 
# 
# 
# 

#                                                  .

#                                              .

#                                                  .

# ## 1.2 - Implementing the database
# 
# ### 1.2.1 - Create empty datatables
# 
# Now that we have explain the design of the database, we have to implement it. We will code exclusively through the coding language Python and therefore we have to open a connection to a database and sending instructions using a cursor that can send and execute SQL instructions.
# 
# The first step here is to remove all pre-existing tables. Then we create a SQL instruction and execute it trough the *.execute()* function of the cursor. Below, we create the 5 tables indicated in the previous part. We defined all the primary and foreign key and the type of each variable.

# In[ ]:


import sqlite3
conn = sqlite3.connect('test.db')
c = conn.cursor()

### REMOVE EXISTING DATA TABLES
print("Dropping tables")
c.execute("DROP TABLE IF EXISTS users")
c.execute("DROP TABLE IF EXISTS news")
c.execute("DROP TABLE IF EXISTS news_labels")
c.execute("DROP TABLE IF EXISTS news_propagation")
c.execute("DROP TABLE IF EXISTS followers")
print("Creating tables")


# In[ ]:


### users DATA TABLE ##########################################################

sql = '''
CREATE TABLE IF NOT EXISTS users (
 userID INT
 )
'''
c.execute(sql)
print("users DT created")


# In[ ]:


### news DATA TABLE ###########################################################

sql = '''
CREATE TABLE IF NOT EXISTS news (
 newsID INT,
 newsTitle TEXT,
 newsText TEXT,
 PRIMARY KEY (newsID)
 )
'''
c.execute(sql)
print("news DT created")


# In[ ]:


### news_labels DATA TABLE ####################################################

sql = '''
CREATE TABLE IF NOT EXISTS news_labels (
 newsID INT,
 label TEXT,
 PRIMARY KEY (newsID),
 FOREIGN KEY (newsID) REFERENCES news(newsID)
 )
'''
c.execute(sql)
print("news_labels DT created")


# In[ ]:


### news_propagation DATA TABLE ###############################################

sql = '''
CREATE TABLE IF NOT EXISTS news_propagation (
 newsID INT,
 userID INT,
 propagCount INT,
 PRIMARY KEY (newsID, userID),
 FOREIGN KEY (newsID) REFERENCES news(newsID),
 FOREIGN KEY (userID) REFERENCES users(userID)
 )
'''
c.execute(sql)
print("news_propagation DT created")


# In[ ]:


### follower DATA TABLE #######################################################

sql = '''
CREATE TABLE IF NOT EXISTS followers (
 userID INT,
 userID_followed INT,
 PRIMARY KEY (userID, userId_followed),
 FOREIGN KEY (userID) REFERENCES users(userID),
 FOREIGN KEY (userId_followed) REFERENCES users(userID)
 )
'''
c.execute(sql)
print("follower DT created")


# ### 1.2.2 Insert data in datatables
# 
# The second step is now to insert the data in the created tables. It actually exists 2 methods to insert the data from the dataset we had from the *CSV* files. Since we have extract the dataset in *Panda* dataframe, we can either do it through a *for* loop where we would add row by row the data in the table, or we can use the function *to_sql* that insert directly all rows in the table. Both methods are presented below for the table **users**, the *to_sql* method is actually used and the row-by-row method is commented which allows us to see the construction of it. For the rest of the tables, we will use the *to_sql* function since it is faster than the insertion row by row.

# In[ ]:


### INSERT DATA IN THE users DATA TABLES ######################################

list_unique_user_id = pd.DataFrame(list(np.union1d(relation_users.follower, relation_users.followed)))

list_unique_user_id.columns = ["userID"]
list_unique_user_id.to_sql(name='users', con=conn, if_exists='append', index = False)

#for id_user in list_unique_user_id[0:10] :
#    print("inserting data...")
#    c.execute("INSERT INTO users (userID) VALUES (%s)" % id_user)


# Above, in order to create the **users** table, we had to create the list of all the ID of each user. For it, we kept the unique IDs that were in the union of the users ID in the *relation_users* Panda dataframe and the followed users ID of this same dataframe. This way, we had all the IDs of every user implied in this network.
# 
# For the **news** table, we had to gather informations from various sources. Indeed, the news are saved by folder according to their label. Therefore, we have to loop on each file for each folder, *test* and *training*, and get the ID of the news that is in the name of the file, the title of the news which is the first line of the news and finally its text. It gives us to *for* loop which finaly gives us a Panda dataframe that we insert in the **news** table with the function *to_sql*.

# In[ ]:


### INSERT DATA IN THE news DATA TABLES #######################################

list_news = []
for filename in os.listdir(data_dir + "/news/training/"):
    id_news = filename.split(".")[0]
    with open (data_dir + "/news/training/" + filename, "r", encoding="utf8") as myfile:
        news = myfile.readlines()
    news = [x.strip() for x in news] 
    title_news = news[0]
    text_news = ' '.join(news[2:len(news)])
    list_news.append([id_news, title_news, text_news])
    
for filename in os.listdir(data_dir + "/news/test/"):
    id_news = filename.split(".")[0]
    with open (data_dir + "/news/test/" + filename, "r", encoding="utf8") as myfile:
        news = myfile.readlines()
    news = [x.strip() for x in news] 
    title_news = news[0]
    text_news = ' '.join(news[2:len(news)])
    list_news.append([id_news, title_news, text_news])

list_news = pd.DataFrame(list_news)

list_news.columns = ["newsID", "newsTitle", "newsText"]
list_news.to_sql(name='news', con=conn, if_exists='append', index = False)


# In[ ]:


### INSERT DATA IN THE news_labels DATA TABLES ################################

labels_training["label"] = labels_training["label"].astype(str)

for filename in os.listdir(data_dir + "/news/test/"):
    id_news = filename.split(".")[0]
    label_news = 'unlabelled'
    labels_training.loc[len(labels_training.index)+1] = [id_news, label_news]

labels_training = pd.DataFrame(labels_training)

labels_training.columns = ["newsID", "label"]
labels_training.to_sql(name='news_labels', con=conn, if_exists='append', index = False)


# Above, for the **news_labels** table, we have to gather the information from 2 sources. For all the news for which we know the label, the news in the *training* folder, we have the label per news ID already given in the dataframe *label_training*. But for the news in the *test* folder, we have to loop on the folder to get the ID of each news and had the type "unlabelled" through a *for* loop. 
# 
# For the 2 final tables, the data is already well formed. Indeed, for the table **news_propagation**, the dataframe *news_users* already gather all the needed informations. The same is true for the table **follower** where the dataframe *relation_users* gathers the needed variables.

# In[ ]:


### INSERT DATA IN THE news_propagation DATA TABLES ###########################

news_users.columns = ["newsID", "userID", "propagCount"]
news_users.to_sql(name='news_propagation', con=conn, if_exists='append', index = False)


# In[ ]:


### INSERT DATA IN THE follower DATA TABLES ###################################

relation_users.columns = ["userID", "userID_followed"]
relation_users.to_sql(name='followers', con=conn, if_exists='append', index = False)


# Finally, to validate and save our insertions, we have to commit our work through the function *.commit()* of the cursor.

# In[ ]:


conn.commit()


# # PART 2 : Analyses of the database using SQL

# ## 2.1 - RAW INFORMATION
# 
# ### 2.1.1 - Total numbers of rows per datatable
# 
# We want in this first part display the total number of rows for each table. Therefore, since we have to do that for 5 tables, we create an array of SQL instructions and we will execute it trough a Python loop.

# In[ ]:


instruc_nb_rows = {
 "Number of rows of users": 
    ''' 
    SELECT COUNT(userID)
    FROM users 
    ''',
 "Number of rows of news": 
    ''' 
    SELECT COUNT(*) 
    FROM news ''',
 "Number of rows of news_labels": 
    ''' 
    SELECT COUNT(*) 
    FROM news_labels 
    ''',
 "Number of rows of news_propagation": 
    ''' 
    SELECT COUNT(*) 
    FROM news_propagation 
    ''',
 "Number of rows of followers": 
    ''' 
    SELECT COUNT(*) 
    FROM followers 
    '''
 }

instructionSequence = ["Number of rows of users", 
                       "Number of rows of news", 
                       "Number of rows of news_labels", 
                       "Number of rows of news_propagation", 
                       "Number of rows of followers"]
for instruction in instructionSequence:
    c.execute(instruc_nb_rows[instruction])
    print(instruction)
    print(c.fetchone()[0])


# ### 2.1.2 - Describing the content of database through queries
# 
# We now want to do a brief description of the content of the database through SQL queries. As before, we create an array of the queries to execute through a Python loop. 
# 
# * **Number of news per label **
# 
# The first query returns the number of news for each label type, real, fake or unlabelled. It gives us a good idea of the proportion of each category in the database and therefore it allows us to have a better understanding for the following queries. The results witness that there is the same amount of fake news that real news and that a half less of unlabelled news.
# 
# * **Number of news that have been shared at least one time**
# * **Number of users that have shared at least one news**
# 
# This query gives us the number of news that has been shared at least once. As we can see, all the 240 news were share at least once. As for the news, we can see that all users have shared at least once one news since we have the same amount return that the total number of users found in the previous question.
# 
# * **Number of users that follow more than 500 times**
# * **Number of users that follow at least one other user**
# 
# The above queries return respectively the number of users that follow more than 500 other users and the number of users that follow at least one other user. We can see that there is only a few users that follow a really important number of users, 104 users with more than 500 follow. The second query gives an interesting information, indee, it shows that only 17832 users follow at least one person. It means than more than 6000 users do not follow any users, which represents more than 25% of the total population.
# 
# * **Number of users with more than 500 followers**
# * **Number of users that have at least one follower**
# * **Number of users with less than 5 followers**
# 
# Here, we wrote queries related to the number of followers of each user. As previously, we have the number of users with more than 500 followers and can therefore be considered as "influencer" or at least users that have an important audience : they are more than 60 in that case. We can note that there is more users actively following (more than 500 follow) than users actively followed (more than 500 followers), with respectively 104 against 62.
# 
# We also find the number of users with at least one follower, they are 22776. This means that more than 1000 users are not followed by anyone. But we complete this information by getting the number of users followed by less than 5 other users. They are more than 10300 in that case, almost the half of the population, which means than the majority of the population does not have an real audience of more than 5 users on what they share. We saw that only 62 users have an important number of followers of more than 500 followers.

# In[ ]:


instruc_stat_des = {
 "Number of users that have shared at least one news": 
    ''' 
    SELECT COUNT(DISTINCT userID) 
    FROM users 
    ''',
 "Number of news that have been shared at least one time": 
    ''' 
    SELECT COUNT(DISTINCT newsID) 
    FROM news 
    ''',
 "Number of news per label": 
    ''' 
    SELECT label, COUNT(label) 
    FROM news_labels 
    GROUP BY label
    ''',
 "Number of users with more than 500 followers" : 
    ''' 
    SELECT COUNT(DISTINCT userID) 
    FROM users 
    JOIN (
        SELECT * 
        FROM (
            SELECT userID_followed, COUNT(userID_followed) AS nb_follower 
            FROM followers 
            GROUP BY userID_followed
            ) 
        WHERE nb_follower > 500
        ) AS most_followed 
    ON most_followed.userID_followed = users.userID 
    ''',
 "Number of users with less than 5 followers" : 
    ''' 
    SELECT COUNT(DISTINCT userID) 
    FROM users 
    JOIN (
        SELECT * 
        FROM (
            SELECT userID_followed, COUNT(userID_followed) AS nb_follower 
            FROM followers 
            GROUP BY userID_followed
            ) 
        WHERE nb_follower < 6
        ) AS one_followed 
    ON one_followed.userID_followed = users.userID 
    ''',
 "Number of users that follow more than 500 times" : 
    ''' 
    SELECT COUNT(DISTINCT userID) 
    FROM users 
    JOIN (
        SELECT * 
        FROM (
            SELECT userID AS userID0, COUNT(userID) AS nb_follow 
            FROM followers 
            GROUP BY userID0
            ) 
        WHERE nb_follow > 500) AS most_follow 
    ON most_follow.userID0 = users.userID 
    ''',
 "Number of users that follow at least one other user": 
    ''' 
    SELECT COUNT(DISTINCT userID) 
    FROM followers 
    ''',
 "Number of users that have at least one follower": 
    ''' 
    SELECT COUNT(DISTINCT userID_followed) 
    FROM followers 
    '''
 }

instructionSequence = ["Number of news per label",
                       "Number of news that have been shared at least one time",
                       "Number of users that have shared at least one news",
                       "Number of users that follow more than 500 times",
                       "Number of users that follow at least one other user",
                       "Number of users with more than 500 followers",
                       "Number of users that have at least one follower",
                       "Number of users with less than 5 followers"]
for instruction in instructionSequence:
    c.execute(instruc_stat_des[instruction])
    print(instruction)
    if instruction == "Number of news per label" :
        print(c.fetchall())
    else :
        print(c.fetchall()[0][0])


# ## 2.2 ADVANCED INFORMATION
# 
# ### 2.2.1 Statistics
# 
# In this second part of the analysis, we want to analysis more deeply the database and provide readable statistics. We first focus our work on the sharing informations of the database and then focus on the mechanism of following.
# 
# #### Statistics on sharing :
# * **Mean of number of sharing**
# * **Median of number of sharing**
# * **Distribution of number of sharing**
# 
# In a first time, we compute the mean of the variable *propagCount* which represent the number of times a news has been shared by one user. We also compute the median and the total distribution of this variable. We can see that the mean and the median have the same value, 1. The beginning of the distribution helps us to understand why they are equal. Indeed, we can see that there is a important number of time that a news has been shared once by one user, but in a descending importance users have shared many times some news. 
# 
# * **Distribution by label for the news shared by 10 most sharing**
# * **Distribution by label for the news shared by 100 most sharing**
# * **Distribution by label for the 10 news the most shared**
# * **Distribution by label for the 50 news the most shared**
# 
# In a second time, we look at the distribution of label type of the news that have been shared by the users that shared the most, the users the most actives. We returns the distribution for the 10 and the 100 users that share the most and find that, if we saw that there is the same number of fake and real news, the news shared by the most actives users are in majority fake, the ratio between real and fake does not respect the distribution of all news. We can say that the most active users shared more easily fake news (68 real against 83 fake news for the 10 most active users and 411 real against 665 fake ones for the 100 most active users).
# 
# We also look at the distribution of the label type for the news that have been the most shared. Again, we can see that the majority of the most shared news are fake. 5 of the 10 most shared news are fake against 1 real and 30 out of the 50 most shared news are fake against 10 real.
# 

# In[ ]:


### STATISTICS ON SHARING ############################################

instruc_stat_propa = {
 "Mean of number of sharing": 
    ''' 
    SELECT CAST(AVG(propagCount) as int) 
    FROM news_propagation 
    ''',
 "Median of number of sharing": 
    ''' 
    SELECT propagCount 
    FROM news_propagation  
    ORDER BY propagCount  
    LIMIT 1  
    OFFSET (
        SELECT COUNT(*) 
        FROM news_propagation
        ) / 2 
    ''',
 "Distribution of number of sharing": 
    ''' 
    SELECT propagCount, COUNT(*) propa_count 
    FROM news_propagation 
    GROUP BY propagCount 
    ORDER BY propagCount ASC 
    '''
 }

instructionSequence = ["Mean of number of sharing", 
                       "Median of number of sharing", 
                       "Distribution of number of sharing"]
for instruction in instructionSequence:
    c.execute(instruc_stat_propa[instruction])
    print(instruction)
    if instruction == "Distribution of number of sharing" :
        print(c.fetchall()[0:10])
    else :
        print(c.fetchall())


instruc_stat_label = {
 "Distribution by label for the news shared by 10 most sharing": 
    ''' 
    SELECT label, COUNT(label) 
    FROM (
        SELECT * 
        FROM news_labels 
        JOIN (
            SELECT * 
            FROM news_propagation 
            JOIN (
                SELECT userID, SUM(propagCount) AS sum_propa 
                FROM news_propagation 
                GROUP BY userID 
                ORDER BY sum_propa DESC 
                LIMIT 10
                ) AS nb_share_table 
            ON news_propagation.userID = nb_share_table.userID
            ) AS nb_propa_10share 
        ON news_labels.newsID = nb_propa_10share.newsID 
        ) 
    GROUP BY label 
    ''',
 "Distribution by label for the news shared by 100 most sharing": 
    ''' 
        SELECT label, COUNT(label) 
        FROM (
            SELECT * 
            FROM news_labels 
            JOIN (
                SELECT * 
                FROM news_propagation 
                JOIN (
                    SELECT userID, SUM(propagCount) AS sum_propa 
                    FROM news_propagation 
                    GROUP BY userID 
                    ORDER BY sum_propa DESC 
                    LIMIT 100
                    ) AS nb_share_table 
                ON news_propagation.userID = nb_share_table.userID
                ) AS nb_propa_100share 
            ON news_labels.newsID = nb_propa_100share.newsID 
            ) 
        GROUP BY label 
        ''',
 "Distribution by label for the 10 news the most shared": 
    ''' 
    SELECT label, COUNT(label) 
    FROM (
        SELECT * 
        FROM news_labels 
        JOIN (
            SELECT newsID, SUM(propagCount) AS sum_propa 
            FROM news_propagation 
            GROUP BY newsID 
            ORDER BY sum_propa DESC 
            LIMIT 10
            ) AS shared10_news 
        ON news_labels.newsID = shared10_news.newsID 
        ) 
    GROUP BY label 
    ''',
 "Distribution by label for the 50 news the most shared": 
    ''' 
    SELECT label, COUNT(label) 
    FROM (
        SELECT * 
        FROM news_labels 
        JOIN (
            SELECT newsID, SUM(propagCount) AS sum_propa 
            FROM news_propagation 
            GROUP BY newsID 
            ORDER BY sum_propa DESC 
            LIMIT 50
            ) AS shared50_news 
        ON news_labels.newsID = shared50_news.newsID 
        ) 
    GROUP BY label 
    '''
 }

instructionSequence = ["Distribution by label for the news shared by 10 most sharing",
                       "Distribution by label for the news shared by 100 most sharing",
                       "Distribution by label for the 10 news the most shared", 
                       "Distribution by label for the 50 news the most shared"]
for instruction in instructionSequence:
    c.execute(instruc_stat_label[instruction])
    print(instruction)
    print(c.fetchall())


# 
# #### Statistics on following : 
# * **Mean of number of following**
# * **Distribution of number of following**
# * **Mean of number of followed**
# * **Distribution of number of followed**
# 
# In a first time, we compute the mean of the number of following of each user, the number of users that he follows. We also compute the total distribution of this variable. We can see that the mean is equal to 32. We also compute the distribution, the number of user per number of following, which shows that the majority of users are concern for small values of the number of following.
# 
# We did the same analysis for the number of followers, and find a mean number of followers of 25, which shows that, in mean, users follow more that they are followed. The distribution of this number witnesses the same dynamic of the previous one, the majority of the users have a small number of followers.
# 
# 
# * **Distribution by label for the news shared by 100 most followed**
# * **Distribution by label for the news shared by 1000 most followed**
# * **Distribution by label for the news shared by followed by one**
# 
# In a second time, we look at the distribution of label type of the news that have been shared by the users who are the most followed, the users that can be qualify as "influencers". We return the distribution for the 10 and the 100 users that are the most followed and find that the news shared by the "influencers" are in majority fake, the ratio between real and fake does not respect the distribution of all news. We can say that the most active users shared more easily fake news (26 real against 74 fake for the 10 biggest "influencers" and 289 against 751 news for the 100 biggest "influencers"). It means that 75% of the news shared by the "influencers" are fake news.
# 
# We also look at the distribution of the label type for the news that have been shared by the users that have only one follower. Again, we can see that the majority of the news are fake. Only 25% of the known label news were real and 75% were fake news.  

# In[ ]:


### STATISTICS ON FOLLOW ###################################################

instruc_stat_following = {
 "Mean of number of following": 
    ''' 
    SELECT CAST(AVG(nb_follow) as int) 
    FROM (
        SELECT userID, COUNT(userID) AS nb_follow 
        FROM followers 
        GROUP BY userID
        ) 
    ''',
 "Distribution of number of following": 
    ''' 
    SELECT nb_follow, COUNT(*) propa_count 
    FROM (
        SELECT userID, COUNT(userID) AS nb_follow 
        FROM followers 
        GROUP BY userID
        ) 
    GROUP BY nb_follow 
    ORDER BY nb_follow ASC '''
 }

instructionSequence = ["Mean of number of following", 
                       "Distribution of number of following"]
for instruction in instructionSequence:
    c.execute(instruc_stat_following[instruction])
    print(instruction)
    if instruction == "Distribution of number of following" :
        print(c.fetchall()[0:10])
    else :
        print(c.fetchall())


instruc_stat_followed = {
 "Mean of number of followed": 
    ''' 
    SELECT CAST(AVG(nb_followed) as int) 
    FROM (
        SELECT userID, COUNT(userID) AS nb_followed 
        FROM followers 
        GROUP BY userID_followed
        ) 
    ''',
 "Distribution of number of followed": 
    ''' 
    SELECT nb_followed, COUNT(*) propa_count 
    FROM (
        SELECT userID, COUNT(userID_followed) AS nb_followed 
        FROM followers 
        GROUP BY userID_followed
        ) 
    GROUP BY nb_followed 
    ORDER BY nb_followed ASC '''
 }

instructionSequence = ["Mean of number of followed", 
                       "Distribution of number of followed"]
for instruction in instructionSequence:
    c.execute(instruc_stat_followed[instruction])
    print(instruction)
    if instruction == "Distribution of number of followed" :
        print(c.fetchall()[0:10])
    else :
        print(c.fetchall())

        
        ### STATISTICS ON LABELS ######################################################

instruc_stat_label = {
 "Distribution by label for the news shared by 100 most followed": 
    '''
    SELECT label, COUNT(label) 
    FROM (
        SELECT * 
        FROM news_labels 
        JOIN (
            SELECT * 
            FROM news_propagation 
            JOIN (
                SELECT userID_followed, COUNT(userID_followed) AS nb_follower 
                FROM followers 
                GROUP BY userID_followed 
                ORDER BY nb_follower DESC 
                LIMIT 100
                ) AS nb_follow_table 
            ON news_propagation.userID = nb_follow_table.userID_followed
            ) AS nb_propa_100follow 
        ON news_labels.newsID = nb_propa_100follow.newsID 
        ) 
    GROUP BY label 
    ''',
 "Distribution by label for the news shared by 1000 most followed": 
    ''' 
    SELECT label, COUNT(label) 
    FROM (
        SELECT * 
        FROM news_labels 
        JOIN (
            SELECT * 
            FROM news_propagation 
            JOIN (
                SELECT userID_followed, COUNT(userID_followed) AS nb_follower 
                FROM followers 
                GROUP BY userID_followed 
                ORDER BY nb_follower DESC 
                LIMIT 1000
                ) AS nb_follow_table 
            ON news_propagation.userID = nb_follow_table.userID_followed
            ) AS nb_propa_1000follow 
        ON news_labels.newsID = nb_propa_1000follow.newsID 
        ) 
    GROUP BY label 
    ''',
 "Distribution by label for the news shared by followed by one": 
    ''' 
    SELECT label, COUNT(label) 
    FROM (
        SELECT * 
        FROM news_labels 
        JOIN (
            SELECT * 
            FROM news_propagation 
            JOIN (
                SELECT * 
                FROM (
                    SELECT userID_followed, COUNT(userID_followed) AS nb_follower 
                    FROM followers 
                    GROUP BY userID_followed
                    ) 
                WHERE nb_follower = 1
                ) AS nb_follow_table 
            ON news_propagation.userID = nb_follow_table.userID_followed
            ) AS nb_propa_100follow 
        ON news_labels.newsID = nb_propa_100follow.newsID 
        ) 
    GROUP BY label 
    '''
}

instructionSequence = ["Distribution by label for the news shared by 100 most followed",
                       "Distribution by label for the news shared by 1000 most followed", 
                       "Distribution by label for the news shared by followed by one"]

for instruction in instructionSequence:
    c.execute(instruc_stat_label[instruction])
    print(instruction)
    print(c.fetchall())


# ### 2.2.2 - Finding sets of interesting users
# 
# We now want to find in the database sets of "interestings" users. In our case of fake news propagation, we decided to define "interesting" by 3 metrics : 
# 
# * **The 10 users that shared the most**
# 
# The fact that a user shares a lot of news and therefore propose a lot of contents to its followers can influence them and accelerate the propagation process.
# 
# * **The 10 users that are followed the most**
# 
# The fact that a user is followed by a high number of users, means that he can easily influence a large audience by sharing a news.
# 
# * **The 10 users that follow the most**
# 
# The fact that a user follows a lot can also be an interesting metric since in the social network sphere, a lot of "following back" follow happen, which means that a user follow an other user because this last one followed him in the first place. Therefore a user that follows a lot of people is a really active user that can have an important audience.
# 
# 
# Finally, we also display the set of the news the most shared by all the users :
# 
# * **The 10 news that were the most shared**
# 

# In[ ]:


instruc_stat_label = {
 "The 10 users that follow the most": 
    ''' 
    SELECT userID, COUNT(userID) AS nb_follow 
    FROM followers 
    GROUP BY userID 
    ORDER BY nb_follow DESC 
    LIMIT 10 
    ''',
 "The 10 users that are followed the most": 
    ''' 
    SELECT userID_followed, COUNT(userID_followed) AS nb_follower 
    FROM followers 
    GROUP BY userID_followed 
    ORDER BY nb_follower DESC 
    LIMIT 10 
    '''
 }


instructionSequence = ["The 10 users that follow the most", 
                       "The 10 users that are followed the most"]
for instruction in instructionSequence:
 c.execute(instruc_stat_label[instruction])
 print(instruction)
 print(c.fetchall())


instruc_stat_label = {
 "The 10 news that were the most shared": 
    ''' 
    SELECT newsID, SUM(propagCount) AS sum_propa 
    FROM news_propagation 
    GROUP BY newsID 
    ORDER BY sum_propa DESC 
    LIMIT 10 
    ''',
 "The 10 users that shared the most": 
    ''' 
    SELECT userID, SUM(propagCount) AS sum_propa 
    FROM news_propagation 
    GROUP BY userID 
    ORDER BY sum_propa DESC 
    LIMIT 10 
    '''
 }


instructionSequence = ["The 10 news that were the most shared", 
                       "The 10 users that shared the most"]
for instruction in instructionSequence:
    c.execute(instruc_stat_label[instruction])
    print(instruction)
    print(c.fetchall())


# ### 2.2.3 Representative news propagation

# For this final part, we analyze the representative news propagation through the database.
# 
# In order to do it, in a first time, we do hierarchical query on a specific user, the user with *ID* =  13973. We find this user in the previous question, when we were looking for the most active "sharers", the users that share the most. It allows us to get a better understanding of the results. Below, we can find the results which represents the following link between users (we display only the 40 first rows of the results since the entire table is quiet long). The first line shows the user we took as startin point, the user 13973. The 6 next lines are the followers of the user 22, they are 6 users that follow him. And from those 6 followers, we can find the users that follow each of them. For exemple, the next 27 lines are the followers of the user 1761 that himself follows the user 13973. And after we have the followers of the others followers of user 13973, and so on. We choose here to stop the iteration to 2 "generations" of followers just to have a clear view of what can be foud.

# In[ ]:


c.execute('''
    WITH RECURSIVE
      under_part(user, follower, level) AS (
         VALUES('User', '13973', 0)
         UNION
         SELECT followers.userID_followed, followers.userID, under_part.level+1 
             FROM followers, under_part
         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 1
    )
    SELECT SUBSTR('..........',1,level*3) || "(" || user || ", " || follower || ")" 
    FROM under_part
    ''').fetchall()[0:40]


# Then, from what we saw previously, we can compute the number of unique user that can be indirectly influence by an "original user" through the game of following and followers. It shows the web that surrounds a user. Below, we have computed the number of unique users that can be influenced for 2 users randomly picked : user 22 and user 229, and this through 3 generations and 5 generations of followers.
# 
# The results first shows that the number of unique users that have an indirect connection with the user 22 through *n* generations. We can see here that it grows drastically from 3 generations to 5 generations, from 465 indirect connections to more than 15115. For the user 229, it goes from 12916 to 17096. The main difference is that the user 22 only has 3 direct followers when user 229 has 107 direct followers. Therefore the number of indirect relations that the user 229 got in 3 generations is significantly higher than for the user 22 but we can notice that through 5 generations, the number of direct followers barely inbfluence the result anymore.
# 
# In order to touch 70% of the popualtion, 16700 users approximately, we need to go through 6 generations for the user 22. For the user 229, it is only 5 generations. It shows the difference for those 2 users to influence a maximum amount of people by sharing fake news or real ones.

# In[ ]:


instruc_stat_label = {
    "News propagation from user 22 (3 generations)": 
    '''
    WITH RECURSIVE
      under_part(user, follower, level) AS (
         VALUES('?', '22', 0)
         UNION
         SELECT followers.userID_followed, followers.userID, under_part.level+1 
             FROM followers, under_part
         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 2
    )
    SELECT COUNT(DISTINCT follower)
    FROM under_part
    GROUP BY follower
    ''',
    "News propagation from user 22 (5 generations)": 
    '''
    WITH RECURSIVE
      under_part(user, follower, level) AS (
         VALUES('?', '22', 0)
         UNION
         SELECT followers.userID_followed, followers.userID, under_part.level+1 
             FROM followers, under_part
         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 4
    )
    SELECT COUNT(DISTINCT follower)
    FROM under_part
    GROUP BY follower
    ''',
    "News propagation from user 22 (6 generations)": 
    '''
    WITH RECURSIVE
      under_part(user, follower, level) AS (
         VALUES('?', '22', 0)
         UNION
         SELECT followers.userID_followed, followers.userID, under_part.level+1 
             FROM followers, under_part
         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 5
    )
    SELECT COUNT(DISTINCT follower)
    FROM under_part
    GROUP BY follower
    ''',
    "News propagation from user 229 (3 generations)": 
    '''
    WITH RECURSIVE
      under_part(user, follower, level) AS (
         VALUES('?', '229', 0)
         UNION
         SELECT followers.userID_followed, followers.userID, under_part.level+1 
             FROM followers, under_part
         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 2
    )
    SELECT COUNT(DISTINCT follower)
    FROM under_part
    GROUP BY follower
    ''',
    "News propagation from user 229 (5 generations)": 
    '''
    WITH RECURSIVE
      under_part(user, follower, level) AS (
         VALUES('?', '229', 0)
         UNION
         SELECT followers.userID_followed, followers.userID, under_part.level+1 
             FROM followers, under_part
         WHERE followers.userID_followed = under_part.follower AND  under_part.level <= 4
    )
    SELECT COUNT(DISTINCT follower)
    FROM under_part
    GROUP BY follower
    '''
}


instructionSequence = ["News propagation from user 22 (3 generations)", 
                       "News propagation from user 22 (5 generations)", 
                       "News propagation from user 22 (6 generations)", 
                       "News propagation from user 229 (3 generations)",
                       "News propagation from user 229 (5 generations)"]
for instruction in instructionSequence:
    c.execute(instruc_stat_label[instruction])
    print(instruction)
    print(len(c.fetchall()))


# In[ ]:


#close the connection (cursor "c" will be closed as well)
conn.close()
print("inserting task finished")

