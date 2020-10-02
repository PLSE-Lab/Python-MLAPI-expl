#!/usr/bin/env python
# coding: utf-8

# **Hi There!** I'am going to use the Neo4j graph database [based on this](https://neo4j.com/use-cases/real-time-recommendation-engine/), on my local computer and post here the neccesary codes, to import the CSV to the 
# databased and use it for the "who can answer these questions" recommendation. Maybe even pictures from the database, because I love graph visualisations. :D

# Please remember to **upvote** if you find the work useful! Thank you for visiting.

# **First download the [data](https://www.kaggle.com/c/data-science-for-good-careervillage/data) to your computer from kaggle to a separated folder on your local computer or server** (I will be going with local computer because this is a proof of concept not a production ready thing.)

# **[Install](https://neo4j.com/docs/operations-manual/current/installation/) Neo4j on you local computer** Yes it is open source ;)

# * **Start a project in Neo4j**
# * **Start a graph database in the project**
# * **After the grapd db had been spinned up, start the Neo4j browser**
# * Optional: **Copy this command into the Neo4j browser: "*play northwind-graph*"** to see how to import a relational database into a graph database
# * Optional: **Or read it [here](https://neo4j.com/developer/guide-importing-data-and-etl/)**
# I'm going to do the import based on these 2 nothing magical

# **So we had set up the graph db** 
# *Let's see what we have here.* (I used [this](https://www.kaggle.com/anu0012/quick-start-eda-careervillage-org/notebook) kernel to speed up the data analysis)

# Importing the python libraries

# In[ ]:


import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(color_codes=True)
from wordcloud import WordCloud,STOPWORDS
import warnings
warnings.filterwarnings('ignore')
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# Importing all the CSVs

# In[ ]:


answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')
comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')
emails = pd.read_csv("../input/data-science-for-good-careervillage/emails.csv")
group_memberships = pd.read_csv('../input/data-science-for-good-careervillage/group_memberships.csv')
groups = pd.read_csv('../input/data-science-for-good-careervillage/groups.csv')
matches = pd.read_csv('../input/data-science-for-good-careervillage/matches.csv')
professionals = pd.read_csv("../input/data-science-for-good-careervillage/professionals.csv")
questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')
school_memberships = pd.read_csv('../input/data-science-for-good-careervillage/school_memberships.csv')
students = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')
tag_questions = pd.read_csv("../input/data-science-for-good-careervillage/tag_questions.csv")
tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')
tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')


# Okay that's set, now I map the relational db schema to speed up the import and the understanding of the whole space with [this](https://dbdiagram.io/) free tool. using the head of all of the dataframes and the explanation given in the data [description](https://www.kaggle.com/c/data-science-for-good-careervillage/data).

# In[ ]:


df_list = [answers,comments,emails,group_memberships,groups,matches,professionals,questions,school_memberships,students,tag_questions,tag_users,tags]


# In[ ]:


# sanity check there is 13 csv file.
len(df_list)


# In[ ]:


for table in df_list:
    print(table.columns)


# Based on previously mentioned sources and the printed one [this](https://dbdiagram.io/d/5c7a39cbf7c5bb70c72f2bb4) can be produced. (I used [Sublime3](https://www.sublimetext.com/3) for faster code production)

# In[ ]:


answers.head()


# In[ ]:


comments.head()


# In[ ]:


emails.head() #to find out the type of the emails_frequency_level -> varchar


# In[ ]:


group_memberships.head()


# In[ ]:


groups.head() #to find out the type of the groups_group_type -> varchar


# In[ ]:


matches.head()


# In[ ]:


professionals.dropna().head() #to find out the type of the cols and ignore the nan's -> varchar*3


# In[ ]:


questions.head() #same here


# In[ ]:


school_memberships.head()


# In[ ]:


students.dropna().head()


# In[ ]:


tag_questions.head()


# In[ ]:


tag_users.head()


# In[ ]:


tags.head()


# In[ ]:


matches.head()


#  So we have [this](https://dbdiagram.io/d/5c7a39cbf7c5bb70c72f2bb4) ***the full map of the db schema*** and the relationships separately. 
#  **This is important** we use the relationships as first class citizen of a graph database, so we can look for patterns and travelsals and not just for tags. (If you want to imporve it use the "Edit as New" button and tell me :) )

# **Before we begin the import based on the Northwind example into Neo4j I'am going to collect the nodes, properties relations ships and labels.**
# (Disclaimer: I will not going to handle the changers between professional and student, however easy it is to give two labels to a node and extra properties, because it it not needed for recommendation)
# Note: it is not necessary to store the ID-s the database can generate it automatically, however it is useful for the full picture and for the matching in the beginning.
# 
# ***Relationships:*** (with **label** and *properties*) Note: One relationship could have just one label
# * **MEMBER_IN** based on the
#   * Table group_memberships tabel { group_memberships_group_id char, group_memberships_user_id char }
#     * Ref: "groups"."groups_id" < "group_memberships"."group_memberships_group_id"
#     * Ref: "students"."students_id" < "group_memberships"."group_memberships_user_id"
#     * Ref: "professionals"."professionals_id" < "group_memberships"."group_memberships_user_id"
#   * Table school_memberships { school_memberships_school_id int, school_memberships_user_id char }
#     * Ref: "professionals"."professionals_id" < "school_memberships"."school_memberships_user_id"
#     * Ref: "students"."students_id" < "school_memberships"."school_memberships_user_id"  
# * **IS_IN** based on the  
#   * Table matches { matches_email_id int, matches_question_id char }
#     * Ref: "emails"."emails_id" < "matches"."matches_email_id"
#     * Ref: "questions"."questions_id" < "matches"."matches_question_id"
# * **HAS_TAG** based on the
#   * Table tag_questions {  tag_questions_tag_id int,  tag_questions_question_id char }
#     * Ref: "tags"."tags_tag_id" < "tag_questions"."tag_questions_tag_id"
#     * Ref: "questions"."questions_id" < "tag_questions"."tag_questions_question_id"
#   * Table tag_users { tag_users_tag_id int,  tag_users_user_id char }
#     * Ref: "tags"."tags_tag_id" < "tag_users"."tag_users_tag_id"
#     * Ref: "professionals"."professionals_id" < "tag_users"."tag_users_user_id"
#     * Ref: "students"."students_id" < "tag_users"."tag_users_user_id"
# * **AUTHOR_OF** based on the    
#   * Ref: "students"."students_id" < "questions"."questions_author_id"
#   * Ref: "professionals"."professionals_id" < "answers"."answers_author_id"
#   * Ref: "professionals"."professionals_id" < "comments"."comments_author_id"
#   * Ref: "students"."students_id" < "comments"."comments_author_id" 
# * **IS_REPLY_TO** based on the
#   * Ref: "questions"."questions_id" < "answers"."answers_question_id"
#   * Ref: "questions"."questions_id" < "comments"."comments_parent_content_id"
#   * Ref: "answers"."answers_id" < "comments"."comments_parent_content_id"
# * **GOT_EMAIL** based on the
#   * Ref: "professionals"."professionals_id" < "emails"."emails_recipient_id"
#   * Ref: "students"."students_id" < "emails"."emails_recipient_id"
# 

# ***Nodes*** (with **labels** and *properties*):  
# * **Student**
#   * *students_id char* PK
#   * *students_location* varchar
#   * *students_date_joined* date
# * **Question**
#   * *questions_id* char PK
#   * *questions_author_id* char
#   * *questions_date_added* date
#   * *questions_title* varchar
#   * *questions_body* text
# * **Professional**
#   * *professionals_id* char PK
#   * *professionals_location* varchar
#   * *professionals_industry* varchar
#   * *professionals_headline* varchar
#   * *professionals_date_joined* date
# * **Answer**
#   * *answers_id* char PK
#   * *answers_author_id* char
#   * *answers_question_id* char
#   * *answers_date_added* date
#   * *answers_body* text
# * **Comment**
#   * *comments_id* char PK
#   * *comments_author*_id char
#   * *comments_parent_content_id* char
#   * *comments_date_added* date
#   * *comments_body* text 
# * **Email**
#   * *emails_id* int PK
#   * *emails_recipient_id* char
#   * *emails_date_sent* date
#   * *emails_frequency_level* varchar 
# * **Group**
#   * *groups_id* char PK
#   * *groups_group_type* varchar
# * **Tag**
#   * *tags_tag_id* int PK
#   * *tags_tag_name* varchar
# * **School**
#   * *school_memberships_school_id* int
#   

# The graph database space is addtive so we can easily make a new node type from professional location or from any desired and usefull property, moreover we can make new connection based on insights, or reduce the tags as labels (**bear with me, we just building the fundations, a few more scroll and the FUN begins**)

# The next step is to draw the graph database schema and then import the files to begin the data analysis and then the building of the recommendation engine.

# For the drawing I'm going to use [this](http://console.neo4j.org/) tool. Not the best, drawing in google draw will be faster for the picture, but we will use it for explaining the import, for the readers.

# First delete everything from the console with the following [Cypher](https://neo4j.com/developer/cypher-query-language/) query, do not use as an admin in a prod database ;)
# > match (n) detach delete n
# 
# Or click on the **Clear DB** button

# Insert this query into the console:
# 
# > create (student:Student {students_id:"StudentJerry"}),(question:Question {questions_id:"Why?"}),(professional:Professional {professionals_id:"Mr. CEO"}),(answer:Answer {answers_id:"Keep delivering!"}),(comment:Comment {comments_id:"Thank you!"}),(email:Email {emails_id:"You got a question!"}),
# (group:Group {groups_id:"Awesome Group"}),(tag:Tag {tags_tag_id:"Future Saver"}),(school:School {schools_id:"Oxford"}), 
# (student)-[:MEMBER_IN]->(group), (professional)-[:MEMBER_IN]->(group), (student)-[:MEMBER_IN]->(school),(professional)-[:MEMBER_IN]->(school),(student)-[:HAS_TAG]->(tag), (professional)-[:HAS_TAG]->(tag),
# (question)-[:HAS_TAG]->(tag), (student)-[:AUTHOR_OF]->(question), (student)-[:AUTHOR_OF]->(comment), (professional)-[:AUTHOR_OF]->(comment),(professional)-[:AUTHOR_OF]->(answer),(answer)-[:IS_REPLY_TO]->(question),(comment)-[:IS_REPLY_TO]->(question), (comment)-[:IS_REPLY_TO]->(answer), (professional)-[:GOT_EMAIL]->(email),(student)-[:GOT_EMAIL]->(email), (question)-[:QUESTION_IS_IN_EMAIL]->(email)
# 
# Then you will see the basic schema of the db, where any nodes can have any type of connections.

# If you want to be more professional import the upper Cypher query from the upper cell to a Neo4j instance, then call:
# > call db.schema()
# 
# You have to allow [APOC](https://neo4j.com/developer/neo4j-apoc/) to do this.

# After a few visualisation trick you can see the following schema:

# In[ ]:


from IPython.display import Image
Image(filename="../input/cv-graph-schema2/graph_schema_kaggle.png")


# **Note1: COMPARE TO [THIS](https://dbdiagram.io/d/5c7a39cbf7c5bb70c72f2bb4) SQL SCHEMA :D **
# 
# Note2: As you can see there is a self loop in the schema but I could not figure out why, if you do ***please tell me***.

# Now we finally begin to [import](https://neo4j.com/developer/guide-importing-data-and-etl/) the CSVs to the Neo4j graph database. The previous section was just for the understanding of the process, thins are speeding up from here ;)

# For the load and for other fun things [locate](https://neo4j.com/docs/operations-manual/current/configuration/file-locations/#table-file-locations-environment-variables) the neo4j.conf file on you computer, and change the line of:
# > #dbms.security.procedures.whitelist=
# 
# to this:
# > dbms.security.procedures.whitelist=apoc.*
# 
# and change the:
# > dbms.directories.import=import
# 
# to this:
# > #dbms.directories.import=import
# 
# moreover change this:
# > #dbms.security.procedures.unrestricted=my.extensions.example,my.procedures.*
# 
# to this:
# > dbms.security.procedures.unrestricted=apoc.*

# **All the [importing](https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/) Cypher queryes are here, without file location obviously ;)
# 
# Note: the dates are handled this way because I do not care if the answer was in 1 hour or in a full day.**
# 

# ***Import the NODES, nodes' labels, and properties first:***
# *students*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/students.csv" AS row
# CREATE (student:Student)
# SET student._id = row.students_id,
# 	student.location = row.students_location,
# 	student.join_date = date(LEFT(row.students_date_joined,10)),
#   student.join_date_text = row.students_date_joined
#   
# *questions*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/questions.csv" AS row
# CREATE (question:Question)
# SET question._id = row.questions_id,
#     question.author_id = row.questions_author_id,
#     question.date_added = date(LEFT(row.questions_date_added,10)),
#     question.date_added_text = row.questions_date_added,
#     question.title = row.questions_title,
#     question.body = row.questions_body
# 
# *professionals*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/professionals.csv" AS row
# CREATE (professional:Professional)
# SET professional._id = row.professionals_id,
#   	professional.location = row.professionals_location,
#   	professional.industry = row.professionals_industry,
#   	professional.headline = row.professionals_headline,
#   	professional.date_joined = date(LEFT(row.professionals_date_joined,10)),
#     professional.date_joined_text = row.professionals_date_joined
# 
# *asnwers*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/answers.csv" AS row
# CREATE (answer:Answer)
# SET answer._id = row.answers_id,
#   	answer.author_id = row.answers_author_id,
#   	answer.question_id = row.answers_question_id,
#   	answer.date_added = date(LEFT(row.answers_date_added,10)),
#     answer.date_added_text = row.answers_date_added,
#   	answer.body = row.answers_body
#     
# *comments*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/comments.csv" AS row
# CREATE (comment:Comment)
# SET comment._id = row.comments_id,
#   	comment.author_id = row.comments_author_id,
#   	comment.parent_content_id = row.comments_parent_content_id,
#   	comment.date_added = date(LEFT(row.comments_date_added,10)),
#     comment.date_added_text = row.comments_date_added,
#   	comment.body = row.comments_body
#     
# *emails*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/emails.csv" AS row
# CREATE (email:Email)
# SET email._id = toInteger(row.emails_id),
#   	email.recipient_id = row.emails_recipient_id,
#   	email.date_sent = date(LEFT(row.emails_date_sent,10)),
#     email.date_sent_text = row.emails_date_sent,
#   	email.frequency_level = row.emails_frequency_level
# 
# *groups*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/groups.csv" AS row
# CREATE (group:Group)
# SET group._id = row.groups_id,
#     group.type = row.groups_group_type
# 
# *tags*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/tags.csv" AS row
# CREATE (tag:Tag)
# SET tag._id = toInteger(row.tags_tag_id),
#     tag.name = row.tags_tag_name
# 
# *schools*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/school_memberships.csv" AS row
# CREATE (school:School)
# SET school._id = toInteger(row.school_memberships_school_id)
# 
#   

# ***You just imported all the student nodes to the database!***

# **Create indexes for faster maching**
# > CREATE INDEX ON :School(_id)
# > CREATE INDEX ON :Student(_id)
# > CREATE INDEX ON :Question(_id)
# > CREATE INDEX ON :Question(author_id)
# > CREATE INDEX ON :Professional(_id)
# > CREATE INDEX ON :Answer(_id)
# > CREATE INDEX ON :Answer(author_id)
# > CREATE INDEX ON :Answer(question_id)
# > CREATE INDEX ON :Comment(_id)
# > CREATE INDEX ON :Comment(author_id)
# > CREATE INDEX ON :Comment(parent_content_id)
# > CREATE INDEX ON :Email(_id)
# > CREATE INDEX ON :Email(recipient_id)
# > CREATE INDEX ON :Group(_id)
# > CREATE INDEX ON :Tag(_id)
# > CREATE INDEX ON :School(_id)
# > CREATE INDEX ON :School(member_id)

# **We imported the schools from a connecting table so we are getting rid of the duplications:**
# > match (n:School) return count(n)
# > MATCH (n:School)
# WITH n._id AS _id, COLLECT(n) AS nodelist, COUNT(*) AS count
# WHERE count > 1 return Count(nodelist)
# > MATCH (n:School)
# WITH n._id AS _id, COLLECT(n) AS nodelist, COUNT(*) AS count
# WHERE count > 1
# CALL apoc.refactor.mergeNodes(nodelist) YIELD node
# RETURN node
# > MATCH (n:School)
# WITH n._id AS _id, COLLECT(n) AS nodelist, COUNT(*) AS count
# WHERE count > 1 return Count(nodelist)
# > match (n:School) return count(n)

# ***The fun part the EDGES and edges' labels, (no properties here)*** (or relationships)
# *(student)-[:MEMBER_IN]->(group)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/group_memberships.csv" AS row
# MATCH (student:Student {_id: row.group_memberships_user_id})
# MATCH (group:Group {_id: row.group_memberships_group_id})
# MERGE (student)-[:MEMBER_IN]->(group)
# 
# *(student)-[:MEMBER_IN]->(school)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/school_memberships.csv" AS row
# MATCH (student:Student {_id: row.school_memberships_user_id})
# MATCH (school:School {_id: toInteger(row.school_memberships_school_id)})
# MERGE (student)-[:MEMBER_IN]->(school)
# 
# * (professional)-[:MEMBER_IN]->(group)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/group_memberships.csv" AS row
# MATCH (professional:Professional {_id: row.group_memberships_user_id})
# MATCH (group:Group {_id: row.group_memberships_group_id})
# MERGE (professional)-[:MEMBER_IN]->(group)
# 
# *(professional)-[:MEMBER_IN]->(school)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/school_memberships.csv" AS row
# MATCH (professional:Professional {_id: row.school_memberships_user_id})
# MATCH (school:School {_id: toInteger(row.school_memberships_school_id)})
# MERGE (professional)-[:MEMBER_IN]->(school)
# 
# *(student)-[:HAS_TAG]->(tag)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/tag_users.csv" AS row
# MATCH (student:Student {_id: row.tag_users_user_id})
# MATCH (tag:Tag {_id: toInteger(row.tag_users_tag_id)})
# MERGE (student)-[:HAS_TAG]->(tag)
# 
# *(professional)-[:HAS_TAG]->(tag)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/tag_users.csv" AS row
# MATCH (professional:Professional {_id: row.tag_users_user_id})
# MATCH (tag:Tag {_id: toInteger(row.tag_users_tag_id)})
# MERGE (professional)-[:HAS_TAG]->(tag)
# 
# *(question)-[:HAS_TAG]->(tag)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/tag_questions.csv" AS row
# MATCH (question:Question {_id: row.tag_questions_question_id})
# MATCH (tag:Tag {_id: toInteger(row.tag_questions_tag_id)})
# MERGE (question)-[:HAS_TAG]->(tag)
# 
# *(student)-[:AUTHOR_OF]->(question)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/questions.csv" AS row
# MATCH (question:Question {_id: row.questions_id})
# MATCH (student:Student {_id: row.questions_author_id})
# MERGE (student)-[:AUTHOR_OF]->(question)
# 
# *(student)-[:AUTHOR_OF]->(comment)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/comments.csv" AS row
# MATCH (comment:Comment {_id: row.comments_id})
# MATCH (student:Student {_id: row.comments_author_id})
# MERGE (student)-[:AUTHOR_OF]->(comment)
# 
# *(professional)-[:AUTHOR_OF]->(comment)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/comments.csv" AS row
# MATCH (comment:Comment {_id: row.comments_id})
# MATCH (professional:Professional {_id: row.comments_author_id})
# MERGE (professional)-[:AUTHOR_OF]->(comment)
# 
# *(professional)-[:AUTHOR_OF]->(answer)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/answers.csv" AS row
# MATCH (answer:Answer {_id: row.answers_id})
# MATCH (professional:Professional {_id: row.answers_author_id})
# MERGE (professional)-[:AUTHOR_OF]->(answer)
# 
# *(answer)-[:IS_REPLY_TO]->(question)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/answers.csv" AS row
# MATCH (answer:Answer {_id: row.answers_id})
# MATCH (question:Question {_id: row.answers_question_id})
# MERGE (answer)-[:IS_REPLY_TO]->(question) 
# 
# *(comment)-[:IS_REPLY_TO]->(question)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/comments.csv" AS row
# MATCH (comment:Comment {_id: row.comments_id})
# MATCH (question:Question {_id: row.comments_parent_content_id})
# MERGE (comment)-[:IS_REPLY_TO]->(question)
# 
# *(comment)-[:IS_REPLY_TO]->(answer)*
# > USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/comments.csv" AS row
# MATCH (comment:Comment {_id: row.comments_id})
# MATCH (answer:Answer {_id: row.comments_parent_content_id})
# MERGE (comment)-[:IS_REPLY_TO]->(answer)
# 
# *(question)-[:QUESTION_IS_IN_EMAIL]->(email)* **different iterative approach had taken, due to the high volume of edges and limited resources**
# Importing the matches as nodes
# >  USING PERIODIC COMMIT 10000
# LOAD CSV WITH HEADERS FROM "file:///<location-of-the-files>/matches.csv" AS row
# CREATE (matches:Matches)
# SET matches.email_id = toInteger(row.matches_email_id),
#     matches.question_id = row.matches_question_id
# 
# Creating indexes
# > CREATE INDEX ON :Matches(email_id)
# > CREATE INDEX ON :Matches(question_id)
# 
# Connect matches to question and email with and apoc method:
# > call apoc.periodic.iterate("MATCH (question:Question) with question match(matches:Matches {question_id:question._id}) return question,matches","CREATE (question)-[:GOT_EMAIL]->(matches)", {batchSize:1000}) yield batches, total return batches, total
# > call apoc.periodic.iterate("MATCH (email:Email) with email match(matches:Matches {email_id:email._id}) return email,matches","CREATE (email)-[:GOT_EMAIL]->(matches)", {batchSize:1000}) yield batches, total return batches, total
# 
# Make the relationships between email and questions:
# > call apoc.periodic.iterate("MATCH (question:Question)-[:GOT_EMAIL]->(matches:Matches)<-[:GOT_EMAIL]-(email:Email) return email,question","CREATE (question)-[:IS_IN]->(email)", {batchSize:1000}) yield batches, total return batches, total
# 
# Delete matches nodes:
# > call apoc.periodic.iterate("MATCH (matches:Matches) return matches", "DETACH DELETE matches", {batchSize:1000}) yield batches, total return batches, total

# ***Congratulations if you have the nodes and relationships in neo4j as it is written previously. You just uploaded a relational database into a graph database and you are AWESOME!***

# Note:Disclaimer about the import, it is not the BEST way to import this and not the prettiest there could be multiple improvements alogn the way I'm doing this way to get to analysis as fast as I could, and take just the neccesary trade offs.

# ***Some cosmetics: making group type, location, and industry nodes***

# > //make nodes from group properties
# MATCH (n:Group) 
# WITH COLLECT(DISTINCT n.type) AS gts
# UNWIND gts AS group_type
# MERGE (t:Group_Type {type:group_type})
# WITH t,group_type
# MATCH (s:Group {type:group_type}) 
# MERGE (s)-[:HAS_TYPE]->(t)

# > //make nodes from student location
# MATCH (n:Student)
# WITH COLLECT(DISTINCT n.location) as locs 
# UNWIND locs as loc
# MERGE (t:Location {location:loc})
# WITH t,loc
# MATCH (s:Student {location:loc}) 
# MERGE (s)-[:BASED_IN]->(t)

# //make nodes from professional location
# MATCH (n:Professional) WITH COLLECT(DISTINCT n.location) as locs
# UNWIND locs as loc
# MERGE (t:Location {location:loc})
# WITH t,loc
# MATCH (p:Professional {location:loc}) 
# MERGE (p)-[:BASED_IN]->(t)
# 

# //make nodes from professional industry
# MATCH (n:Professional) WITH COLLECT(DISTINCT n.industry) as inds
# UNWIND inds as ind
# MERGE (t:Industry {industry:ind})
# WITH t,ind
# MATCH (p:Professional {industry:ind}) 
# MERGE (p)-[:WORKING_IN]->(t)

# ***I will end this kernel, it is just the basics for the network analysis and the recommendation engine, the kernel about that will be here!***

# More to come, we are just at the beginning of the journey!

# Check out the other kernel where I do the recommendation engine.

# * **This is just the data transformation for the EDA, NLP and Recomendation Engine please check out the followng kernel:[https://www.kaggle.com/ironben/networkbased-eda-nlp-rec-engine/edit](https://www.kaggle.com/ironben/networkbased-eda-nlp-rec-engine/edit)***

# Thank you for your effort following this through! If you have any questions, hit me with that :) 
