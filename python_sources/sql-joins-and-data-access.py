#!/usr/bin/env python
# coding: utf-8

# In this notebook, I intend to outline how to link and extract data from the two main data containing tables of the provided SQLite database using the `JOIN` command. Then I will investigate the database records of myself and some co-workers
# 
# First we need to load the sqlite package and connect to the database;

# In[ ]:


import sqlite3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()


# Perhaps we have the doi of a paper, and want to see who the authors is, this can be achived by joining the three tables together, the first to link the paperID's, to determine the authorID's corresponding to the doi, and again to link the authorID's with the actual author names;

# In[ ]:


doi = "10.1021/ja004728r"
c.execute('SELECT forename, initials, surname            FROM Authors a            JOIN Paper_Authors pa            ON a.authorID = pa.authorID            JOIN Papers p            ON pa.paperID = p.paperID            WHERE doi = ?;', (doi,))
print(c.fetchall())


# In another situation, we might like to know the dois of papers which I have authored, which is possible using the method outlined above, but this time we want to go from the authors table to the papers table;

# In[ ]:


forename = "Mathew"
surname  = "Savage"
c.execute('SELECT  doi            FROM Papers p            JOIN Paper_Authors pa            ON p.paperID = pa.paperID            JOIN Authors a            ON pa.AuthorID = a.authorID            WHERE forename = ? AND surname = ?;', (forename, surname))
print(c.fetchall())


# Using the dois for the papers which I have authored, we can go one further and generate a list of all of the other people I have authored papers with;

# In[ ]:


c.execute('SELECT DISTINCT forename, initials, surname            FROM Authors a            JOIN Paper_Authors pa            ON a.authorID = pa.authorID            JOIN Papers p            ON pa.paperID = p.paperID            WHERE doi = "10.1021/jacs.6b01323" OR doi = "10.1021/jacs.6b08059"            GROUP BY surname, forename;')
print(c.fetchall())


# Although this approach is hardly satisfactory, as we have to use 2 queries, and know the number of papers we are expecting, in the case above 2.
# 
# We can overcome this problem by generating an inital table *in-situ* using the `WITH` command, to generate a table of paperIDs (papID) which were published by me, and then selecting and `COUNT()`ing authors who also have paperIDs in my table of published paperIDs;

# In[ ]:


def coauthors(forename, initials, surname):
    c.execute('WITH papID AS(                SELECT paperID                FROM Paper_Authors pa                JOIN Authors a                ON pa.authorID = a.authorID                WHERE forename = ? AND initials = ? AND surname = ?)                SELECT COUNT(), forename, initials, surname                FROM Authors a                JOIN Paper_Authors pa                ON a.authorID = pa.authorID                WHERE paperID IN (SELECT paperID FROM papID)                AND forename != ? AND surname != ?                GROUP BY surname, forename;', 
              (forename, initials, surname, forename, surname))
    return c.fetchall()
out = coauthors('Mathew', '', 'Savage')
print(out)

print('\n I have published with ' + str(len(out)) + ' co-workers in JACS')


# As we can see, the result is the same, only with the counts of times I have co-authored with each person, plotting this is quite boring, as I have only published 2 papers in JACS, and usually with the same co-authors;

# In[ ]:


def plotcoauth(out):
    out.sort(key=lambda x: x[0], reverse=True) 

    data, forn, init, surn = zip(*out)
    name = zip(forn, init, surn)
    x_pos = np.arange(len(data)) 

    plt.figure(figsize=(9,3))
    plt.bar(x_pos, data,align='center')
    plt.xticks(x_pos, name, rotation=90) 
    plt.ylabel('Number of papers co-authored with')
    plt.show()
plotcoauth(out)


# Repeating the analysis with someone who has published more papers is a more interestin result, in this analysts Stepehen, the top alpabetically has published 6 papers, and so has a more interesting mix.

# In[ ]:


out = coauthors('Stephen', 'P.', 'Argent')
plotcoauth(out)
print('\n Stephen has published with ' + str(len(out)) + ' co-workers in JACS')


# Another useful sql access technique is the self-join, this is important for when we want to generate a network of author relationships, as each edge of the network would be defined by the connection from one author to the next;
# 
# [Many thanks to CL. on stack overflow for this solution.][1]
# 
#  [1]: http://stackoverflow.com/a/42002707/6813373

# In[ ]:


c.execute('SELECT           a1.authorID AS author1,           a2.authorID AS author2            FROM Paper_Authors AS a1           JOIN Paper_Authors AS a2 USING (paperID)           WHERE a1.authorID < a2.authorID;')

out = c.fetchall()

