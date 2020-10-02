"""Star Wars-themed posts can be found in different subreddits. This Python script
performs linear classification. It reads in reddit bodies from two subreddits where Star Wars-themed 
posts can occur, "movies" and "StarWars", and attempts to classify whether or not the reddit body came 
from the "movies" subreddit or the "StarWars" subreddit based on the number of times words of certain categories 
appear in that body. The classification is done through Linear SVM. Linear SVM creates a linear classifier and 
assigns weights to each category; the larger the magnitude of the weight, the more significant the category 
is to classification. The main goal is to determine the most significant category in classifying these bodies. 
The output from this code will two data tables: a data table showing different Star Wars categories, and the
number of times words from those categories appear in each body; and a smaller data table at the end, showing 
the weights obtained from Linear SVM, and an overall proportion that was correctly classified through Linear SVM. 
The categories will be ranked according to weight, starting with the weight of largest magnitude."""

import numpy as np
import pandas as pd
from sklearn import svm
import sqlite3
import matplotlib.pyplot as plt
"""Note: this Python code was run through Kaggle's website."""

sql_connect= sqlite3.connect('../input/database.sqlite')


"""Specify a desired number of rows to import and only import rows
whose subreddit is either "StarWars" or "movies". Order by id.
Note: 55000 rows will require this program to run for about 20 minutes if run on Kaggle's website."""

n=55000



df = pd.read_sql('select "subreddit", "body" from May2015 where "subreddit" = "StarWars" or "subreddit"="movies" order by id limit '+ str(n),sql_connect)
subreddit = pd.DataFrame(df['subreddit'])

"""In case n rows were not imported, redefine n to be the length of the data frame."""

writer = pd.ExcelWriter('StarWarsReddit.xlsx', engine='xlsxwriter')
df.to_excel(writer)
writer.save()





