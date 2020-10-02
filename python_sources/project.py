# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
connection = sqlite3.connect('../input/database.sqlite')
#findDeleted = pd.read_sql('SELECT * FROM May2015 WHERE body = \'[deleted]\'', connection)
BESTCOMMENTS = pd.read_sql('SELECT subreddit, gilded, author, edited, score, body  FROM May2015 WHERE score < 0 ORDER BY score DESC', connection)
#findControversialDeleted.to_csv('controversialDeletions', sep = ',');
#sortedDeletions = findDeleted['subreddit'].value_counts()
#sortedDeletions.to_csv('deleted.csv', sep = ',');

#sortedDeletions['removal_reason'].to_csv('removalREason.csv', sep = ',');

#howControversialDeletions = pd.read_csv('deletedComments');
BESTCOMMENTS.head(1000000).to_csv('bestComments.csv', sep = ',');