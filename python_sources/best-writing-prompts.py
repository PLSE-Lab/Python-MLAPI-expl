# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

sql_conn = sqlite3.connect('../input/database.sqlite')

#df = pd.read_sql("SELECT score, body FROM May2015 WHERE score > 50 AND subreddit=='WritingPrompts' LIMIT 10000", sql_conn)
#df['body'].to_csv("WritingPrompts_50.txt", index=False)


df = pd.read_sql("SELECT score, body FROM May2015 WHERE score > 10 AND subreddit=='WritingPrompts' LIMIT 10000", sql_conn)

print(df.shape)

print(df['score'][0])
print(df['body'][0])