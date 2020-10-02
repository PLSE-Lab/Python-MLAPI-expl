# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3


import matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
connection = sqlite3.connect("../input/database_17_0.sqlite")
election_df = pd.read_sql_query("SELECT * FROM data",connection)
connection.close()
print(election_df.head())
print(election_df['location'])
country = ['france','France']

election_df = election_df[election_df.location.str.contains('|'.join(country))==True]
election_df_france = election_df[['mention_Fillon',
                           'mention_Hamon','mention_Dupont-Aignan',
                           'mention_Le Pen',
                           'mention_Macron',
                           'mention_Mélenchon','location']]
election_tweets = election_df_france.groupby('location').sum()
print(election_tweets)
election_tweets.plot()
plt.show()
# Any results you write to the current directory are saved as outpu'