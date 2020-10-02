# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd
import sqlite3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt

from scipy.stats import linregress

import statsmodels.formula.api as smf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Read sqlite into a pandas DataFrame
con = sqlite3.connect("../input/database.sqlite")
c = con.cursor()

c.execute('select name from sqlite_master where type=\'table\'')
for table in c:
  print(table[0])
  
Player = pd.read_sql_query("SELECT * from Player", con)
Team = pd.read_sql_query("SELECT * from Team", con)
League = pd.read_sql_query("SELECT * from League", con)
Match = pd.read_sql_query("SELECT * from Match", con)
con.close()

Player.head()

