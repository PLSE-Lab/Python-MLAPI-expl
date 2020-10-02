# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


con = sqlite3.connect("../input/database.sqlite")

dfCon = pd.read_sql_query('select * from Country', con)
print("dfCon")
print(list(dfCon.columns.values))
print("*****************************************")

dfLeg = pd.read_sql_query('select * from League', con)
print("dfLeg")
print(list(dfLeg.columns.values))
print("*****************************************")

dfMatch = pd.read_sql_query('select * from Match', con)
print("dfMatch")
print(list(dfMatch.columns.values))
print("*****************************************")

dfPlayer = pd.read_sql_query('select * from Player', con)
print("dfPlayer")
print(list(dfPlayer.columns.values))
print("*****************************************")

dfPS = pd.read_sql_query('select * from Player_Stats', con)
print("dfPS")
print(list(dfPS.columns.values))
print("*****************************************")

dfTeam = pd.read_sql_query('select * from Team', con)
print("dfTeam")
print(list(dfTeam.columns.values))
print("*****************************************")
