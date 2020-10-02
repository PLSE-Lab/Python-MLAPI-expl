import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
table_names = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

