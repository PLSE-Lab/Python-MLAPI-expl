# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sqlite3
import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
con = sqlite3.connect("../input/database.sqlite")
cursor = con.cursor()

def get_names_for_year(y):
    #get all names for year y
    qstr = "SELECT Name FROM NationalNames WHERE Year = '{}'".format(y)
    cursor.execute(qstr)
    return cursor.fetchall()

def get_name_count_for_year(n, y):
    #get the count of the name n in year y
    qstr = "SELECT Count FROM NationalNames WHERE Year = '{}' AND Name = '{}'".format(y,n)
    cursor.execute(qstr)
    tmp_result = cursor.fetchall()
    if len(tmp_result) > 0:
        result = 0
        for i in range(0,len(tmp_result)):
            result = result + tmp_result[i][0]
        return result
    else:
        return 4
    
#get first year
cursor.execute("SELECT Year FROM NationalNames ORDER BY Year ASC LIMIT 1;")
first_year = cursor.fetchall()[0][0]

#cannot run the script from first_year, due to timeout problem (more than 1200 seconds)
for y in range(2012, dt.datetime.now().year):
    #loop years
    names = get_names_for_year(y)
    max_rise = 0
    for n in names:
        count_this_year = get_name_count_for_year(n[0],y)
        count_last_year = get_name_count_for_year(n[0],y-1)
        rise = (count_this_year/count_last_year*100) -100
        if rise > max_rise:
            max_rise = rise
            max_rise_name = n[0]
    print("Name with the maximum popularity rise for year {}: {} ({}% rise)".format(y,max_rise_name,'{number:.{digits}f}'.format(number=max_rise, digits=2)))
