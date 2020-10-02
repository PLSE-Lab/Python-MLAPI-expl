# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sqlite3
import datetime as dt
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
con = sqlite3.connect("../input/database.sqlite")
cursor = con.cursor()


def get_names_for_year(y):
    #gets names for year y
    qstr = "SELECT Name FROM NationalNames WHERE Year = '{}'".format(y)
    cursor.execute(qstr)
    return cursor.fetchall()
    
def get_boy_names_for_year(y):
    #gets ten boy names for the year y
    qstr = "SELECT Name FROM NationalNames WHERE Year = '{}' AND Gender = 'M'".format(y)
    cursor.execute(qstr)
    return cursor.fetchall()
    
def get_girl_names_for_year(y):
    #get all names for year y
    qstr = "SELECT Name FROM NationalNames WHERE Year = '{}' AND Gender ='F'".format(y)
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

def top_ten_girls():
    cursor.execute("SELECT Year FROM NationalNames ORDER BY Year ASC LIMIT 1;")
    first_year = cursor.fetchall()[0][0]
    for y in range(dt.datetime.now().year):
        #loop years
        count_dict = {}
        girl_names = get_girl_names_for_year(y)
        top_girls = girl_names[:10]
        for n in top_girls:
            count_this_year = get_name_count_for_year(n[0],y)
            count_dict[n[0]] = count_this_year
        print(count_dict)
        # keeps a manageble number of loops
        if y == first_year + 5: break

def top_ten_boys():
    cursor.execute("SELECT Year FROM NationalNames ORDER BY Year ASC LIMIT 1;")
    first_year = cursor.fetchall()[0][0]
    for y in range(dt.datetime.now().year):
        #loop years
        count_dict = {}
        boys_names = get_boy_names_for_year(y)
        top_boys = boys_names[:10]
        for n in top_boys:
            count_this_year = get_name_count_for_year(n[0],y)
            count_dict[n[0]] = count_this_year
        print(count_dict)
        # keeps a manageble number of loops
        if y == first_year + 5: break

def biggest_change():
    percent_change_b = 0
    percent_change_g = 0
    name_girl = ''
    name_boy =''
    cursor.execute("SELECT Year FROM NationalNames ORDER BY Year ASC LIMIT 1;")
    first_year = cursor.fetchall()[0][0]
    for y in range(first_year + 1,first_year +10):
        girls_names = get_girl_names_for_year(y)
        boys_names = get_boy_names_for_year(y)
        for n in boys_names:
            count_this_year = get_name_count_for_year(n[0],y)
            count_last_year = get_name_count_for_year(n[0],y-1)
            temp_percent = (count_this_year/count_last_year)*100
            if temp_percent > percent_change_b:
                percent_change_b = temp_percent
                name_boy = n[0]
        for n in girls_names:
            count_this_year = get_name_count_for_year(n[0],y)
            count_last_year = get_name_count_for_year(n[0],y-1)
            temp_percent = (count_this_year/count_last_year)*100
            if temp_percent > percent_change_g:
                percent_change_g = temp_percent
                name_girl = n[0]
        print (name_boy, name_girl)

def average_length():
    cursor.execute("SELECT Year FROM NationalNames ORDER BY Year ASC LIMIT 1;")
    first_year = cursor.fetchall()[0][0]
    avg_len = []
    for y in range(first_year + 1, dt.datetime.now().year):
        names = get_names_for_year(y)
        names_len = 0
        for n in names:
            names_len += len(n[0])
            mean = names_len/len(names)
        avg_len.append(mean)
    return avg_len

def most_common_start():
    cursor.execute("SELECT Year FROM NationalNames ORDER BY Year ASC LIMIT 1;")
    first_year = cursor.fetchall()[0][0]
    letter_count = {}
    for y in range(first_year + 1, dt.datetime.now().year):
        names = get_names_for_year(y)
        for n in names:
            if (n[0][0]) in letter_count:
                letter_count[n[0][0]] += 1
            else:
                letter_count[n[0][0]] = 1
    print (letter_count)

biggest_change()





