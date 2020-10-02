# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import cx_Oracle

con = cx_Oracle.connect('HMIA1/HMIA1@xx.xxx.xxx.xxx')
import cx_Oracle

dsn_tns = cx_Oracle.makedsn('xx.xxx.xxx.xxx, '1521', service_name='Service') #if needed, place an 'r' before any parameter in order to address any special character such as '\'.
conn = cx_Oracle.connect(user=r'xxxx', password='xxxx', dsn=dsn_tns)  
c = conn.cursor()
c.execute('select * from database.table') # use triple quotes if you want to spread your query across multiple lines
for row in c:
print (row[0], '-', row[1]) # this only shows the first two columns. To add an additional column you'll need to add , '-', row[2], etc.

conn.close()



set ORACLE_HOME=("C:\Users\xxxx\instantclient-basic-windows.x64-18.5.0.0.0dbru\instantclient_18_5")

set PATH=%ORACLE_HOME%;%PATH%

SETPATH="C:\Users\xxxxx\Downloads\instantclient-basic-windows.x64-18.5.0.0.0dbru\instantclient_18_5" #;%PATH%

ip = '10.54.xxx.xxx'

port = 1111

SID = 'Service'

dsn_tns = cx_Oracle.makedsn(ip, port, SID)



db = cx_Oracle.connect('ssss', 'ssss', dsn_tns)



import cx_Oracle



dsn = cx_Oracle.makedsn("10.xx.ss.xx","333","Service")

connection = cx_Oracle.connect(dsn,mode = cx_Oracle.SYSDBA)

query = "SELECT * FROM MYTABLE"

cursor = connection.cursor()

cursor.execute(query)

resultSet=cursor.fetchall()

connection.close()
