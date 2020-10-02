# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
file= "../input/30days_interests.csv"
data = pd.read_csv(file, sep=",")
time=data['Time']
emmanuel = data['emmanuel macron: (france)']
francois = data['francois fillon: (france)']
marine = data['marine le pen: (france)']
jean = data['jean-luc melenchon: (france)']
benoit = data['benoit hamon: (france)']
#dates = ['01/02/1991','01/03/1991','01/04/1991']
x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in time]
print(type(x))
#print(x)
plt.figure(figsize=(20,10))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/2017'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,emmanuel)
plt.plot(x,francois)
plt.plot(x,marine)
plt.plot(x,jean)
plt.plot(x,benoit)

plt.legend(['emmanuel macron', 'francois fillon', 'marine le pen', 'jean-luc melenchon','benoit hamon'], loc='upper right')

plt.xlabel("Date Time")
plt.ylabel("Interests point")
plt.title("30 days interest points of candidates ")

plt.gcf().autofmt_xdate()

plt.savefig("date.jpg")
