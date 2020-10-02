import csv
import matplotlib.pyplot as plot
import numpy as np
from numpy import genfromtxt
from collections import Counter as c
import re

years = []
unique_years = []
value_error_count = 0
location = r'../input/meteorite-landings.csv'
with open(location,'r', encoding='utf8')as f:
    reader = csv.reader(f)
    for line in reader:
        if(line[6] == 'year'):
            print("header removal")
        else:
            try:
                years.append(int(line[6]))
            except(ValueError):
                print("Could not enter datapoint, moving on")
                value_error_count += 1
print("finished with: "+str(value_error_count)+" ERRORS")

count = c(years)
top_ten = count.most_common(10)
top_ten_x = []
top_ten_y = []

for year, count in top_ten:
    top_ten_y.append(count)
    top_ten_x.append(year)

plot.bar(range(10), top_ten_y)
plot.xticks(range(10), top_ten_x)
plot.show()