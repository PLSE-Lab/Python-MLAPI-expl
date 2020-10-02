#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv

fpath = '/kaggle/input/international-football-results-from-1872-to-2017/results.csv'
csvfile = open(fpath)
content = list(csv.reader(csvfile, delimiter=','))
header = list(content[0])
data = list(content[1:])
print("Total number of rows = {:,}; Total number of columns = {:,}\n\n".format(len(data), len(header)))

print("=" * 120)
print("HEADER")
print("=" * 120)
for counter, value in enumerate(header):
    print("[Column: {0:>2d}] {1:}".format(counter, value))

print('\n' * 2)
print("=" * 120)
print("DATA (first 10 rows)")
print("=" * 120)
for counter, row in enumerate(data[:10]):
    print("[Row: {0:>2d}] [{1:}]".format(counter, ', '.join(row)))

