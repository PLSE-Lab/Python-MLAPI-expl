# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
import csv
data = []
with open('../input/open-2-shopee-code-league-order-brushing/order_brush_order.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Columns name are {", ". join(row)}')
            line_count += 1
        else:
            # print(f'{row[0]} {row[1]} {row[2]} {row[3]}')
            row_data = list(csv.reader(row))
            data.append(row_data)
            line_count += 1
    
def sellerID(elem):
    return elem[1]

def timesort(elem):
    return elem[3]

data.sort(key=sellerID)
data.sort(key=timesort)

for x in range(len(data)):
    for y in range(len(data[x])):
        print(data[x][y])
    
    
    

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

