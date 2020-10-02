# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

inp1 = pd.read_csv("../input/BreadBasket_DMS.csv")
inp = inp1
inp.dropna()
inp["Time"] = inp["Time"].str.slice(0, 2).apply(int)
#print(inp)
#z1 = inp.groupby(inp.Item).agg({'Time':np.median}).reset_index()
#z = z1.head(70)
#x = z["Item"]
#y = z["Time"]
#plt.scatter(y, x) 
#plt.figure(figsize=(80,40))
#plt.show()

inp_time = inp[["Time","Item"]]
dumbed = pd.get_dummies(inp_time)
#print(dumbed.columns)
custom_bucket_array = np.linspace(1, 23, 6)
dumbed["Time"] = pd.cut(dumbed["Time"],custom_bucket_array)

fin1 = dumbed.groupby(dumbed.Time).agg({'Item_Coffee':np.sum, 'Item_Bread':np.sum,'Item_Cake':np.sum,'Item_Tea':np.sum }).reset_index()
#print(fin1)
categories = fin1['Time'].cat.categories
ind = np.array([x for x, _ in enumerate(categories)])
width = 0.15 
plt.bar(ind,fin1["Item_Coffee"],width, label='Coffee')
plt.bar(ind + width,fin1["Item_Bread"],width, label='Bread')
plt.bar(ind + width + width,fin1["Item_Cake"],width, label='Cake')
plt.legend()
plt.xticks(ind + width / 2, categories)
plt.xticks(rotation = 90)
plt.show()

dict_days = ["MONDAY","TEUSDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
#print(dict_days[0])
inp["Date"] = pd.to_datetime(inp['Date'])
inp['day_of_week'] = inp['Date'].dt.dayofweek
inp_day = inp[['day_of_week','Item']]
dumbed_day = pd.get_dummies(inp_day,prefix=['Item'],drop_first=True)
#print(dumbed_day.columns)
fin2 = dumbed_day.groupby(dumbed_day['day_of_week']).agg({'Item_Coffee':np.mean, 'Item_Bread':np.mean,'Item_Cake':np.mean,'Item_Tea':np.mean }).reset_index()
#print(fin2.columns)
plt.bar(fin2['day_of_week'],fin2["Item_Coffee"],width, label='Coffee')
plt.bar(fin2['day_of_week'] + width,fin2["Item_Bread"],width, label='Bread')
plt.bar(fin2['day_of_week'] - width,fin2["Item_Cake"],width, label='Cake')
plt.legend(loc = 'best')
plt.xticks(fin2['day_of_week'] + width / 2, dict_days)
plt.xticks(rotation = 90)
plt.show()