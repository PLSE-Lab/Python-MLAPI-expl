import csv as csv 
import numpy as np

csv_file_object = csv.reader(open('../input/train.csv', 'r', newline='', encoding='utf8')) 
header = next(csv_file_object)

data=[]
for row in csv_file_object:
    data.append(row)
data = np.array(data)

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print(proportion_survivors)