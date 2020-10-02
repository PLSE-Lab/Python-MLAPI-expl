import numpy as np # linear algebra
import csv as csv;
import math as math;

data = [];
process_data = [];
process_data_answer = [];
centroid1 = [2,2,2,2]
centroid2 = [3,3,3,3]
centroid3 = [4,4,4,4]
groups = [];

def euclidistance(a1,b1,c1,d1,a2,b2,c2,d2):
    a3 = a1-a2;
    b3 = b1-b2;
    c3 = c1-c2;
    d3 = d1-d2;
    a4 = a3*a3;
    b4 = b3*b3;
    c4 = c3*c3;
    d4 = d3*d3;
    e = a4+b4+c4+d4;
    return math.sqrt(e)
    
def printlist():
    for row in data:
        print(row);

def group(a1,a2,a3,a4):
    b1 = euclidistance(centroid1[0],centroid1[1],centroid1[2],centroid1[3],a1,a2,a3,a4)
    b2 = euclidistance(centroid2[0],centroid2[1],centroid2[2],centroid2[3],a1,a2,a3,a4)
    b3 = euclidistance(centroid3[0],centroid3[1],centroid3[2],centroid3[3],a1,a2,a3,a4)
    if b1 < b2:
        if b1 < b3:
            return 1;
    if b2 < b1:
        if b2 < b3:
            return 2;
    if b3 < b1:
        if b3 < b2:
            return 3;
            
def adjustcentroids():
    global centroid1;
    global centroid2;
    global centroid3;
    group1 = [];
    group2 = [];
    group3 = [];
    for row in process_data:
        g = group(row[0],row[1],row[2],row[3])
        if g == 1:
            group1.append([row[0],row[1],row[2],row[3]])
        if g == 2:
            group2.append([row[0],row[1],row[2],row[3]])
        if g == 3:
            group3.append([row[0],row[1],row[2],row[3]])
    centroid1 = meandataset(group1)
    centroid2 = meandataset(group2)
    centroid3 = meandataset(group3)
    
def meandataset(i):
    row1 = 0;
    row2 = 0;
    row3 = 0;
    row4 = 0;
    for row in i:
        row1 = row1 + row[0];
        row2 = row2 + row[1];
        row3 = row3 + row[2];
        row4 = row4 + row[3];
    final = [];
    final.append(row1 / len(i));
    final.append(row2 / len(i));
    final.append(row3 / len(i));
    final.append(row4 / len(i));
    return final;
    
def accuracey():
    total = len(process_data)
    right = 0;
    for index,row in enumerate(process_data):
        thegroup = group(row[0],row[1],row[2],row[3]);
        thegroupanswer = valuetonumber(process_data_answer[index]);
        if thegroup == thegroupanswer:
            right = right + 1;
    return str(right) + "/" + str(total);

def valuetonumber(i):
    if i == "Iris-setosa":
        return 1;
    if i == "Iris-versicolor":
        return 2;
    if i == "Iris-virginica":
        return 3;
        
def printcentroids():
    print(centroid1)
    print(centroid2)
    print(centroid3)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

io = open("../input/Iris.csv")

#readers

reader = csv.reader(io)

#reading
reader.__next__()
for row in reader:
    data.append(row);
    
for row in data:
    process_data.append([float(row[1]),float(row[2]),float(row[3]),float(row[4])]);
    process_data_answer.append(row[5]);
    
printcentroids();
adjustcentroids();
adjustcentroids();
adjustcentroids();
adjustcentroids();
adjustcentroids();
adjustcentroids();
adjustcentroids();
printcentroids();

#for row in process_data:
#    print(group(row[0],row[1],row[2],row[3]));

print(accuracey())







