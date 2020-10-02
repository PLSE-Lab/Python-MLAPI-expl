import numpy as np 
import pandas as pd 
import csv as csv 
test_file = open('../input/test.csv', 'r')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()
alive_file = open("gendermodel.csv", "w")
alive_file_object = csv.writer(alive_file)

alive_file_object.writerow(["PassengerId", "Sex", "Survived"])
for row in test_file_object:       
    if row[3] == 'female':                                               
        alive_file_object.writerow([row[0],row[3],'1'])
    else:                                   
        alive_file_object.writerow([row[0],row[3],'0'])    
test_file.close()
alive_file.close()