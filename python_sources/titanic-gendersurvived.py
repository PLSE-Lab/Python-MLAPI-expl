import numpy as np 
import pandas as pd 
import csv as csv 
test_file = open('../input/test.csv', 'r')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()
alive_file = open("gendersurvived.csv", "w")
alive_file_object = csv.writer(alive_file)

alive_file_object.writerow(["PassengerId", "Survived", "Sex"])
for row in test_file_object:       
    if row[3] == 'female':                                               
        alive_file_object.writerow([row[0],'1',row[3]])    
    else:                                   
        alive_file_object.writerow([row[0],'0',row[3]])    
test_file.close()
alive_file.close()
