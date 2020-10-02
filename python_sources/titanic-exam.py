import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv 
test_file = open('../input/test.csv', 'r')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()
survived_file = open("survivedpassenger.csv", "w")
survived_file_object = csv.writer(survived_file)

survived_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       
    if row[3] == 'female':                                               
        survived_file_object.writerow([row[0],'1'])    
    else:                                   
        survived_file_object.writerow([row[0],'0'])    
test_file.close()
survived_file.close()
