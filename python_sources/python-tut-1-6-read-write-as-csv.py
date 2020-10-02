import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
test_file = open('../input/test.csv', 'rU')
test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

prediction_file = open("genderbasedmodel.csv", "w")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()