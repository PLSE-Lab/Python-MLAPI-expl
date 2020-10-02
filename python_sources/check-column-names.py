# Find column names used multiple times.
import csv
with open('../input/train.csv') as csvfile:
    reader = csv.reader(csvfile)
    head = next(reader)
    #print(head)
    headDict = {}
    for key in head:
        if key in headDict:
            headDict[key] += 1
        else:
            headDict[key] = 1
    print("columnName: multiple")
    for columnName, multiple in headDict.items():
        if multiple > 1:
            print("%s: %s" % (columnName, multiple))