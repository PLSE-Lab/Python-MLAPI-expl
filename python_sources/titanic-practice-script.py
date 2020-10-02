#import numpy as np
#import pandas as pd

#print('Hello Titanic <==||=||=||=||==>')

#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

###########

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv # CSV file operations
import os


trainFile = open('../input/train.csv','r')
train = csv.reader(trainFile)
header = str(next(train))
data = []
for row in train:
    data.append(row)    
data = np.array(data)
trainFile.close()


NumPassengers = data[:,1].size
NumSurvivors = sum(data[:,1].astype('int'))
PctSurvived = (NumSurvivors/NumPassengers)*100
print('Only {} people survived out of the total {} passengers, which is a shocking {:.4f}%'.format(NumSurvivors,NumPassengers,PctSurvived))


femaleSel = data[:,4] == 'female'
maleSel = data[:,4] == 'male'
MaleData = data[maleSel,1].astype('int')
#print(MaleData,MaleData.size)
NumMale = MaleData.size
MaleSur = MaleData.sum()
NumFemale = data[femaleSel,0].size
FemaleSur = data[femaleSel,1].astype('int').sum()
print(NumMale,MaleSur,NumFemale,FemaleSur)


testFile = open('../input/test.csv','r')
test = csv.reader(testFile)
testHeader = str(next(test))
testData = []
outputFile = open('output.csv','w')
output = csv.writer(outputFile, lineterminator = '\n')
output.writerow(['PassengerId','Survived'])
for row in test:
    testData.append(row)
    if row[3] == 'male':
        output.writerow([row[0],'0'])
    elif row[3] == 'female':
        output.writerow([row[0],'1'])
testFile.close()
outputFile.close()

