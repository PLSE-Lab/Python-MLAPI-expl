

import numpy as np
import pandas as pd
import csv as csv
import random

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)


csv_file_object = csv.reader(open('../input/train.csv', 'rU')) 
header = csv_file_object.__next__()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format
men = data[0::,4] != "female"
female = data[0::,4] == "female"

#print(data)
print('Total Test data Rates')

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = round(number_survived / number_passengers*100, 2)
print(str(number_passengers) + ' Total Passengers')
print(str(number_survived) + ' Total Survivors')
print(str(proportion_survivors) + ' Survivor Rate')

print('Men Data')

numPassangerM = np.size(data[men,1].astype(np.float))
numSurvivedM = np.sum(data[men,1].astype(np.float))
xx = round(numSurvivedM/numPassangerM*100,0) 
y = '%'
z = str(xx) + y 

print(str(numPassangerM) + ' Total Male Passangers')
print(str(numSurvivedM) + ' Total Male Survivors')
print(z + ' Male Survivors\n')

print('Women Data')


numPassangerF = np.size(data[female,1].astype(np.float))
numSurvivedF = np.sum(data[female,1].astype(np.float))

x = round(numSurvivedF/numPassangerF*100,0)
y = '%'
z = str(x) + y 


print(str(numPassangerF) + ' Total Female Passangers')
print(str(numSurvivedF) + ' Total Female Survivors')

print(z + ' Female Survivors\n')

##########Random Survivor Generator############
M1 = random.randrange(1,101,1)
if M1 > (100-xx): ##Men  If Random num > xx(survivor % for men) Then Man lived
    M1Y = 'Yes'
else: 
    M1Y = 'No' 
    
F1 = random.randrange(1, 101,1)
if F1 > (100-x):  ## If Random num > x (suvivior % for women) Then Women Lived

    F1Y = 'Yes'
else:
    F1Y = 'No' 

print(M1)

print('Did man live {}'.format(M1Y))
print(F1)
print('Did women live {}'.format(F1Y))


##############################################################################
##### Writing the model ######################################################


test_file = open('../input/test.csv', 'rU')

test_file_object = csv.reader(test_file)
header = test_file_object.__next__()

prediction_file = open('genderbasemodel1.csv', 'w')
prediction_file_object = csv.writer(prediction_file)



prediction_file_object.writerow(["PassengerId", "Survived", "Gender"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':# is it a female, if yes then
        M1 = random.randrange(1,101,1)
        if M1 > (100-x):
            M1Y = 1
        else: 
            M1Y = 0
        prediction_file_object.writerow([row[0], M1Y])    # Predict Female 
    else: # or else if male,       
        F1 = random.randrange(1,101, 1)
        if F1 > (100 - xx):
            F1Y = 1
        else: 
            F1Y = 0
    prediction_file_object.writerow([row[0], F1Y])    # Predict Male
test_file.close()
prediction_file.close()


csv_file_object = csv.reader(open('genderbasemodel1.csv', 'rU')) 
header = csv_file_object.__next__()  # The next() command just skips the 


                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			                     # Be aware that each item is currently

                                 # a string in this format



print('Predictive Test Data \n')

numPassanger = np.size(data[0::,0].astype(np.float))
numSurvived = np.sum(data[0::,1].astype(np.float))
xx = round(numSurvived/numPassanger*100,2) 
y = '%'
z = str(xx) + y 



print(numPassanger)
print(numSurvived)
print(z + '\n')


test_file.close()
prediction_file.close()















