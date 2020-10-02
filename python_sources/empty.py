import numpy as np
import pandas as pd
import csv as csv
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
csv_file_object = csv.reader(open('../input/train.csv')) 
header = next(csv_file_object)
data=[] 

for row in csv_file_object:
    data.append(row)
data = np.array(data)
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

first_class_stats = data[0::,2] == "1"
second_class_stats = data[0::,2] == "2"
third_class_stats = data[0::,2] == "3"
first_class = data[first_class_stats,1].astype(np.float)     
second_class = data[second_class_stats,1].astype(np.float)
third_class = data[third_class_stats,1].astype(np.float)

age_stats = [[],[],[],[],[],[]]
for i in data:
    if (i[5] < '20'):
        age_stats[0].append(True)
        age_stats[1].append(False)
        age_stats[2].append(False)
        age_stats[3].append(False)
        age_stats[4].append(False)
        age_stats[5].append(False)
    elif (i[5] < '30'):
        age_stats[0].append(False)
        age_stats[1].append(True)
        age_stats[2].append(False)
        age_stats[3].append(False)
        age_stats[4].append(False)
        age_stats[5].append(False)
    elif (i[5] < '40'):
        age_stats[0].append(False)
        age_stats[1].append(False)
        age_stats[2].append(True)
        age_stats[3].append(False)
        age_stats[4].append(False)
        age_stats[5].append(False)
    elif (i[5] < '50'):
        age_stats[0].append(False)
        age_stats[1].append(False)
        age_stats[2].append(False)
        age_stats[3].append(True)
        age_stats[4].append(False)
        age_stats[5].append(False)
    elif (i[5] < '60'):
        age_stats[0].append(False)
        age_stats[1].append(False)
        age_stats[2].append(False)
        age_stats[3].append(False)
        age_stats[4].append(True)
        age_stats[5].append(False)
    else:
        age_stats[0].append(False)
        age_stats[1].append(False)
        age_stats[2].append(False)
        age_stats[3].append(False)
        age_stats[4].append(False)
        age_stats[5].append(True)
ages = [0,0,0,0,0,0]
for i in range(len(ages)):
    ages[i] = data[age_stats[i],1].astype(np.float)
# Then we finds the proportions of them that survived
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard) 
proportion_first_survived = \
                       np.sum(first_class) / np.size(first_class)
proportion_second_survived = \
                       np.sum(second_class) / np.size(second_class)
proportion_third_survived = \
                       np.sum(third_class) / np.size(third_class)
proportion_child_survived = \
                       np.sum(ages[0]) / np.size(ages[0])
proportion_20s_survived = \
                       np.sum(ages[1]) / np.size(ages[1])
proportion_30s_survived = \
                       np.sum(ages[2]) / np.size(ages[2])
proportion_40s_survived = \
                       np.sum(ages[3]) / np.size(ages[3])
proportion_50s_survived = \
                       np.sum(ages[4]) / np.size(ages[4])
proportion_60_plus_survived = \
                       np.sum(ages[5]) / np.size(ages[5])
# and then print it out
print('Number of passengers aboard is %s' % number_passengers)
print('Number of passengers who survived is %s' % number_survived)
print('Proportion of people who survived is %s' % proportion_survivors)
print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)
print('Proportion of first class who survived is %s' % proportion_first_survived)
print('Proportion of second class who survived is %s' % proportion_second_survived)
print('Proportion of third class who survived is %s' % proportion_third_survived)
print('Proportion of children who survived is %s' % proportion_child_survived)
print('Proportion of people in their twenties who survived is %s' % proportion_20s_survived)
print('Proportion of people in their thirties who survived is %s' % proportion_30s_survived)
print('Proportion of people in their fourties who survived is %s' % proportion_40s_survived)
print('Proportion of people in their fifties who survived is %s' % proportion_50s_survived)
print('Proportion of people above sixty who survived is %s' % proportion_60_plus_survived)
test_file = open('../input/test.csv', 'r')
test_file_object = csv.reader(test_file)
next(test_file_object)
prediction_file = open("genderbasedmodel.csv", "w")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object: # For each row in test.csv
    if (row[4] == 'female'):
        gender = proportion_women_survived
    else:
        gender = proportion_men_survived
    if (row[2] == '1'):
        pass_class = proportion_first_survived
    elif (row[2] == '2'):
        pass_class = proportion_second_survived
    else:
        pass_class = proportion_third_survived
    if (row[5] < '20'):
        age = proportion_child_survived
    elif (row[5] < '30'):
        age = proportion_20s_survived
    elif (row[5] < '40'):
        age = proportion_30s_survived
    elif (row[5] < '50'):
        age = proportion_40s_survived
    elif (row[5] < '60'):
        age = proportion_50s_survived
    else:
        age = proportion_60_plus_survived
    if (gender + pass_class + age >= 1):
        prediction_file_object.writerow([row[0],'1'])
    else:
        prediction_file_object.writerow([row[0],'0'])
test_file.close()
prediction_file.close()