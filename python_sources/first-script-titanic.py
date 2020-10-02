import numpy as np
import pandas as pd
import csv as csv
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64,"Pclass": np.float64},)
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64,"Pclass": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)
data = np.array(train)
nb_passengers = np.size(data[0::,1].astype(np.float))
nb_survivors = np.sum(data[0::,1].astype(np.float))
proportion_survivors = nb_survivors/nb_passengers
print(proportion_survivors)
women_stats = data[0::,4] == "female"
men_stats = data[0::,4] != "female"
women_onboard = data[women_stats,1].astype(np.float)
men_onboard = data[men_stats,1].astype(np.float)
prop_women_survivors = np.sum(women_onboard)/np.size(women_onboard)
prop_men_survivors = np.sum(men_onboard)/np.size(men_onboard)
print('Proportion of women who survived is %s' % prop_women_survivors)
print('Proportion of men who survived is %s' % prop_men_survivors)

test_file = open("../input/test.csv","rt")
test_file_object = csv.reader(test_file)
next(test_file)
prediction_file = open("genderbasedmodel.csv", "wt")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId","Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()
