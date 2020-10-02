import numpy as np
import pandas as pd

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

import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../input/train.csv', 'rU')) 
header = next(csv_file_object)  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format
#print(data[0::,2].astype(np.float))           
print(data[0])

num_of_passengers = np.size(data[0::,1].astype(np.float))
num_of_survivors = np.sum(data[0::,1].astype(np.float))
prob = num_of_survivors/num_of_passengers
print(prob)

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

prob_women = np.sum(women_onboard) / np.size(women_onboard)
prob_men = np.sum(men_onboard) / np.size(men_onboard)

print ('Women Prob - %s' % prob_women)
print ('Men Prob - %s' % prob_men)

test_file = open('../input/test.csv', 'rU')
test_file_object = csv.reader(test_file)
header = next(test_file_object)

prediction_file = open("genderbasedmodel_py.csv", "w")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:       # For each row in test.csv
    if row[3] == "female":         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close()
prediction_file.close()

# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0
#print(data[0::,9])

fare_bracket_size = 10
number_of_price_brackets = (int)(fare_ceiling / fare_bracket_size)

# I know there were 1st, 2nd and 3rd classes on board
number_of_classes = 3

# But it's better practice to calculate this from the data directly
# Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::,2])) 

# Initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in range(number_of_classes):       #loop through each class
    for j in range(number_of_price_brackets):   #loop through each price bin

        women_only_stats = data[                          #Which element           
                         (data[0::,4] == "female")    #is a female
                       &(data[0::,2].astype(np.float) #and was ith class
                             == i+1)                        
                       &(data[0:,9].astype(np.float)  #was greater 
                            >= j*fare_bracket_size)   #than this bin              
                       &(data[0:,9].astype(np.float)  #and less than
                            < (j+1)*fare_bracket_size)#the next bin    
                          , 1]                        #in the 2nd col                           
 						                                    									


        men_only_stats = data[                            #Which element           
                         (data[0::,4] != "female")    #is a male
                       &(data[0::,2].astype(np.float) #and was ith class
                             == i+1)                                       
                       &(data[0:,9].astype(np.float)  #was greater 
                            >= j*fare_bracket_size)   #than this bin              
                       &(data[0:,9].astype(np.float)  #and less than
                            < (j+1)*fare_bracket_size)#the next bin    
                          , 1] 

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
        
survival_table[ survival_table != survival_table ] = 0
print(survival_table)

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 

test_file = open('../input/test.csv', 'rU')
test_file_object = csv.reader(test_file)
header = next(test_file_object)
predictions_file = open("genderclassmodel_py.csv", "w")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:                 # We are going to loop
                                              # through each passenger
                                              # in the test set                     
    for j in range(number_of_price_brackets):  # For each passenger we
                                              # loop thro each price bin
        try:                                      # Some passengers have no
                                              # Fare data so try to make
            row[8] = float(row[8])                  # a float
        except:                                   # If fails: no data, so 
            bin_fare = 3 - float(row[1])            # bin the fare according Pclass
            break                                   # Break from the loop
    
        if row[8] > fare_ceiling:              # If there is data see if
                                              # it is greater than fare
                                              # ceiling we set earlier
            bin_fare = number_of_price_brackets-1   # If so set to highest bin
            break                                   # And then break loop
    
        if row[8] >= j * fare_bracket_size\
        and row[8] < \
        (j+1) * fare_bracket_size:             # If passed these tests 
                                              # then loop through each bin 
            bin_fare = j                            # then assign index
            break                                   
    if row[3] == 'female':                             #If the passenger is female
        p.writerow([row[0], "%d" % \
                   int(survival_table[0, float(row[1])-1, bin_fare])])
    else:                                          #else if male
        p.writerow([row[0], "%d" % \
                   int(survival_table[1, float(row[1])-1, bin_fare])])
     
# Close out the files.
test_file.close() 
predictions_file.close()