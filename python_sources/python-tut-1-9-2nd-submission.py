# The first thing to do is to import the relevant packages
# that I will need for my script, 
# these include the Numpy (for maths and arrays)
# and csv for reading and writing csv files
# If i want to use something from this I need to call 
# csv.[function] or np.[function] first

import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('../input/train.csv', 'rU')) 
header = csv_file_object.__next__()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format
print(data)

# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

# I know there were 1st, 2nd and 3rd classes on board
number_of_classes = 3

# But it's better practice to calculate this from the data directly
# Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::,2])) 

# Initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))


for i in range(number_of_classes):       #loop through each class
  for j in range(number_of_price_brackets):   #loop through each price bin

    women_only_stats = data[                                     
                         (data[0::,4] == "female")    
                       &(data[0::,2].astype(np.float) 
                             == i+1)                        
                       &(data[0:,9].astype(np.float)   
                            >= j*fare_bracket_size)   
                       &(data[0:,9].astype(np.float)  
                            < (j+1)*fare_bracket_size)    
                          , 1]                                   
                          
    men_only_stats = data[                                      
                         (data[0::,4] != "female")    
                       &(data[0::,2].astype(np.float) 
                             == i+1)                                      
                       &(data[0:,9].astype(np.float)   
                            >= j*fare_bracket_size)                 
                       &(data[0:,9].astype(np.float)  
                            < (j+1)*fare_bracket_size)   
                          , 1]
                          
  survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 
  survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
  
  survival_table[ survival_table != survival_table ] = 0.
  
  print(survival_table)
  
  survival_table[ survival_table < 0.5 ] = 0
  survival_table[ survival_table >= 0.5 ] = 1 
  
  print(survival_table)
  
  test_file = open('../input/test.csv', 'rU')
  test_file_object = csv.reader(test_file)
  header = test_file_object.__next__()
  predictions_file = open("genderclassmodel.csv", "w")
  p = csv.writer(predictions_file)
  p.writerow(["PassengerId", "Survived"])
  
  for row in test_file_object:                                 
    for j in range(number_of_price_brackets):  # For each passenger we
      try:
        row[8] = float(row[8])                  # a float
      except:
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
  
