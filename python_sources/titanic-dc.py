import csv as csv
import numpy as np

train = csv.reader(open('../input/train.csv','rt'))
header = next(train)

data = []
for row in train:
    data.append(row)
print(type(data))
print(data[0])
data = np.array(data)
print(type(data))
print(data[0])


class_nos = len(np.unique(data[0::,2]))
fare_ceiling = 40
fare_bracket = 10
fare_nos = fare_ceiling // fare_bracket
data[data[0::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0
print(class_nos)
print(fare_nos)
survival_table = np.zeros([2,class_nos,fare_nos],float)
print(survival_table)

for i in range(class_nos):
    for j in range(fare_nos):
        women = data[(data[0::,4] == "female") \
              & (data[0::,2].astype(np.float) == i+1) \
              & (data[0::,9].astype(np.float) >= j*fare_bracket) \
              & (data[0::,9].astype(np.float) < (j+1)*fare_bracket) , 1]

        men = data[(data[0::,4] != "female") \
              & (data[0::,2].astype(np.float) == i+1) \
              & (data[0::,9].astype(np.float) >= j*fare_bracket) \
              & (data[0::,9].astype(np.float) < (j+1)*fare_bracket) , 1]
              
        survival_table[0,i,j] = np.mean(women.astype(np.float))
        survival_table[1,i,j] = np.mean(men.astype(np.float))    
    
print(survival_table)
print(type(survival_table))

survival_table[survival_table != survival_table] = 0
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1
print(survival_table)

test = open('../input/test.csv','rt')
test_object = csv.reader(test)
header = next(test_object)

predict = open("GenderClassFare.csv", 'wt')
predict_object = csv.writer(predict)
predict_object.writerow(["PassengerID","Survived"])

for row in test_object:
    for j in range(fare_nos):
        try:
            row[8] = float(row[8])
        except:
            fare_bin = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            fare_bin = fare_nos - 1
            break
        if row[8] >= j*fare_bracket \
        and row[8] < (j+1)*fare_bracket:
            fare_bin = j
            break
    
    if row[3] == "female":
        predict_object.writerow([row[0], "%d" % int(survival_table[ 0, float(row[1]) - 1, fare_bin])])
    else:
        predict_object.writerow([row[0], "%d" % int(survival_table[ 1, float(row[1]) - 1, fare_bin])])
        
test.close()
predict.close()














