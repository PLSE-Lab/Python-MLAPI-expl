import numpy
import csv
from sklearn.ensemble import RandomForestClassifier

# load train data into a numpy array
train_data = numpy.loadtxt(open("../input/train.csv","r"),delimiter=",",skiprows=1)
train_features = train_data[:,1:]
train_labels = train_data[:,0]

#load test features into a numpy array
test_features = numpy.loadtxt(open("../input/test.csv","r"),delimiter=",",skiprows=1)

#the very basic RandomForestClassifier model, with no alterations
model = RandomForestClassifier()
model.fit(train_features, train_labels)

file = open('predicted_labels.csv','w',newline = '')

writer = csv.writer(file, delimiter=',')
writer.writerow(['ImageId','Label'])

cnt = 1

#for each test feature - make a prediction
for features_set in test_features:
    writer.writerow([str(cnt),str(int(round(model.predict(features_set)[0])))])
    cnt +=1
    
file.close()


