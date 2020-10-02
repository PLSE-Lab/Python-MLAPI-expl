
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/mushrooms.csv")

#print(dataset)
#print(dataset.iloc[0:10,0:1]) #rows then columns 

clf = MLPClassifier(solver = "lbfgs",hidden_layer_sizes = (30,20)) #Standard neural net classifier with 2 hidden layers, each with 30 and 20 neurons in their layers respectivley

y_data = dataset[['class']] #All labels each label is p or e

labels = np.array([])

for i in range(0,8124):
    char = y_data.iloc[i,0]

    if char == 'p':
        labels = np.append(labels, 1) #changing from a string label to int
    elif char == 'e':
        labels = np.append(labels, 0)
    
#print(labels)
#8124 rows and 23 columns

features = dataset.as_matrix() #Convert data frame to numpy array
features = np.delete(features, 0, 1) #delete first column, this column contains the labels 
#print(features)

for i in range(0,8124):
    for j in range(0,22):
        features[i][j] = ord(features[i][j]) #Convert feature names to integers, using their ascii codes 

#print(features)
    
spit_point = 7311 #90% of data for training, 10% for testing 
labels_train = labels[:7311]
features_train = features[:7311]

labels_test = labels[7311:]
features_test = features[7311:]
clf = clf.fit(features_train,labels_train)

print(clf.score(features_test, labels_test))

predictions = clf.predict(features_test)

count = 0

for i in range(0,predictions.size):
    if predictions[i] == labels_test[i]:
        count += 1
        
print(count ," correct out of ", predictions.size)

#Accuracy 99.2% 
