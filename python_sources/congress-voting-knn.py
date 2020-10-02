import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#the programs attempts to predict the party('democrat or republic') to which a congressman belongs by analyzing his vating on various bills.
#There are total 16 bills for which the 435 congressmen have voted and the party to which they belonged. Using this data we are trying to predict
#the party of a new congressman on the basis of his votes to various bills. I have used the K Nearest Neighbors algorithm.

df = pd.read_csv('../input/voting_data.csv')
y = df['party'].values
X = df.drop('party', axis=1).values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, stratify = y, random_state = 21)

knn =  KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)


df_test=pd.read_csv('../input/testdata.csv', header=None)    # add your custom test cases to this file
y_pred_file = knn.predict(df_test)    #y_pred_file has the predicted parties of the congressmen whose data is present in the testdata.csv file.
y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test))

print("Prediction: {}".format(y_pred_file))



#now let's analyze the performance of the algorithm for different values of K.
#The following code draws a line plot between efficiency and different values of k for both testing and training data

neighbors = np.arange(1,20)
train_accuracy = np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))

for i,k in enumerate(neighbors) :
    knn = KNeighborsClassifier(k)
    knn.fit(X,y)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)


plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show(block = True)
plt.savefig('fig1.pdf')
 # the image is saved as pdf file in the working directory
