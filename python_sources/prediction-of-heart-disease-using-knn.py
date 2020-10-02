#Used libraries during the whole project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Reading csv file using pandas
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
print(df.head())

#Counting no. of 'target' attribute which can be 1 for presence heart disease and 0 for its absence
print(df['target'].value_counts())

#plotting a histogram to show 'target' distribution
df.hist('target', bins=20)
plt.show()

#Printing columns titles
print(df.columns)

#Converting dataframe to array so it can be used with sklearn
x = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']]

##The variable that contains array of 'target' values
y = df['target'].values

#Normaalizing data for good practice( zero mean and unit variance)
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

#Train Test Split equation with setting test size to be 20% of the dataset
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state =4)

print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)

#A loop that test the accutacy of diferent values of k to choose the one with the best accuracy
ks = 10

mean_acc=np.zeros((ks-1))
std_acc=np.zeros((ks-1))
confusionMx=[];

for n in range(1,ks):
    #training
    neigh = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)
    #prediction
    yhat = neigh.predict(x_test)
    #Accuracy evaluation
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

print('Mean Accuracy for ks=1:9 consecutively: ',mean_acc)

#plotting each k value aganist its accuracy
plt.plot(range(1, ks), mean_acc, 'g')
plt.fill_between(range(1, ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax() + 1)