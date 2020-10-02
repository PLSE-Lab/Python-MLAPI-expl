import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

link=("../input/iris/Iris.csv")
names1=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data=pd.read_csv(link,names=names1)
data

x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
print(error)
min(error)
    
    
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='green', linestyle='dashed', marker='*',
         markerfacecolor='yellow', markersize=15)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')    