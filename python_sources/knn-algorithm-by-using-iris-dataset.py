import pandas as pd
data=pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
data
feature_columns = ['sepal_length', 'sepal_width','petal_length','petal_width']
X = data[feature_columns].values
y = data['species'].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
classes = {0:'setosa',1:'versicolor',2:'virginica'}
x_new = [[1,1,2,2],
         [3,1,4,4]]
y_predict = knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])
