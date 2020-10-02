import pandas as pd 
data=pd.read_csv('../input/iris/Iris.csv')
data
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = data[feature_columns].values
y = data['Species'].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_train
X_test
y_train
y_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range=range(1,26)
scores={}
scores_list=[]
for k in k_range:
     knn=KNeighborsClassifier(n_neighbors=k)
     knn.fit(X_train,y_train)
     y_pred=knn.predict(X_test)
     scores[k]=metrics.accuracy_score(y_test,y_pred)
     scores_list.append(metrics.accuracy_score(y_test,y_pred))
classes={0:'setosa',1:'versicolor',2:'virginica'}  
x_new=[[3,4,5,5],[5,4,2,5]]
y_predict=knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])
