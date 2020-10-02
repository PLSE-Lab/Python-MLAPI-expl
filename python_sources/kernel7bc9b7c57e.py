from sklearn import datasets
from sklearn import model_selection as ms
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
iris=datasets.load_iris() #loads iris data
x=iris.data
y=iris.target

#splitting the dataset into train and test (70:30)
x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=.3,random_state=1)

k=range(1,105)
scores=[]

#finds accuracy scores for the given k values
for i in k:
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
scores