"""
    @script_author: Devika M R
    @script_name:  K-Nearest Neighbors (KNN) Algorithm.
    @script_description: The program  predit the target values using the Iris dataset.
    @script_package_used: sklearn
"""
#import the load_iris function from datasets module
from sklearn.datasets import load_iris
import sklearn.model_selection as ms
import sklearn.metrics as metrics            #import metrics model to check the accuracy
from sklearn.neighbors import KNeighborsClassifier #KNN package
from sklearn.model_selection  import cross_val_score
#load dataset
iris=load_iris()
x=iris.data
y=iris.target
type(iris)
#splitting the data into training and test sets
x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.24,random_state=4)
#try running from k=1 through 31 and record testing acuracy
knn=KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
k_range=range(1,31)
k_score=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
    k_score.append(scores.mean())
print (k_score)
