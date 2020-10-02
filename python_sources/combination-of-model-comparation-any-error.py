import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from  sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../input/diamonds.csv", usecols=[1,2,3,4,5,6,7])

df3 = pd.read_csv("../input/diamonds.csv", usecols=[1,2,3,4,5,6,7])

df2=df


title_mapping = {"Ideal": 3, "Premium": 4, "Good": 1, "Fair": 5, "Very Good": 2}

df2['cut'] = df2['cut'].map(title_mapping)

title_mapping = {"J": 3, "H": 4, "E": 1, "F": 5, "I": 2,"D": 6,"G": 7}
df2['color'] = df2['color'].map(title_mapping)
   


title_mapping = {"SI1": 3, "SI2": 4, "VS2": 1, "VS1": 5, "I1": 2,"IF": 6,"VVS2": 7,"VVS1": 8}
df2['clarity'] = df2['clarity'].map(title_mapping)

df2.loc[ df2['price'] <= 500, 'price'] = 0  
df2.loc[(df2['price'] > 500) & (df2['price'] <= 1000), 'price'] = 1 
df2.loc[(df2['price'] > 1000) & (df2['price'] <= 2000), 'price'] = 2
df2.loc[(df2['price'] > 2000) & (df2['price'] <= 3000), 'price'] = 3
df2.loc[(df2['price'] > 3000) & (df2['price'] <= 4000), 'price'] = 4  
df2.loc[(df2['price'] > 4000) & (df2['price'] <= 5500), 'price'] = 5 
df2.loc[(df2['price'] > 5500) & (df2['price'] <= 7000), 'price'] = 6            
df2.loc[ df2['price'] > 7000, 'price'] = 7 

df2[['clarity', 'price']].groupby(['clarity'], as_index=False).mean()       





train_df, test_df = train_test_split(df2, test_size = 0.2)

X_train = train_df.drop("price", axis=1)
Y_train = train_df["price"]
X_test  = train_df.drop("price", axis=1)
Y_test = train_df["price"]
X_train.shape, Y_train.shape, X_test.shape



svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest



sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
