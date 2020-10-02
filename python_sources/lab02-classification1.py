#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# Logistic regression is a classification algorithm that transforms its output using the logistic sigmoid function

# ## Sigmoid (Logistic) function
# ![](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

# ## Binary Classification

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# p i = 1 / 1 + exp[ - ( b0 + b1 * x )]

x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])


#column vector
X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],[3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
# print(X)


#row vector
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])
#print(y)


plt.plot(x1,y1,'ro',color='blue')
plt.plot(x2,y2,'ro',color='red')

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X,y)
print("Accuracy:", model.score(X, y))

print("b0 is:", model.intercept_)
print("b1 is:", model.coef_)

def logistic(classifier, x):
	return 1/(1+np.exp(-(model.intercept_ + model.coef_ * x)))
	
for i in range(1,120):
	plt.plot(i/10.0-2,logistic(model,i/10.0),'ro',color='green')

plt.axis([-2,10,-0.5,2])
plt.show()

x_test = np.array([-2, 0, 1, 8, 20])
x_test = x_test.reshape(-1, 1);
pred = model.predict_proba(x_test)
print("Prediction (probabilities): ", pred)
pred = model.predict(x_test)
print("Prediction: ", pred)


# ## Checkpoint 1
# From code above, predict new input values (x=[-2, 0, 1, 8, 20])

# In[ ]:


import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

file_path = "../input/titanic"
creditData = pd.read_csv(os.path.join(file_path,'titanic_data.csv'))

print(creditData.head())
print(creditData.describe())
print(creditData.corr())

features = creditData[["Fare","Pclass"]]
target = creditData.Survived

feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = LogisticRegression(solver='lbfgs', max_iter=10000)
model.fit = model.fit(feature_train, target_train)
predictions = model.predict(feature_test)


print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))


# ## Cross Validation

# In[ ]:


# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score

# file_path = "../input/"
# creditData = pd.read_csv(os.path.join(file_path,'credit_data.csv'))

# features = creditData[["income","age","loan"]]
# target = creditData.default

# model = LogisticRegression()
# predicted = cross_validate.cross_val_predict(model,features,target, cv=10)

# print(accuracy_score(target,predicted))


# ## Multi-class Classification

# In[ ]:


import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

file_path = "../input/titanic"
creditData = pd.read_csv(os.path.join(file_path,'titanic_data.csv'))

print(creditData.head())
print(creditData.describe())
print(creditData.corr())

features = creditData[["Pclass","Fare"]]
target = creditData.Survived
feature_train, feature_test, target_train, target_test = train_test_split(features,target, test_size=0.3)

model = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='auto', max_iter=10000)
model.fit(feature_train, target_train)
predictions = model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test,predictions))


# ## Checkpoint 2
# From the url https://medium.com/mmp-li/logistic-regression-%E0%B9%84%E0%B8%A1%E0%B9%88%E0%B8%A1%E0%B8%B5%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B9%80%E0%B8%9B%E0%B9%87%E0%B8%99%E0%B9%84%E0%B8%9B%E0%B8%95%E0%B8%B2%E0%B8%A1%E0%B8%AD%E0%B8%A2%E0%B9%88%E0%B8%B2%E0%B8%87%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%84%E0%B8%B4%E0%B8%94%E0%B9%80%E0%B8%AA%E0%B8%A1%E0%B8%AD-machine-learning-101-bba2f666234d
# 
# Implement logistic regression to predict survivors on the Titanic
