#importing the dataset:
import pandas as pd
a=pd.read_csv("../input/heart-disease-uci/heart.csv")
print(a.head(n=6))
print(a.info())
#checking for correlation using spearmans method:
c=a.corr(method="spearman")
print(c)
#using pairplot to show the distribution of the dataset:
import seaborn as sns
sns.pairplot(a)
print(a.columns)
#slicing x and y:
x=a.drop(["target"],axis=1)
y=a.drop(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'],axis=1)
print(y)
#splitting training and testing set in the order (80,20):
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
print(xtrain)
#steps to ignore dataconversion warnings:
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore",category=DataConversionWarning)
#fitting a logistic regression model:
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
model=lm.fit(xtrain,ytrain)
#showing the coefficient and intetrcept values:
print("The coefficient of the model is : " ,model.coef_)
print("The intercept of the model is : " ,model.intercept_)
#prdicting using the x_test values:
pred=model.predict(xtest)
print("The predicted values are : " ,pred)
#using confusion matrix to display the classification score:
from sklearn.metrics import confusion_matrix
d=confusion_matrix(ytest,pred)
print(d)
#classification model to show the accuracy of model and the f-score:
from sklearn import metrics
print(metrics.classification_report(ytest,pred))
