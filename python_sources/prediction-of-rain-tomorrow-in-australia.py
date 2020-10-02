# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:15:41 2019

@author: Osama Ahmed
"""

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




data = pd.read_csv("weatherAUS.csv")

print('Size of weather data frame is :',data.shape)

print(data.head())
print(data.info())

print(data.tail())
print(data.sample(5))
print(data.columns)

print(data.dtypes)
print(data.count().sort_values())


data = data.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
data = data.dropna(how='any')
print("DAta is")
print(data.isnull().sum())

print(data.shape)



plt.figure(figsize=(8,8))
sns.countplot(data=data,x='WindGustDir')





plt.figure(figsize=(8,8))
sns.countplot(data=data,x='WindDir3pm')

plt.figure(figsize=(8,8))
sns.countplot(data=data,x='RainToday')


plt.figure(figsize=(8,8))
sns.countplot(data=data,x='RainTomorrow')

plt.figure(figsize=(8,8))
sns.FacetGrid(data, hue="RainTomorrow", size=8).map(sns.kdeplot, "MinTemp").add_legend()
plt.ioff() 
plt.show()

plt.figure(figsize=(8,8))
sns.boxplot(data=data,x="RainTomorrow",y="MinTemp")



plt.figure(figsize=(8,8))
sns.FacetGrid(data, hue="RainTomorrow", size=8).map(sns.kdeplot, "MaxTemp").add_legend()
plt.ioff() 
plt.show()



plt.figure(figsize=(8,8))
sns.boxplot(data=data,x="RainTomorrow",y="MaxTemp")


plt.figure(figsize=(8,8))
sns.FacetGrid(data, hue="RainTomorrow", size=8).map(sns.kdeplot, "Rainfall").add_legend()
plt.ioff() 
plt.show()



plt.figure(figsize=(8,8))
sns.boxplot(data=data,x="RainTomorrow",y="Rainfall")



plt.figure(figsize=(8,8))
sns.FacetGrid(data, hue="RainTomorrow", size=8).map(sns.kdeplot, "WindGustSpeed").add_legend()
plt.ioff() 
plt.show()


plt.figure(figsize=(8,8))
sns.boxplot(data=data,x="RainTomorrow",y="WindGustSpeed")

plt.figure(figsize=(8,8))
sns.FacetGrid(data, hue="RainTomorrow", size=8).map(sns.kdeplot, "Humidity9am").add_legend()
plt.ioff() 
plt.show()













##



from scipy import stats
z = np.abs(stats.zscore(data._get_numeric_data()))
print(z)
data= data[(z < 3).all(axis=1)]
print(data.shape)







data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)


categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_columns:
    print(np.unique(data[col]))
    
    


data = pd.get_dummies(data, columns=categorical_columns)
data.iloc[4:9]



from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
data.iloc[4:10]






from sklearn.feature_selection import SelectKBest, chi2
X = data.loc[:,data.columns!='RainTomorrow']
y = data[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)]) 



data =data[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
X = data[['Humidity3pm']] 
y = data[['RainTomorrow']]





#Logistic Regression 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_logreg = LogisticRegression(random_state=0)
clf_logreg.fit(X_train,y_train)
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test,y_pred)


print("x-train data shape is",X_train.shape)
print("y-train data shape is",y_train.shape)
print("x-pred data shape is",X_test.shape)
print("y-pred data shape is",y_test.shape)


print("Logistic Regression \n")
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)





#Random Forest Classifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
clf_rf.fit(X_train,y_train)
y_pred = clf_rf.predict(X_test)
score = accuracy_score(y_test,y_pred)
print("Random Forest Classifier \n")
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)

from sklearn.metrics import classification_report
print(classification_report(y_test,clf_rf.predict(X_test)))

from sklearn.metrics  import confusion_matrix
import seaborn as sns
f_cm=confusion_matrix(y_test,y_pred)
sns.heatmap(f_cm,annot=True,fmt='.2f',xticklabels=["rain","no rain"],yticklabels=["rain","no rain"])
plt.ylabel("True class")
plt.xlabel("predicted class")

plt.title("Random Forsest")




from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train,y_train)
y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test,y_pred)
print("Decision Tree Classifier\n")
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)




import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()




mpl.style.use('ggplot')
plt.figure(figsize=(6,4))
plt.hist(data['RainTomorrow'],bins=2,rwidth=0.8)
plt.xticks([0.25,0.75],['No Rain','Rain'])
plt.title('Frequency of No Rain and Rainy days\n')
print(data['RainTomorrow'].value_counts())











#Support vector machine
from sklearn import svm
from sklearn.model_selection import train_test_split

t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_svc = svm.SVC(kernel='linear')
clf_svc.fit(X_train,y_train)
y_pred = clf_svc.predict(X_test)
score = accuracy_score(y_test,y_pred)
print("Support Vector Machine\n")
print('Accuracy :',score)
print('Time taken :' , time.time()-t0)







#y_prob_rain = rf.predict_proba(X_test)

# To convert x-axis to a percentage
from matplotlib.ticker import PercentFormatter

# Plot histogram of predicted probabilities
fig,ax = plt.subplots(figsize=(10,6))
plt.hist(y_prob_rain[:,1],bins=50,alpha=0.5,color='teal',label='Rain')
plt.hist(y_prob_rain[:,0],bins=50,alpha=0.5,color='orange',label='No Rain')
plt.xlim(0,1)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability (%)')
plt.ylabel('Frequency')

ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(0.025,0.83,'n = 33,878',transform=ax.transAxes)

plt.legend()
##