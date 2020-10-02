# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
# -*- coding: utf-8 -*-


import os as os
import pandas as pd
 
import matplotlib.pyplot as plt
import seaborn as sns  
 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv("creditcardcsvpresent.csv")
dataset.info()
dataset.head(6)
# Déterminer si les attribut contiennent des valuers nulles
dataset.isnull().any()


dataset=dataset.drop(labels=['Transaction date', 'Merchant_id'], axis=1)
dataset.info()
# Rendre les attributs catégorale en attributs binaire
dataset = dataset.replace(to_replace={'N':0,'Y':1})

# Visualisation des données
sns.countplot("isFradulent",data=dataset)
sns.heatmap(dataset.corr(),cmap='RdBu')

fraud = dataset.loc[dataset.isFradulent == 1]
declined = fraud.groupby(['Is declined']).sum()

f, axes = plt.subplots(1,1, figsize=(8,8))
axes.set_title("% of fraud transaction declined")
declined.plot(kind='pie',y='isFradulent',ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Declined','Declined'])
plt.show()

# Partitionnement du jeu de donnée en apprentissage et test
dataset=dataset.drop(labels=['Average Amount/transaction/day', '6-month_chbk_freq','6_month_avg_chbk_amt','Daily_chargeback_avg_amt'], axis=1)
X = dataset.drop(['isFradulent'],axis=1)
y = dataset['isFradulent']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


"---------------------    Decision Tree    ------------------------------"
model4 = DecisionTreeClassifier()
model4.fit(x_train,y_train)
DT=model4.predict(x_test)
acc4=accuracy_score(y_test,DT)

print("Decision Tree Accuracy: %.2f%%" %(acc4*100.0))

"---------------------  Logistic Regression ------------------------------"

model3= LogisticRegression()
model3.fit(x_train,y_train)
LR=model3.predict(x_test)
acc3=accuracy_score(y_test,LR)

print("LogisticRegression Accuracy: %.2f%%" %(acc3*100.0))

"---------------------     Random Forest    ------------------------------"
model2= RandomForestClassifier(n_estimators=15)
RF = model2.fit(x_train, y_train.values.ravel()).predict(x_test)
acc2= accuracy_score(y_test,RF)

print("Random Forest Accuracy: %.2f%%" %(acc2*100.0))

#################################################################### 


    
print("   ==========================================    ")
  
print("                 Models' accuracy                ")

print("Decision Tree Accuracy:         %.2f%%" %(acc4*100.0))
print("LogisticRegression Accuracy:    %.2f%%" %(acc3*100.0)) 
print("Random Forest Accuracy:         %.2f%%" %(acc2*100.0))
 
print("   ==========================================    ")  
  
