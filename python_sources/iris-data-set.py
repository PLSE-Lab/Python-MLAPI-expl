#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv("../input/Iris.csv")


# In[ ]:


data.head()


# In[ ]:


del data["Id"]


# In[ ]:


data.info()


# In[ ]:


species=data['Species'].unique()


# In[ ]:


listofcolumns=data.columns
print(listofcolumns)


# In[ ]:


listofcolumns=data.columns
listofNumericalcolumns=[]

for i in listofcolumns:
    if data[i].dtype == 'float64':
        listofNumericalcolumns.append(i)
print('listofNumericalcolumns :',listofNumericalcolumns)
print('Species:',species)



# In[ ]:


for i in range(len(listofNumericalcolumns)):
    for j in range(len(species)):  
        print(listofNumericalcolumns[i]," : ",species[j])


# In[ ]:


data.describe()


# In[ ]:


data.groupby("Species").size()


# In[ ]:


data.plot(kind='box')


# In[ ]:


data.hist(figsize=(10,5))
plt.show()


# In[ ]:


print("HIST PLOT OF INDIVIDUAL Species")
print(species)

for spice in species:
        data[data['Species']==spice].hist(figsize=(10,5))


# In[ ]:


sns.violinplot(data=data,x='Species',y='PetalLengthCm')


# In[ ]:


sns.violinplot(data=data,x='Species',y='PetalWidthCm')


# In[ ]:


sns.violinplot(data=data,x='Species',y='SepalLengthCm')


# In[ ]:


sns.violinplot(data=data,x='Species',y='SepalWidthCm')


# In[ ]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
#Importing the dataset 
d = data.iloc[:, :] 
  
#checking for null values 
print("Sum of NULL values in each column. ") 
print(d.isnull().sum()) 
  
#seperating the predicting column from the whole dataset 
X = d.iloc[:, :-1].values 
y = data.iloc[:, 4].values 
  
#Encoding the predicting variable 
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y) 
  
#Spliting the data into test and train dataset 
X_train, X_test, y_train, y_test = train_test_split( 
              X, y, test_size = 0.3, random_state = 0) 
  
#Using the random forest classifier for the prediction 
classifier=RandomForestClassifier() 
classifier=classifier.fit(X_train,y_train) 
predicted=classifier.predict(X_test) 
  
#printing the results 
print ('Confusion Matrix :') 
print(confusion_matrix(y_test, predicted)) 
print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
print ('Report : ') 
print (classification_report(y_test, predicted)) 

