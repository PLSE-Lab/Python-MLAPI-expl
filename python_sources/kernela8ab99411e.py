# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:35:54 2019

@author: Prakash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset_traain_set
dataset_t = pd.read_csv('../input/test.csv')
dataset_t.drop(["PassengerId","Name","Ticket","Cabin"],axis = 1,inplace = True)

#imputing missing values_train
dataset_t.info()

dataset_t.iloc[152,5]=dataset_t.iloc[:,5].mean()
fare_t = dataset_t.iloc[:, [0,5]].values
age_t = dataset_t.iloc[:, 2].values

fare_test_t = []
indices_t=[]
for i_t in range(0,len(age_t)):
    if (np.isnan(age_t[i_t])):
        fare_test_t.append(fare_t[i_t])
        indices_t.append(i_t)
        
    else:
        continue
    
fare_t = np.delete(fare_t,indices_t,axis = 0)    
age_t = np.delete(age_t,indices_t,axis = None)    


fare_test_t = np.array(fare_test_t)
fare_test_t = fare_test_t.reshape(-1,2)




from sklearn.linear_model import LinearRegression
regressor_t = LinearRegression()
regressor_t.fit(fare_t,age_t)

age_pred_t = regressor_t.predict(fare_test_t)
j=0
for i in range(0,418):
    if (np.isnan(dataset_t.iloc[i,2])):
        dataset_t.iloc[i,2]=age_pred_t[j]
        j=j+1
#Dropping unnecessary features    
#Final_data_to_train_system
dataset_t.info()
I_t= dataset_t.iloc[:,[0,1,2,3,4,5,6]].values
#categorical_feature_processing_train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_I_t = LabelEncoder()
I_t[:, 1] = labelencoder_I_t.fit_transform(I_t[:, 1])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_I_t1 = LabelEncoder()
I_t[:, 6] = labelencoder_I_t1.fit_transform(I_t[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [1,6])
I_t = onehotencoder.fit_transform(I_t).toarray()



# Importing the dataset_train_set
dataset = pd.read_csv('../input/train.csv')

#Dropping unnecessary features    
dataset.drop(["PassengerId","Name","Ticket","Cabin"],axis = 1,inplace = True)
common_value = 'S'
dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

#imputing missing values_train
fare = dataset.iloc[:, [1,6]].values
age = dataset.iloc[:, 3].values
fare_test = []
indices=[]
for i in range(0,len(age)):
    if (np.isnan(age[i])):
        fare_test.append(fare[i])
        indices.append(i)
        
    else:
        continue
    
fare = np.delete(fare,indices,axis = 0)    
age = np.delete(age,indices,axis = None)    

fare_test = np.array(fare_test)
fare_test = fare_test.reshape(-1,2)


    

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(fare, age)

age_pred = regressor.predict(fare_test)
j=0
for i in range(0,891):
    if (np.isnan(dataset.iloc[i,3])):
        dataset.iloc[i,3]=age_pred[j]
        j=j+1
       
    else:
        continue


dataset.info()
#Final_data_to_train_system
O= dataset.iloc[:,0].values
I= dataset.iloc[:,[1,2,3,4,5,6,7]].values
#categorical_feature_processing_train
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_I = LabelEncoder()
I[:, 1] = labelencoder_I.fit_transform(I[:, 1])

labelencoder_I1 = LabelEncoder()
I[:, 6] = labelencoder_I1.fit_transform(I[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [1,6])
I = onehotencoder.fit_transform(I).toarray()



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(I,O)

Y_predictiont1 = random_forest.predict(I_t)

random_forest.score(I,O)
acc_random_forest = round(random_forest.score(I, O) * 100, 2)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, I, O, cv=3)
confusion_matrix(O, predictions)

from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(O, predictions))
print("Recall:",recall_score(O, predictions))

from sklearn.metrics import f1_score
f1_score(O, predictions)

from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(I)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(O, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(O, y_scores)
print("ROC-AUC-Score:", r_a_score)



dataset_1t = pd.read_csv('../input/test.csv')
x=dataset_1t.iloc[:,0]

df = pd.DataFrame(x)
l=[]
for i in range(0,418):
     l.append(Y_predictiont1[i])
     
df['Survived']=l



df.to_csv('prediction.csv')



