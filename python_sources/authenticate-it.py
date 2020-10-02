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
data = pd.read_csv("../input/BankNote_Authentication.csv")

data.head()

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = 'class',data=data)

#data preparation

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(data.drop('class',axis=1))
scale_features = scalar.transform(data.drop('class',axis = 1))

df_feat = pd.DataFrame(scale_features,columns = data.columns[:-1])
df_feat.head()


#train_test_split

X = df_feat
y = data['class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train , y_test = train_test_split(X,y,test_size = 0.3)

import tensorflow as tf

df_feat.columns

feat_cols = []

for col in df_feat.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
    
    
feat_cols

classifier = tf.estimator.DNNClassifier(hidden_units= [10,20,10],n_classes = 2, feature_columns= feat_cols)

input_func = tf.estimator.inputs.pandas_input_fn(x= X_train,y=y_train,batch_size = 20, shuffle = True)

classifier.train(input_fn= input_func,steps = 500)

pred_fn = tf.estimator.inputs.pandas_input_fn(x= X_test,y=y_test,batch_size= len(X_test), shuffle = False)

note_predictions = list(classifier.predict(input_fn= pred_fn))

note_predictions[0]

final_pred = []
for note in note_predictions:
    final_pred.append(note["class_ids"][0])
    

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(confusion_matrix(y_test,final_pred))

print(classification_report(y_test,final_pred))

#99% accuracy 
