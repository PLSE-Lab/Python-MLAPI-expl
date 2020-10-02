import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

train_x = df.drop(columns=['Severity','Accident_ID'])

train_y = df['Severity']

train_y = pd.get_dummies(train_y)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_x = sc_X.fit_transform(train_x)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(train_x,train_y)

test_x = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')
tst_x = test_x.drop(columns=['Accident_ID'])
tst_x = sc_X.transform(tst_x)

y_pred = classifier.predict(tst_x)

dataframe = pd.DataFrame(data= y_pred,columns=['Highly_Fatal_And_Damaging','Minor_Damage_And_Injuries','Significant_Damage_And_Fatalities','Significant_Damage_And_Serious_Injuries'])                   
ans = dataframe.idxmax(axis=1)
submission = pd.DataFrame(data = test_x['Accident_ID'],columns =['Accident_ID','Severity']) 
submission['Severity'] = ans
submission.to_csv('submission.csv', index=False)


