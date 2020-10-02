import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the training data
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

#seperating the label from training data
label = df.label
df.drop('label', axis=1, inplace=True)

#training a svc model 
from sklearn.svm import SVC
svc = SVC()
svc.fit(df, label)

#reading test data
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#making prediction on the test data using the trained svc model
predictions = svc.predict(test)

#converting the predictions in the required format
table = []
for i in range(0,len(predictions)):
    row = {}
    row['ImageId'] = i+1
    row['Label'] = predictions[i]
    table.append(row)
answer = pd.DataFrame(table)
    
#saving the prediction in a csv file
answer.to_csv("DigitRecognizer_answer.csv",index=False)