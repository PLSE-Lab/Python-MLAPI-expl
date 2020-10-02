import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv(r"../input/data.csv")
df=df.drop("Unnamed: 32",axis=1)
df=df.set_index("id")
encoder=LabelEncoder()
df.diagnosis=encoder.fit_transform(df.diagnosis)
x=df.drop("diagnosis",axis=1)
y=df["diagnosis"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
algo=KNeighborsClassifier(n_neighbors=11)
algo.fit(x_train,y_train)
values=algo.predict(x_test)
print(accuracy_score(y_test,values))
print(classification_report(y_test,values))
