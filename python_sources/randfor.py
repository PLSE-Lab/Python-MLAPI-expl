

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
df_x=train.iloc[:,1:]
df_y=train.iloc[:,0]

rf=RandomForestClassifier(n_estimators=100)
rf.fit(df_x,df_y)

pred=rf.predict(test)

print (pred)