
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

train_set = pd.read_csv('../input/mercedesbenz-greener-manufacturing/train.csv')
test_set = pd.read_csv('../input/mercedesbenz-greener-manufacturing/test.csv')

clean_df = train_set.copy()

for f in clean_df.columns:
    if clean_df[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(clean_df[f].values))
        clean_df[f] = label.transform(list(clean_df[f].values))
clean_df.head()

train_y = clean_df.y.values
train_x = clean_df.drop(["y"],axis=1)
train_x = train_x.drop(["ID"],axis=1)
train_x = train_x.values

model = xgb.XGBRegressor()
model.fit(train_x,train_y)
print (model)


id_vals = test_set.ID.values

clean_test = test_set.copy()
for f in clean_test.columns:
    if clean_test[f].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(clean_test[f].values))
        clean_test[f] = label.transform(list(clean_test[f].values))
clean_test.fillna((-999), inplace=True)
test = clean_test.drop(['ID'],axis=1)
x_test = test.values

output = model.predict(data=x_test)
final_df = pd.DataFrame()
final_df["ID"] = id_vals
final_df["Prediction_by_XGB"] = output
final_df.head()


