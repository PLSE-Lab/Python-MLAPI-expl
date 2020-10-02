import pandas as pd 
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('/kaggle/input/realestatepriceprediction/train.csv')

X = df.drop('Price', axis=1)
y = df['Price']

X_final = pd.read_csv('/kaggle/input/realestatepriceprediction/test.csv')

preds_final = pd.DataFrame()
preds_final['Id'] = X_final['Id'].copy()

to_del_list = ["Id", "DistrictId", "LifeSquare", "Healthcare_1", "Ecology_2", "Ecology_3", "Shops_2"]
X.drop(to_del_list, axis=1, inplace=True)
X_final.drop(to_del_list, axis=1, inplace=True)

model = RandomForestRegressor(n_estimators=1000, max_depth=16, random_state=42, max_features=7)
model.fit(X, y)

y_pred_final = model.predict(X_final)

preds_final['Price'] = y_pred_final
preds_final.to_csv('predictions.csv', index=False)