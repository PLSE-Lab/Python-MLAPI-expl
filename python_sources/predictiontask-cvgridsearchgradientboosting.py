# Team members: Stefano Barindelli, Matteo Sangiorgio. Task: Prediction

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import os
from dateutil.relativedelta import relativedelta


df = pd.read_csv('../input/train_electricity.csv')
test_df = pd.read_csv('../input/test_electricity.csv')

print("Dataset has", len(df), "entries.")

print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")
for col_name in df.columns:
    col = df[col_name]
    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")
    

def add_datetime_features(df):
    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
                "Hour", "Minute", "Quarter"]
    one_hot_features = ["Month", "Dayofweek", "Quarter"]

    datetime = pd.to_datetime(df.Date * (10 ** 9))

    df['Datetime'] = datetime  # <-- We won't use this for training, but we'll remove it later

    for feature in features:
        new_column = getattr(datetime.dt, feature.lower())
        if feature in one_hot_features:
            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)
        else:
            df[feature] = new_column
            
    return df

df = add_datetime_features(df)

print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")
for col_name in df.columns:
    col = df[col_name]
    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")


label_col = "Consumption_MW"  # The target values are in this column

###
to_drop = [label_col, "Date", "Datetime"] # Define input to be dropped
eval_from = df['Datetime'].max() + relativedelta(months=-24)  # Here we set the 6 months threshold
###

train_df = df[df['Datetime'] >= eval_from]
valid_df = df[df['Datetime'] <  eval_from]

print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")
print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")

# Create the parameter grid based on the results of random search 
param_grid = {
    'learning_rate': [0.1],
    'n_estimators': [2000],
    'max_depth': [15],
    'min_samples_split': [5],
    'min_samples_leaf': [5],
    'max_features': [int(np.sqrt(train_df.shape[1]))]
}

# Create a based model
gb = GradientBoostingRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, cv = 4, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train_df.drop(columns=to_drop).values, train_df[label_col].values)
model = grid_search.best_estimator_

print('\n')
print('learning_rate        : '+str(model.learning_rate)+'\n'+
      'n_estimators         : '+str(model.n_estimators)+'\n'+
      'max_depth            : '+str(model.max_depth)+'\n'+
      'min_samples_split    : '+str(model.min_samples_split)+'\n'+
      'min_samples_leaf     : '+str(model.min_samples_leaf)+'\n'+
      'max_features         : '+str(model.max_features))

print('\n')
print( np.sqrt(mean_squared_error(train_df[label_col].values, model.predict(train_df.drop(columns=to_drop).values))) )
print( r2_score(train_df[label_col].values, model.predict(train_df.drop(columns=to_drop).values)) )

print('\n')
print( np.sqrt(mean_squared_error(valid_df[label_col].values, model.predict(valid_df.drop(columns=to_drop).values))) )
print( r2_score(valid_df[label_col].values, model.predict(valid_df.drop(columns=to_drop).values)) )
print('\n')

test_df = add_datetime_features(test_df)
solution_df = pd.DataFrame(test_df["Date"])
solution_df["Consumption_MW"] = model.predict(test_df.drop(columns=["Date", "Datetime"]).values)
solution_df.to_csv("submission.csv", index=False)