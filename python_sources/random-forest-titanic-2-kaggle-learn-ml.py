# Kaggle ML Learn Section Lv2 Practice

# In this Kernel, I used what I learned from the Kaggle 'Learn ML' course. This kernel uses techniques through the second chapter of level 2.
# Future Kernels will use what I learn from chapters after this one.

# IN THIS KERNEL, I ADDED ONE-HOT ENCODED CATEGORICAL VARIABLES

# I'll be using the Imputer() method from the 'handling missing data' section

# import necessary packages
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score, GridSearchCV

# import data
train = pd.read_csv('../input/train.csv')
test_X = pd.read_csv('../input/test.csv')

# create X and y variables to represent predictor and target data
# I'm excluding columns with the data type 'object' in this submission (the next Kaggle Learn Section covers categorical variables)
train_y = train.Survived
train_X = train.drop(['Survived'], axis=1)

# Use One-hot encoding to use categorical variables
onehot_train_X = pd.get_dummies(train_X)
onehot_test_X = pd.get_dummies(test_X)
train_X, test_X = onehot_train_X.align(onehot_test_X, join='left', axis=1)

# create the Imputer to impute the mean for NaN values
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(test_X)


#-----------------------------------------------------------------------------------
# Grid Search CV section
params = {'max_depth': [3, 4, 5, 6, 7], 'max_leaf_nodes':[100, 150, 200], 'min_samples_split':[0.5, 2, 3, 5, 10]}
rf = RandomForestRegressor()
rf_grid = GridSearchCV(estimator=rf, param_grid=params, scoring='roc_auc', cv=5, n_jobs=-1, refit=False)
rf_grid.fit(imputed_X_train, train_y)
best_params = rf_grid.best_params_
print(best_params)

# create the model
model = RandomForestRegressor(max_depth = 6, max_leaf_nodes = 200, min_samples_split = 5, random_state=42)
model.fit(imputed_X_train, train_y)
predictions = model.predict(imputed_X_test)
print(predictions)

# change the predictions to integers 1 and 0 (you either survive or you don't - no inbetween)
final_preds = [round(x) for x in predictions]

# creating the submission file
len(final_preds) == len(test_X.PassengerId)   # make sure predictions and ID's match up
my_submission = pd.DataFrame({'PassengerID': test_X.PassengerId, 'Survived':final_preds})
my_submission = my_submission.astype(int)
my_submission.to_csv('submission.csv', index=False)






#----------------------------------------------------------------------------------
# evaluating cv error and training error to see if the model underfits or overfits
cv_scores = - cross_val_score(model, imputed_X_train, train_y, cv=10, scoring = 'neg_mean_squared_error', n_jobs=-1)
cv_rmse = cv_scores.mean() ** 0.5

model2 = RandomForestRegressor()
model2.fit(imputed_X_train, train_y)
y_pred_train = model2.predict(imputed_X_train)
train_rmse = MSE(train_y, y_pred_train) ** 0.5

print('CV RMSE: {:.2f}'.format(cv_rmse))
print('Train RMSE: {:.2f}'.format(train_rmse))











