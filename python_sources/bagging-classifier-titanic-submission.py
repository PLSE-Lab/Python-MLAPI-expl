# Bagging practice from DataCamp course


# import packages 
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import scale
import statistics

# import data
train = pd.read_csv('../input/train.csv')
test_X = pd.read_csv('../input/test.csv')
#------------------------------------------------------------------------------

data = pd.concat([test_X, train]).reset_index(drop=True)


# first, we create a new column to hold the title of each person
data['Title'] = ''

  
#data.to_csv('data.csv')
#files.download('data.csv')

# lists to hold titles
all = []
unique = []

# populate all and unique with titles:
for name in data.Name:
  #print(name)
  parts = name.split(', ')
  #print(parts[1])
  title = parts[1].split()
  title = title[0]
  all.append(title)

for title in all:
  if title not in unique:
    unique.append(title)
    
# fill in titles to Title column:
for index, title in enumerate(all):
  #title_dict[title].append(0)
  data.Title[index] = title


# populate dictionary to hold title average values
title_dict = {}

for item in unique:
  title_dict[item] = []
  
for age, title in zip(data.Age, data.Title):
    #print(age, title)
    if age > 0:
      #rint(age, title)
      title_dict[title].append(age)


# create another dictionary to hold average ages per title
avg_age_dict = {}
for item in unique:
  if len(title_dict[item]) > 0:
    avg_age_dict[item] = statistics.mean(title_dict[item])
 

# imputing mean age values for those who have nan as their age
index = 0
for age, title in zip(data.Age, data.Title):
  if not(age>0):
    data.Age[index] = avg_age_dict[title]    
  index += 1

test_X = data[0:418]
train = data[418:]
#---------------------------------------------------------------------------------------


# create X and y variables to represent predictor and target data
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


# create the classifier
params = {'n_estimators': [40, 42], 'base_estimator__max_leaf_nodes':[10, 15], 'base_estimator__max_depth':[4, 5, 6]}
dt = DecisionTreeClassifier()
bc = BaggingClassifier(base_estimator=dt, oob_score=True, random_state=1) #n_estimators=70, random_state=1)

# Grid Search to determine best parameters
bc_grid = GridSearchCV(estimator=bc, param_grid=params, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
bc_grid.fit(imputed_X_train, train_y)
best_params = bc_grid.best_params_
print(best_params)


#------------------------------------------------------------------------------------
# create the final classifier for submission
        # max_leaf_nodes=10, max_depth=5, n_estimators=40
        # ^^ that's the best set of hyperparameters I've gotten so far
final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=5)                   
final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True)

final_bc.fit(imputed_X_train, train_y)
final_preds = final_bc.predict(imputed_X_test)

# compare OOB accuracy to test-set accuracy
acc_oob = final_bc.oob_score_
print(acc_oob)                              # 0.8159371492704826 is the best score I've gotten so far

# creating the submission file
len(final_preds) == len(test_X.PassengerId)   # make sure predictions and ID's match up
my_submission = pd.DataFrame({'PassengerID': test_X.PassengerId, 'Survived':final_preds})
my_submission = my_submission.astype(int)
my_submission.to_csv('submission.csv', index=False)





#----------------------------------------------------------------------------------
# evaluating cv error and training error to see if the model is underfitting or overfitting
cv_scores = - cross_val_score(final_bc, imputed_X_train, train_y, cv=10, scoring = 'neg_mean_squared_error', n_jobs=-1)
cv_rmse = cv_scores.mean() ** 0.5


y_pred_train = final_bc.predict(imputed_X_train)
train_rmse = MSE(train_y, y_pred_train) ** 0.5

print('CV RMSE: {:.2f}'.format(cv_rmse))
print('Train RMSE: {:.2f}'.format(train_rmse))





