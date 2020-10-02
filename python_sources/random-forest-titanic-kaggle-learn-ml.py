# Kaggle ML Learn Section Lv2 Practice

# In this Kernel, I used what I learned from the Kaggle 'Learn ML' course. This kernel uses techniques through the first chapter of level 2.
# Future Kernels will use what I learn from chapters after this one.

# I'll be using the Imputer() method from the 'handling missing data' section

# import necessary packages
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# create X and y variables to represent predictor and target data
# I'm excluding columns with the data type 'object' in this submission (the next Kaggle Learn Section covers categorical variables)
train_y = train.Survived
train_X_drop = train.drop(['Survived'], axis=1)
train_X = train_X_drop.select_dtypes(exclude=['object'])

test_X = test.select_dtypes(exclude=['object'])

# create the Imputer to impute the mean for NaN values
my_imputer = SimpleImputer(strategy='mean')
imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(test_X)

# create the model
model = RandomForestRegressor()
model.fit(imputed_X_train, train_y)
predictions = model.predict(imputed_X_test)
print(predictions)

# change the predictions to integers 1 and 0 (you either survive or you don't - no inbetween)
final_preds = [round(x) for x in predictions]

# creating the submission file
len(final_preds) == len(test.PassengerId)   # make sure predictions and ID's match up
my_submission = pd.DataFrame({'PassengerID': test.PassengerId, 'Survived':final_preds})
my_submission = my_submission.astype(int)
my_submission.to_csv('submission.csv', index=False)






