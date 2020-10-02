# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_file_path = "../input/goodreadsbooks/books.csv"
data = pd.read_csv(data_file_path, skiprows=[4011, 5687, 7055, 10600, 10667])

################################ EDA ####################################
data.info()
data.shape
print("Dataset contains {} rows and {} columns".format(data.shape[0], data.shape[1]))
data.describe(include = 'all')
data.head(5)

# check for doublications
data.duplicated().any()

missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0]) # no missing value

data.columns
# numerical cols
numericCols = [cname for cname in data.columns if data[cname].dtype in ['int64','float64']]
numericCols
# categorical/text cols
categoricalCols = [cname for cname in data.columns if data[cname].dtype == 'object']
[cname + "_" + str(data[cname].nunique()) for cname in categoricalCols]
## so basically, title and isbn are text fields 
## and language_code is categorical with cardinality  30
## authors nunique is 7600..
## I think it's rather text as of now but can be converted to categoriacl with lesser cardinality

#unique number of authors
data['authors'].nunique()
##lets keep only the 1st author in the list
books = data.copy()

f = lambda x: (x['authors'].split('-'))[0]
books['authors'] = books.apply(f, axis=1)

#noww check
# number of authors
books['authors'].nunique() #7600 reduced to 4619

##removing isbn and isbn13
books = books.drop(['title','isbn','isbn13'],axis=1)
books.shape
#########################################################################\

# 1) train test split
# 2) create pipeline - preprocessing required only for categorical/str variables...no missing vales
# 3) apply and predict

## step 1: train test split
from sklearn.model_selection import train_test_split

y = books.average_rating
X = books.drop(['average_rating'], axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
[x.shape for x in (X_train, X_valid, y_train, y_valid)]

# 2) Create Pipeline
# Preprocessing for categorical data
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

my_pipeline = Pipeline(steps=[('preprocessor',  OneHotEncoder(handle_unknown = 'ignore')),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

#from sklearn.model_selection import cross_val_score
# Multiply by -1 since sklearn calculates *negative* MAE
#scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
#                              cv=5,
#                              scoring='neg_mean_absolute_error')
#print("MAE scores:\n", scores)

# 3) Apply and predict

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

pd.DataFrame({'Actual': y_valid.tolist(), 'Predicted': preds.tolist(),}).head(25)