# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython import get_ipython
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

global train, test
train = pd.read_csv('../input/train.csv') # get train data
test = pd.read_csv('../input/test.csv') # get test data

# def getCombinedData(train, test):
#     dropped = train["Survived"]
#     train.drop("Survived", 1, inplace=True)
#     # merge 2 sets
#     combined = train.append(test)
#     combined.reset_index(inplace=True)
#     combined.drop('index', inplace=True, axis=1)
#     return combined

# combined = getCombinedData(train, test)

# groupedTrain = train.groupby(['Sex', 'Pclass', 'Parch'])
# groupedTrainMedian = groupedTrain.median()
# groupedTest = combined[:891].groupby(['Sex', 'Pclass', 'Parch'])
# groupedTestMedian = groupedTest.median()

# def fillAges(row, dataset):
#     # print("dataset", dataset)
#     if row['Sex'] == 'female' and row['Pclass'] == 1:
#         if row['Parch'] == 0:
#             return dataset.loc['female', 1, 0]['Age']
#         elif row['Parch'] == 1:
#             return dataset.loc['female', 1, 1]['Age']
#         elif row['Parch'] == 2:
#             return dataset.loc['female', 1, 2]['Age']
#         elif row['Parch'] == 3:
#             return dataset.loc['female', 1, 3]['Age']
#         elif row['Parch'] == 4:
#             return dataset.loc['female', 1, 4]['Age']
#         elif row['Parch'] == 5:
#             return dataset.loc['female', 1, 5]['Age']
#         elif row['Parch'] == 6:
#             return dataset.loc['female', 1, 6]['Age']
#     elif row['Sex'] == 'female' and row['Pclass'] == 2:
#         if row['Parch'] == 0:
#             return dataset.loc['female', 1, 0]['Age']
#         elif row['Parch'] == 1:
#             return dataset.loc['female', 1, 1]['Age']
#         elif row['Parch'] == 2:
#             return dataset.loc['female', 1, 2]['Age']
#         elif row['Parch'] == 3:
#             return dataset.loc['female', 1, 3]['Age']
#         elif row['Parch'] == 4:
#             return dataset['female', 1, 4]['Age']
#         elif row['Parch'] == 5:
#             return dataset.loc['female', 1, 5]['Age']
#         elif row['Parch'] == 6:
#             return dataset.loc['female', 1, 6]['Age']
#     elif row['Sex'] == 'female' and row['Pclass'] == 3:
#         if row['Parch'] == 0:
#             return dataset.loc['female', 1, 0]['Age']
#         elif row['Parch'] == 1:
#             return dataset.loc['female', 1, 1]['Age']
#         elif row['Parch'] == 2:
#             return dataset.loc['female', 1, 2]['Age']
#         elif row['Parch'] == 3:
#             return dataset.loc['female', 1, 3]['Age']
#         elif row['Parch'] == 4:
#             return dataset.loc['female', 1, 4]['Age']
#         elif row['Parch'] == 5:
#             return dataset.loc['female', 1, 5]['Age']
#         elif row['Parch'] == 6:
#             return dataset.loc['female', 1, 6]['Age']
#     elif row['Sex'] == 'male' and row['Pclass'] == 1:
#         if row['Parch'] == 0:
#             return dataset.loc['female', 1, 0]['Age']
#         elif row['Parch'] == 1:
#             return dataset.loc['female', 1, 1]['Age']
#         elif row['Parch'] == 2:
#             return dataset.loc['female', 1, 2]['Age']
#         elif row['Parch'] == 3:
#             return dataset.loc['female', 1, 3]['Age']
#         elif row['Parch'] == 4:
#             return dataset.loc['female', 1, 4]['Age']
#         elif row['Parch'] == 5:
#             return dataset.loc['female', 1, 5]['Age']
#         elif row['Parch'] == 6:
#             return dataset.loc['female', 1, 6]['Age']
#     elif row['Sex'] == 'male' and row['Pclass'] == 2:
#         if row['Parch'] == 0:
#             return dataset.loc['female', 1, 0]['Age']
#         elif row['Parch'] == 1:
#             return dataset.loc['female', 1, 1]['Age']
#         elif row['Parch'] == 2:
#             return dataset.loc['female', 1, 2]['Age']
#         elif row['Parch'] == 3:
#             return dataset.loc['female', 1, 3]['Age']
#         elif row['Parch'] == 4:
#             return dataset.loc['female', 1, 4]['Age']
#         elif row['Parch'] == 5:
#             return dataset.loc['female', 1, 5]['Age']
#         elif row['Parch'] == 6:
#             return dataset.loc['female', 1, 6]['Age']
#     elif row['Sex'] == 'male' and row['Pclass'] == 3:
#         if row['Parch'] == 0:
#             return dataset.loc['female', 1, 0]['Age']
#         elif row['Parch'] == 1:
#             return dataset.loc['female', 1, 1]['Age']
#         elif row['Parch'] == 2:
#             return dataset.loc['female', 1, 2]['Age']
#         elif row['Parch'] == 3:
#             return dataset.loc['female', 1, 3]['Age']
#         elif row['Parch'] == 4:
#             return dataset.loc['female', 1, 4]['Age']
#         elif row['Parch'] == 5:
#             return dataset.loc['female', 1, 5]['Age']
#         elif row['Parch'] == 6:
#             return dataset.loc['female', 1, 6]['Age']

# def processAge():
#     train['Age'] = train.apply(lambda r : fillAges(r, groupedTrainMedian) 
#         if np.isnan(r['Age'])
#         else r['Age'], axis=1)
#     print("train", train['Age'])
#     # print("test", test['Age'])
#     # test['Age'] = test.apply(lambda r : fillAges(r, groupedTestMedian) 
#     #     if np.isnan(r['Age']) 
#     #     else r['Age'], axis=1)
# def processFares():
#     # there's one missing fare value - replacing it with the mean.
#     train.head(891).["Fare"].fillna(train["Fare"].mean(), inplace=True)
# def process_embarked():
#     # two missing embarked values - filling them with the most frequent one (S)
#     train["Embarked"].fillna('S', inplace=True)                                           
# processAge()
# #processFares()

train = train.drop(['Ticket','Cabin'], axis=1)
train = train.dropna() # Remove NaN values

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' 
# create a results dictionary to hold our regression results for easy analysis later        
results = {}
# create a regression friendly dataframe using patsy's dmatrices function
y,x = dmatrices(formula, data=train, return_type='dataframe')

# instantiate our model
model = sm.Logit(y,x)

# fit our model to the training data
res = model.fit()

# save the result for outputing predictions later
results['Logit'] = [res, formula]
res.summary()

print(train)
            
#print(groupedTrainMedian)