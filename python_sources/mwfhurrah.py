import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import csv as csv
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

#this code doesnt work :( no error, just doesn't work
csv_file_object = csv.reader(open('../input/train.csv', 'rU'))
header = csv_file_object.next()
data=[]
for row in csv_file_object:      
    data.append(row)             
data = np.array(data) 	 
print (data[0])
#I HAVE NO IDEA WHY >:(
def has_title(name):
    for s in ['Mr.', 'Mrs.', 'Miss.', 'Dr.', 'Sir.']:
        if name.find(s) >= 0:
            return True
    return False


def munge(df):
    # Pclass
    for i in range(3):
        cls_fn = lambda x: 0.5 if x == i + 1 else -0.5
        df['C%d' % i] = df['Pclass'].map(cls_fn)

    # Sex => Gender
    gender_fn = lambda x: -0.5 if x == 'male' else 0.5
    df['Gender'] = df['Sex'].map(gender_fn)

    # Name => Title
    title_fn = lambda x: 0.5 if has_title(x) else -0.5
    title_col = df['Name'].map(title_fn)
    title_col.name = 'Title'
    dfn = pd.concat([df, title_col], axis=1)

    # Embarked
    s3fa_col = (dfn.Pclass == 3).mul(dfn.Sex == 'female').mul(dfn.Embarked == 'S').mul(dfn.Title > 0)
    s3fa_fn = lambda x: 0.5 if x else -0.5
    s3fa_col = s3fa_col.map(s3fa_fn)
    s3fa_col.name = 'S3FA'
    dfne = pd.concat([dfn, s3fa_col], axis=1)

    # Result
    cols = ['C0', 'C1', 'C2', 'Gender', 'Title', 'S3FA']
    return dfne[cols]

#DIS HAS ERROR 2
df = pd.read_csv('../input/train.csv')
mdf = munge(df)
X = mdf
y = df['Survived']
tuned_parameters = {'penalty': ['l1', 'l2'],
                    'C': np.logspace(-2, 0, 5),
                    'max_iter': np.logspace(2, 3, 5)}
clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, n_jobs=4)
clf.fit(X, y)
print(clf.best_estimator_)
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))
test_df = pd.read_csv('../input/test.csv')
res = clf.predict(munge(test_df))
res = Series(res, name='Survived', index=test_df.index)
res = pd.concat([test_df, res], axis=1)[['PassengerId', 'Survived']]
res.to_csv('out-1-lr.csv', index=False)

#:(