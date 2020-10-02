import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


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

    # Sex => Genero
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


df = pd.read_csv('../input/train.csv')
mdf = munge(df)
X = mdf
y = df['Survived']

tuned_parameters = {'C': np.logspace(-2, 0, 40)}
clf = GridSearchCV(LogisticRegression(penalty='l2', solver='sag'),
                   tuned_parameters,
                   cv=10)
clf.fit(X, y)
print(clf.best_estimator_)
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))
          
          
test_df = pd.read_csv('../input/test.csv')
res = clf.predict(munge(test_df))

res = Series(res, name='Survived', index=test_df.index)
print(res)
res = pd.concat([test_df, res], axis=1)[['PassengerId', 'Survived']]
res.to_csv('out-1-lr.csv', index=False)