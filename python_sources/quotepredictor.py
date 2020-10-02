### ~0.7 accuracy with full dataset, now testing subsets of columns.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def changedate(X):
    newdate = datetime.strptime(X, '%Y-%m-%d')
    utc = (newdate - datetime(1970,1,1)).total_seconds()
    return utc

df_train_raw = pd.read_csv("../input/train.csv")
df_test_raw = pd.read_csv("../input/test.csv")

basecols = [col for col in df_train_raw.columns if col[-1] not in ['A', 'B']]
base_a = basecols+[col for col in df_train_raw.columns if col[-1] == 'A']
base_b = basecols+[col for col in df_train_raw.columns if col[-1] == 'B']
#%%

#convert dates to Unix-style timestams
df_train_raw['Original_Quote_Date']=[changedate(date) for date in df_train_raw['Original_Quote_Date'].values.tolist()]
df_test_raw['Original_Quote_Date']=[changedate(date) for date in df_test_raw['Original_Quote_Date'].values.tolist()]

#replace NaN entries in the frame
df_train_raw.fillna(0, inplace = True)
df_test_raw.fillna(0, inplace = True)

#convert categorical columns to numeric encodings
from sklearn import preprocessing

for f in df_train_raw.columns:
    if df_train_raw[f].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(np.unique(list(df_train_raw[f].values) + list(df_test_raw[f].values))) # be sure to include BOTH frames' data....
        df_train_raw[f] = le.transform(list(df_train_raw[f].values))        # encode each frame separately
        df_test_raw[f] = le.transform(list(df_test_raw[f].values))          # naturally
#%%

df_train = df_train_raw
#df_train = df_train_raw[basecols]
#df_train = df_train_raw[base_a]
#df_train = df_train_raw[base_b]

df_test = df_test_raw
#df_test = df_test_raw[basecols]
#df_test = df_test_raw[base_a]
#df_test = df_test_raw[base_b]

#drop useless column
df_train = df_train.drop(['QuoteNumber'], axis=1)

#build test/train datasets
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier

clf1 = ExtraTreesClassifier()
clf2 = RandomForestClassifier()
clf3 = DecisionTreeClassifier()
clf4 = SGDClassifier()
clf5 = PassiveAggressiveClassifier()
clf6 = RidgeClassifier()
clf7 = GradientBoostingClassifier()
clf8 = AdaBoostClassifier()

classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]

X_train = df_train.drop("QuoteConversion_Flag",axis=1).values
Y_train = df_train["QuoteConversion_Flag"].values
X_test  = df_test.drop("QuoteNumber",axis=1).copy().values

"""selector = VarianceThreshold()
X_new = selector.fit_transform(X_train)
print('Old: {0}||New: {1}'.format(X_train.shape, X_new.shape))"""

#for i in range(len(classifiers)):
#    clf = Pipeline([('feature_selection', SelectFromModel(DecisionTreeClassifier())),
#        ('classification', classifiers[i])])
i = 1
while i < 2:
    clf = classifiers[i]
    scores = cross_val_score(clf, X_train, Y_train, cv=KFold(Y_train.size, 10))
    clf.fit(X_train, Y_train)
    score = clf.score(X_train, Y_train)
    print ('Classifier #{0}\tbase score:{1}\n'.format(i, score))
    print ('CV score:{0}\taccuracy +/-{1}'.format(scores.mean(), scores.std()*2))
    print ('*****')
    
    pred = clf.predict(X_test)
    df_test['QuoteConversion_Flag'] = pred
    
    out = df_test[['QuoteNumber','QuoteConversion_Flag']]
    out.to_csv('submission_2_'+str(i)+'.csv', index=False)
    i += 1