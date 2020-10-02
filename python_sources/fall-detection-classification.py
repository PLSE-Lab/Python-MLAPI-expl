
import pandas as pd
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../input/falldeteciton.csv')

X = train_df.drop('ACTIVITY', axis=1)
y = train_df['ACTIVITY']

X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float64),
    y.astype(np.float64), train_size=0.75, test_size=0.25)
    
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')


