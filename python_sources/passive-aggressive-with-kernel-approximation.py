import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.kernel_approximation import Nystroem

dataset_train = pd.read_csv('../input/train.csv', dtype='float32')

X_train = dataset_train.iloc[:,1:].values
y_train = dataset_train.iloc[:,0].values

dataset_test = pd.read_csv('../input/test.csv', dtype='float32')
X_test = dataset_test.values

pipeline = make_pipeline(
    Nystroem(kernel='rbf', gamma=4e-7, n_components=1250),
    OneVsOneClassifier(PassiveAggressiveClassifier(max_iter=25, fit_intercept=False)),
)
print(pipeline)

pipeline.fit(X_train, y_train)

pd.Series(
    name="Label",
    data=pipeline.predict(X_test).astype('int32'),
    index=pd.Index(np.arange(1, len(X_test)+1), name="ImageId"),
).to_frame().to_csv('predictions.csv')