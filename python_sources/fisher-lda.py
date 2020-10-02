from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
data = pd.read_csv("../input/train.csv")
target = data[[0]].values.ravel()
train = data.iloc[:, 1:].values
test = pd.read_csv("../input/test.csv").values

lda_model = LinearDiscriminantAnalysis(solver="eigen", n_components=9, shrinkage="auto")
lda_model.fit(train, target)

prediction = lda_model.predict(test)
np.savetxt('submission_lda.csv', np.c_[range(1, len(test) + 1), prediction], delimiter=',', comments = '', header = 'ImageId,Label', fmt='%d')