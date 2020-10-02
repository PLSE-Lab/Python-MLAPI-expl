import pandas as pd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

target=train["label"]
train=train.drop("label",axis=1)


pca = PCA(n_components=50, whiten=True)
pca.fit(train)
train = pca.transform(train)
test = pca.transform(test)
from sklearn.svm import SVC
#test_df=test.copy()
rf = SVC()
rf.fit(train,target)
print(rf.score(train,target))
Y_Pred = rf.predict(test)

submission = pd.DataFrame(Y_Pred, columns=['Label'], 
                       index=np.arange(1, 28001))
submission.to_csv("rfyoyo.csv")
