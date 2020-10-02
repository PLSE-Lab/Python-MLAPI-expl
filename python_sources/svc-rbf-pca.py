import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
y = train_data[["label"]]
X_train = train_data.drop(["label"], axis=1)

X_train_stan = X_train
X_test_stan = test_data

###################TO FIND NUMBER OF FEATURES TO GAIN ~90% VARIANCE###################
# pca = PCA(whiten=True)
# pca.fit(X_train_stan)
# variance = 100 * pca.explained_variance_ratio_
# cumulativeVariance = np.cumsum(variance)
# xAxis = np.arange(1,785) 
# sns.regplot(x= xAxis, y=cumulativeVariance, x_bins = 50);
# sns.plt.show()
##########################################################################################

pca = PCA(n_components= 40, whiten=True)
pca.fit(X_train_stan)
X_train_stan = pca.transform(X_train_stan)
X_test_stan = pca.transform(X_test_stan)

model = SVC()
model.fit(X_train_stan, y.iloc[:,0]) 
predictions = model.predict(X_test_stan)
counter = np.arange(1,28001)
c1 = pd.DataFrame({'ImageId': counter})
c2 = pd.DataFrame({'Label':predictions})
res = pd.concat([c1, c2], axis = 1)
res.to_csv('output.csv', index = False)
