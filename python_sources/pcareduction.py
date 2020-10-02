#Linear regression with L1 and L2 penalties
#Random Forest
#XGBoost tree or something similar (there's a lot of open source packages for this)

#Try a dimensionality reduction method on all, or a subset of your data (https://scikit-learn.org/stable/modules/unsupervised_reduction.html)
#Try a feature selection method.

#Try one regression method that we *have not* covered in class.

#Put these in Kaggle Kernels(can be one big one if you want) but don't make them public until after the competition is over.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def printresult(y_test, filename):
    ind = np.arange(1,len(y_test)+1,1)
    dataset = pd.DataFrame({'id':ind,'OverallScore':y_test[:,1]})
    dataset.to_csv(filename, sep=',', index=False)

def cleandata(df):
    for col in df.columns:
        if (df[col].isnull().sum())/len(df) > 0.9:
            df.drop(col,inplace=True,axis=1)
    for column in list(df.columns[df.isnull().sum() > 0]):
        df[column].fillna(df[column].mean(), inplace=True)
    df.drop({'ids'}, inplace = True, axis = 1)
    df['erkey'] = df['erkey'].str.slice_replace(0, 3, '')
    df['erkey'] = pd.to_numeric(df['erkey'])
    return df

# Use PCA to reduce the dimension
def PCA_reduc(traindata):
    pca = PCA(n_components=10)
    pca_result = pca.fit(traindata);
    print(pca.explained_variance_ratio_)
    plt.plot(range(10), pca.explained_variance_ratio_)

trainF = pd.read_csv("../input/trainFeatures.csv")
trainL = pd.read_csv("../input/trainLabels.csv")
testF = pd.read_csv("../input/testFeatures.csv")


trainData = cleandata(pd.merge(trainF, trainL, on='ids'))
X_train=trainData.iloc[:,0:47]
Y_train=trainData.iloc[:,47:49]

X_test = cleandata(testF)
PCA_reduc(X_train)
