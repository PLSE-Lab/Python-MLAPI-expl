## Utility script for Music Classication Notebooks
# Competition : https://www.kaggle.com/c/music-classification


#
# Initialization - Imports and Configs
#

import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = '/kaggle/input/music-classification/kaggle/'
SONGS_FOLDER = 'training/'


#
# Loading and Vectorizing Data
#

def loadSong(songhash, songs_folder='training/'):
    dfSong = pd.read_csv(ROOT_DIR + songs_folder + songhash, header=None)
    return dfSong

def getSongMeans(serSongs, songs_folder='training/'):
    dfSongVectors = pd.DataFrame(map(
        lambda sId : loadSong(sId, songs_folder=songs_folder).mean(), serSongs
    ))
    return dfSongVectors

def loadAndVectorizeSong(sId, agg=['mean','std','min','max'], songs_folder='training/'):
    try:
        dfSong = loadSong(sId, songs_folder=songs_folder)
    except:
        print('File Not Found', sId)
        return None
    songVect = dfSong.agg(agg).values.flatten()
    return songVect

def getSongVectors(serSongs, agg=['mean','std','min','max']):
    dfSongVectors = pd.DataFrame(map(lambda sId : loadAndVectorizeSong(sId, agg=agg), serSongs))

    ## normalize vectors (subtract mean, divide by average)
    dfSongVectorsNormed = (dfSongVectors-dfSongVectors.mean())/dfSongVectors.std()
    return dfSongVectorsNormed




#
# Implementing Generic Gaussian Classifier
#

class GaussianSongClassifier():
    def __init__(self):
        self.means = {}
        self.covs = {}
        self.invCovs = {}
        
    def fit(self, X, Y):
        dfCombined = pd.concat([pd.DataFrame(X), pd.Series(Y)], axis=1)
        dfCombined.columns = list(range(len(dfCombined.columns)))
        
        for cat, dfGrp in dfCombined.groupby(dfCombined.columns[-1]):
            self.means[cat] = dfGrp.iloc[:,:-1].mean()
            self.covs[cat] = dfGrp.iloc[:,:-1].cov()
            self.invCovs[cat] = np.linalg.inv(self.covs[cat])
        
    def predict_genre(self, genre, row):
        a = row - self.means[genre]
        return a.dot(self.invCovs[genre]).dot(a.T)
    
    def predict_one(self, row):
        classes = list(self.means.keys())
        preds = [self.predict_genre(cl, row) for cl in classes]
        return classes[np.argmin(preds)]
    
    def predict(self, rows):
        results = []
        for ind, r in rows.iterrows():
            pred = self.predict_one(r)
            results.append(pred)
        return results

    
#
# Implementing Generic KNN Classifier
#

class KnnClassifier():
    def __init__(self, n=3):
        self.n=n
    
    def fit(self, X, Y):
        self.dfX = pd.DataFrame(X)
        self.dfY = pd.DataFrame(Y)
        
    def predict_one(self, row):
        distances = np.sum(np.square(self.dfX.values - row.values), axis=1)
        labels = self.dfY.iloc[np.argsort(distances)[:self.n]]
        return labels.iloc[:,0].value_counts().index[0]
        
    def predict(self, rows):
        results = []
        for ind, r in rows.iterrows():
            pred = self.predict_one(r)
            results.append(pred)
        return results
        
        
#
# Implementing CrossValidation testing loop & helpers
#

def testClassifier(df, model, verbose=False, dropCols=['id', 'category'], targetCol='category'):
    dfX = df.drop(dropCols, axis=1)
    dfY = df[targetCol]

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)

    testResults = []
    for train_index, test_index in sss.split(dfX, dfY):
        X_train, X_test = dfX.iloc[train_index], dfX.iloc[test_index]
        y_train, y_test = dfY.iloc[train_index], dfY.iloc[test_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        clfPerf = (accuracy_score(y_test, preds), f1_score(y_test, preds, average='weighted'))

        resultsObj = {
            'accuracy_score' : accuracy_score(y_test, preds),
            'f1_score' : f1_score(y_test, preds, average='weighted'),
            'y_test':y_test,
            'preds':preds
        }
        testResults.append(resultsObj)
        if verbose : print('accuracy_score, f1_score : ', round(resultsObj['accuracy_score'], 3), round(resultsObj['f1_score'], 3))
        
    return testResults

def resultsToAvgScore(testResults):
    avg_f1_score = np.mean([r['f1_score'] for r in testResults])
    avg_accuracy_score = np.mean([r['accuracy_score'] for r in testResults])
    return {
            'accuracy_score' : avg_accuracy_score,
            'f1_score' : avg_f1_score
    }

# Input : testResults from testClassifier
# Output : Displays aggregated confusion matrix across all test runs
def resultsToConfusionMat(testResults):
    y_test = np.concatenate([g['y_test'] for g in testResults])
    preds = np.concatenate([g['preds'] for g in testResults])
    
    labels = np.unique(y_test)
    arrConfMat = confusion_matrix(y_test, preds, labels=labels)
    dfConfMat = pd.DataFrame(arrConfMat, index=labels, columns=labels)
    display(dfConfMat.style.background_gradient())

    
    
    


