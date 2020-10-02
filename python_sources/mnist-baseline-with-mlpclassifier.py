# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import warnings                     

# Loading Tools
from sklearn import model_selection 

# Loading Classification Models
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# Loading data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Concatenate both train & test data to apply uniform data transformations
train['TYPE-LABEL'] = ['TRAIN'] * train.shape[0]
test['TYPE-LABEL'] = ['TEST'] * test.shape[0]

data = pd.concat([train, test],ignore_index = True, sort=False)
target = data['label']
idxType = data['TYPE-LABEL']
data = data.drop(['label', 'TYPE-LABEL'], axis=1)

# Converting pixels to 1's & 0's
threshold = 85
for column in data.columns.tolist():
	data[column] = data[column].apply(lambda x: 1 if x > threshold else 0)

data['label'] = target
data['TYPE-LABEL'] = idxType

# Seperate out train & test
train = data.loc[data['TYPE-LABEL'] == 'TRAIN']
train.drop('TYPE-LABEL', axis=1, inplace=True)
train.reset_index(drop=True, inplace=True)

test = data.loc[data['TYPE-LABEL'] == 'TEST']
test.drop('TYPE-LABEL', axis=1, inplace=True)
test.reset_index(drop=True, inplace=True)

XTrain = train.copy()
XTrain.drop('label', axis=1, inplace=True)
yTrain = train['label']
XTest = test.copy()
XTest.drop('label', axis=1, inplace=True)

###################
# Picking  Neural Network Classifier
classifier = MLPClassifier(activation='logistic',learning_rate='adaptive',
				alpha=1.e-07, hidden_layer_sizes=(2000,1000,500))
classifier.fit(XTrain , yTrain)

# Predicting classes
yPredicted = classifier.predict(XTest)

# Creating submission file
testID = range(1,len(yPredicted)+1)
imagePredictions = pd.DataFrame({'ImageId':testID, 'Label':yPredicted})
imagePredictions = imagePredictions.astype('int64')
imagePredictions = imagePredictions[['ImageId', 'Label']]
imagePredictions.to_csv('sample_submission.csv', index = False)