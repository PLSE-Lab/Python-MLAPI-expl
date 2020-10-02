# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv("../input/mushrooms.csv")

#Print dataset
dataset.head(5)

#To print all header 
list(dataset)

#All attributes are factor then labeling that

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
encoder = LabelEncoder()
for i in dataset.columns:
    dataset[i] = encoder.fit_transform(dataset[i])
    
#To check unique value
dataset['class'].unique()

dataset.isnull().sum()

##Creating predictor variable and response variable

X = dataset.iloc[:,1:]
y = dataset.iloc[:,0]

#Standardising data
standardize = StandardScaler()

X = standardize.fit_transform(X)

#Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)

#Explained Variance

Explain_variance = pca.explained_variance_

#Plot PCA

with plt.style.context('dark_background'):
    plt.figure(figsize=(6,4))
    plt.bar(range(22),Explain_variance,alpha=0.5,align='center',label='Individual Explained Variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

#Taking first 17 pca

pca_modified = PCA(n_components=17)
pca_modified.fit_transform(X)

#Teain test split
X_Train,X_test,y_Train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
