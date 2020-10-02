# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Graph Plots


# Importing the Datset 
train_dataset = pd.read_csv('../input/data_set_ALL_AML_train.csv')
test_dataset  = pd.read_csv('../input/data_set_ALL_AML_independent.csv')


# Data Cleaning 
# Remove Call Coulmn from Train Dataset 
train_dataset1 = [col for col in train_dataset.columns if "call" not in col]
train_dataset  = train_dataset[train_dataset1]
train_dataset.head()

# Remove Call Coulmn from Test Dataset 
test_dataset1 = [col for col in test_dataset.columns if "call" not in col]
test_dataset  = test_dataset[test_dataset1]
test_dataset.head()

# Dataset Transpose 
# Transpose Train dataset 
train_dataset.T.head()
train_dataset = train_dataset.T

# Transpose Test dataset 
test_dataset.T.head()
test_dataset = test_dataset.T

# Drop Rows from Train dataset  Gene Description, Gene Accession Number
train_dataset2 = train_dataset.drop(['Gene Description','Gene Accession Number'],axis=0)

# Drop Rows from Twat dataset  Gene Description, Gene Accession Number
test_dataset2  = test_dataset.drop(['Gene Description','Gene Accession Number'],axis=0)

# Train dataset Convert to Numeric
train_dataset2.index = pd.to_numeric(train_dataset2.index)
train_dataset2.sort_index(inplace=True)
train_dataset2.head()

# Test dataset Convert to Numeric
test_dataset2.index = pd.to_numeric(test_dataset2.index)
test_dataset2.sort_index(inplace=True)
test_dataset2.head()


# Import Response Variable
y = pd.read_csv('../input/actual.csv')
y['cancer'].value_counts()

# Replace (ALL, AML) with (0,1)
y = y.replace({'ALL':0,'AML':1})
labels = ['ALL', 'AML'] 

# Train dataset
X_train = train_dataset2.reset_index(drop=True)
Y_train = y[y.patient <= 38].reset_index(drop=True)

# Test dataset
X_test = test_dataset2.reset_index(drop=True)
Y_test = y[y.patient > 38].reset_index(drop=True)

Y_test = Y_test.iloc[:,1].values
Y_train =Y_train.iloc[:,1].values

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

# Feature Extraction With PCA
# Applying PCA 
from sklearn.decomposition  import PCA
pca = PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train)
X_train_pca

#Eigenvalues (sum of squares of the distance between the projected data points and the origin along the eigenvector)
print(pca.explained_variance_) 

#Explained variance ratio (i.e. how much of the change in the variables is explained by change in the respective principal component): eigenvalue/(n variables)
print(pca.explained_variance_ratio_) 


#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance in (%)') #for each component
plt.title('Genee Dataset Cumulative Explained Variance')
plt.show()

## Calculating Explained Variance up to 90% of the variance 
total = sum(pca.explained_variance_)
k = 0
current_variance = 0
while current_variance/total < 0.90:
      current_variance += pca.explained_variance_[k]
      k = k + 1
k

#Applying PCA for selecting N Components
from sklearn.decomposition  import PCA
pca = PCA(n_components = k )
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)  

var_exp = pca.explained_variance_ratio_.cumsum()
var_exp = var_exp*100
plt.bar(range(k), var_exp);

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)


# Applying Data Model
# Fitting Logistic Regression to the train Set 
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression(random_state=0)
classifier.fit(X_train_pca,Y_train)

# Predicting the test set Results
Y_pred = classifier.predict(X_test_pca)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

# Confusion Matirx
logit_cm = confusion_matrix(Y_test, Y_pred)  
print(logit_cm)  

# Logit Accuuracy 
logit_ac=accuracy_score(Y_test, Y_pred)
print(logit_ac)


# Applyig Random Forest 
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0,oob_score=True)  
classifier.fit(X_train_pca, Y_train)
print(classifier.oob_score_)
# Predicting the Test set results
Y_pred = classifier.predict(X_test_pca)  

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score

# Confusion Matirx
rf_cm = confusion_matrix(Y_test, Y_pred)  
print(rf_cm) 

# Logit Accuuracy 
rf_ac=accuracy_score(Y_test, Y_pred)
print(rf_ac)
