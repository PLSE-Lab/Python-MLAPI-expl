# ##Imprting Libraries
import pandas as pd
import numpy as np
import scipy 

# ##Importing Image Data
digitData = pd.read_csv('../input/train.csv')
# ##Exploring Data
digitData.head()

# ##Spliting Data into train and test 
msk = np.random.rand(len(digitData)) < 0.8
train_data = digitData[msk]
test_data = digitData[~msk]
print(len(train_data))
print(len(test_data))

# ##spliting target and features
y_train = np.array(train_data['label'])
x_train = np.array(train_data.iloc[:,1:])
print(y_train.shape)
print(x_train.shape)

# ##Applying PCA on the training and testing data
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(whiten = True)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
print(x_train_pca.shape)


# ##Training Model
from sklearn.svm import LinearSVC
digit_model = LinearSVC()
digit_model.fit(x_train_pca,y_train)

y_test= np.array(test_data['label'])

x_test = np.array(test_data.iloc[:,1:])

x_test_pca = pca.transform(x_test)
print(digit_model.score(x_train_pca,y_train))
print(digit_model.score(x_test_pca,y_test))
