
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../input/Indian Liver Patient Dataset (ILPD).csv')
dataset = dataset.fillna(method='ffill')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 10].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=8)

"""

PCA Seems to give an Intel MKL Fatal error on kaggle, leaving it out. 
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
# Choosing only 4 components since the other components contribute little to the variance of the data. 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

"""

#========== Neural Netork =====================================
""" Accuracy with a neural network = 70.39%. 
    Using 2 layers with a sigmoid activation function. (This choice makes little difference for such a dataset)
    Accuracy is used as the metric to adjust the weights accordingly over each iteration. 
  """

#Importing Keras 
import keras
from keras.models import Sequential 
from keras.layers import Dense

#Initializing ANN
classifierANN = Sequential()

#Adding an input layer
classifierANN.add(Dense(activation="sigmoid", kernel_initializer="uniform", input_dim=10, units=6))
classifierANN.add(Dense(activation="sigmoid", kernel_initializer="uniform", units=1))
#Compiling the ANN
classifierANN.compile( optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), loss="binary_crossentropy", metrics=['accuracy'])

#Fitting ANN to the training set
ANN = classifierANN.fit(X_train, y_train, batch_size= 32, nb_epoch=20)



#=========== SVM ========================
""" Accuracy of SVM is 76%. 
    Using an rbf kernel to scale the points to a higher dimension
    so that they are seperable by a hyperplane. 
    The confusion matrix shows that this SVM has a high true negative prediction rate
    But does not do a very good job of predicting true positives. 
 """


# Fitting SVM to the Training set
from sklearn import svm
SVMclassifier = svm.SVC(kernel="rbf")
SVMres = SVMclassifier.fit(X_train, y_train)

# Predicting the Test set results
SVMaccuracy = SVMres.score(X_test,y_test)
y_pred = SVMclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
SVMcm = confusion_matrix(y_test, y_pred)

#================ KNN ===============================
""" Accuracy of KNN is 67.5%. 
    Using 5 neighbors with uniform weights assigned to the points. 
    Manhatten distance is used, signified by p=1.
    Using manhatten seems to give a higher accuracy. 
    Eucledian distance will suffice for most cases but manhatten distance in general
    works better for high dimensional data. 
"""

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
KNNclassifier = KNeighborsClassifier(n_neighbors = 5, p = 1)
KNNclassifier.fit(X_train, y_train)

# Predicting the Test set results
KNNaccuracy = KNNclassifier.fit(X_train, y_train).score(X_test,y_test)
y_pred = KNNclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
KNNcm = confusion_matrix(y_test, y_pred)   


#=============== NAIVE BAYES ========================
""" Accuracy of Naive Bayes is 50.4%.
    Naive Bayes seems to perform badly potentially due to the 
    correlated attributes present and the data in the current dimension
    cannot be seperated by a linear/parabolic/elliptic boundary. 
  """


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
Bayesclassifier = GaussianNB()
Bayes_k = Bayesclassifier.fit(X_train, y_train)

# Predicting the Test set results
BayesAccuracy = Bayes_k.score(X_test,y_test)
y_pred = Bayesclassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
Bayes_cm = confusion_matrix(y_test, y_pred)












