#!/usr/bin/env python
# coding: utf-8

# # Diagnosing Heart Disease

# ## Database contains 76 attributes where we have found any other trends in the heart data to predict certain cardiovascular events or find any clear indications of heart health.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ### Upload data

# In[ ]:


dt = pd.read_csv("../input/heart.csv")


# ### Show the first ten attributes.

# In[ ]:


dt.head(10)


# ### Types of data.

# In[ ]:


dt.dtypes


# ### we separate the predicts

# In[ ]:


predicts = dt.iloc[:, 0:13].values


# ### we separate the class

# In[ ]:


cla = dt.iloc[:,13].values


# ### Training

# In[ ]:


from sklearn.model_selection import train_test_split
predicts_train, predicts_test, class_train, class_test = train_test_split(predicts, cla, test_size=0.2, random_state=42)


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(predicts_train, class_train)
predicts_NB = classifier_NB.predict(predicts_test)


# ### Confusion table Naive Bayes

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
predict_NB = accuracy_score(class_test, predicts_NB)


# In[ ]:


matriz_NB = confusion_matrix(class_test, predicts_NB)


# In[ ]:


plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(matriz_NB,annot=True,cmap="Blues",fmt="d",cbar=False)


# ### Result Naive Bayes

# In[ ]:


print(predict_NB)


# # Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier_Tree = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
classifier_Tree.fit(predicts_train, class_train)
predicts_Tree = classifier_Tree.predict(predicts_test)


# ### Confusion table Tree Classifier

# In[ ]:


predict_Tree = accuracy_score(class_test, predicts_Tree)


# In[ ]:


matriz_Tree = confusion_matrix(class_test, predicts_Tree)


# In[ ]:


plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(matriz_Tree,annot=True,cmap="Greens",fmt="d",cbar=False)


# ### Result Tree Classifier

# In[ ]:


print(predict_Tree)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators=10, criterion= 'entropy', random_state=0)
classifier_RF.fit(predicts_train, class_train)
predicts_RF= classifier_RF.predict(predicts_test)


# ### Confusion table Random Forest

# In[ ]:


predict_RF = accuracy_score(class_test, predicts_RF)


# In[ ]:


matriz_RF = confusion_matrix(class_test, predicts_RF)


# In[ ]:


plt.title("Random Forest Confusion Matrix")
sns.heatmap(matriz_RF,annot=True,cmap="Reds",fmt="d",cbar=False)


# ### Result Random Forest

# In[ ]:


print(predict_RF)


# # KNN

# ### OneHotEncoder

# In[ ]:


predicts_enc = dt.iloc[:, 0:13].values


# In[ ]:


cla_enc = dt.iloc[:,13].values


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
labelencoder_predicts = LabelEncoder()
predicts_enc[:,1] =  labelencoder_predicts.fit_transform(predicts_enc[:,1])
predicts_enc[:,2] =  labelencoder_predicts.fit_transform(predicts_enc[:,2])
predicts_enc[:,3] =  labelencoder_predicts.fit_transform(predicts_enc[:,3])
predicts_enc[:,4] =  labelencoder_predicts.fit_transform(predicts_enc[:,4])
predicts_enc[:,5] =  labelencoder_predicts.fit_transform(predicts_enc[:,5])
predicts_enc[:,6] =  labelencoder_predicts.fit_transform(predicts_enc[:,6])
predicts_enc[:,7] =  labelencoder_predicts.fit_transform(predicts_enc[:,7])
predicts_enc[:,8] =  labelencoder_predicts.fit_transform(predicts_enc[:,8])
predicts_enc[:,9] =  labelencoder_predicts.fit_transform(predicts_enc[:,9])
predicts_enc[:,10] =  labelencoder_predicts.fit_transform(predicts_enc[:,10])
predicts_enc[:,11] =  labelencoder_predicts.fit_transform(predicts_enc[:,11])
predicts_enc[:,12] =  labelencoder_predicts.fit_transform(predicts_enc[:,12])


# In[ ]:


onehotencode = OneHotEncoder(categories='auto')
predicts_enc = onehotencode.fit_transform(predicts_enc).toarray()


# In[ ]:


labelencoder_cla_enc = LabelEncoder()
cla_enc = labelencoder_cla_enc.fit_transform(cla_enc)


# ### Training LabelEncoder

# In[ ]:


predicts_train_enc, predicts_test_enc, class_train_enc, class_test_enc = train_test_split(predicts_enc, cla_enc, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_KNN.fit(predicts_train_enc, class_train_enc)
predicts_KNN = classifier_KNN.predict(predicts_test_enc)


# ### Confusion table KNN

# In[ ]:


predict_KNN = accuracy_score(class_test_enc, predicts_KNN)


# In[ ]:


matriz_KNN = confusion_matrix(class_test_enc, predicts_KNN)


# In[ ]:


plt.title("KNN Confusion Matrix")
sns.heatmap(matriz_KNN,annot=True,cmap="Purples",fmt="d",cbar=False)


# ### Result KNN

# In[ ]:


print(predict_KNN)


# # Regression (best result using scaling)

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier_Regression = LogisticRegression(solver = 'lbfgs')
classifier_Regression.fit(predicts_train_enc, class_train_enc)
predicts_Regression = classifier_Regression.predict(predicts_test_enc)


# ### Confusion table Regression

# In[ ]:


predict_Regression = accuracy_score(class_test_enc, predicts_Regression)


# In[ ]:


matriz_Regression = confusion_matrix(class_test_enc, predicts_Regression)


# In[ ]:


plt.title("Regression Confusion Matrix")
sns.heatmap(matriz_Regression,annot=True,cmap="Oranges",fmt="d",cbar=False)


# ### Result Regression

# In[ ]:


print(predict_Regression)


# # SVM (best result using scaling)

# In[ ]:


from sklearn.svm import SVC
classifier_SVM = SVC(kernel='linear', random_state=1)
classifier_SVM.fit(predicts_train_enc, class_train_enc)
predicts_SVM = classifier_Regression.predict(predicts_test_enc)


# ### Confusion table SVM

# In[ ]:


predict_SVM = accuracy_score(class_test_enc, predicts_SVM)


# In[ ]:


matriz_SVM = confusion_matrix(class_test_enc, predicts_SVM)


# In[ ]:


plt.title("SVM Confusion Matrix")
sns.heatmap(matriz_SVM,annot=True,cmap="Greys",fmt="d",cbar=False)


# ### Result SVM

# In[ ]:


print(predict_SVM)


# ## Neural Networks 

# In[ ]:


from sklearn.neural_network import MLPClassifier
classifier_Neural = MLPClassifier(verbose=True, max_iter=2000, tol=0.00002)
classifier_Neural.fit(predicts_train_enc, class_train_enc)
predicts_Neural = classifier_Neural.predict(predicts_test_enc)


# ### Confusion table Neural Networks

# In[ ]:


predict_Neural = accuracy_score(class_test_enc, predicts_Neural)


# In[ ]:


matriz_Neural = confusion_matrix(class_test_enc, predicts_Neural)


# In[ ]:


plt.title("Neural Networks Confusion Matrix")
sns.heatmap(matriz_Neural,annot=True,cmap="Blues",fmt="d",cbar=False)


# ### Result Neural Networks

# In[ ]:


print(predict_Neural)


# # Neural Networks - Keras

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


classifier_Neural_Keras = Sequential()
classifier_Neural_Keras.add(Dense(units = 5, activation = 'relu', input_dim = 398))


# In[ ]:


classifier_Neural_Keras.add(Dense(units = 5, activation = 'relu'))
classifier_Neural_Keras.add(Dense(units = 1, activation = 'sigmoid'))


# In[ ]:


classifier_Neural_Keras.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


# In[ ]:


classifier_Neural_Keras.fit(predicts_train_enc, class_train_enc, batch_size = 8, epochs = 40)


# In[ ]:


predicts_Neural_Keras = classifier_Neural_Keras.predict(predicts_test_enc)


# ### Confusion table Neural Networks - Keras

# In[ ]:


predict_Neural_Keras = accuracy_score(class_test_enc, predicts_Neural_Keras.round())


# In[ ]:


matriz_Neural_Keras = confusion_matrix(class_test_enc, predicts_Neural_Keras.round())


# In[ ]:


plt.title("Neural Networks_Keras Confusion Matrix")
sns.heatmap(matriz_Neural_Keras,annot=True,cmap="Greens",fmt="d",cbar=False)


# ### Result Neural Networks Keras

# In[ ]:


print(predict_Neural_Keras)


# ## Graph of Algorithms

# In[ ]:


plt.rcParams['figure.figsize'] = (16,5)
names_alg = ['Naive Bayes', 'Tree Classifier', 'Random Forest', 'KNN', 'Regression', 'SVM', 'Neural Networks', 'Keras']
result_alg = [predict_NB, predict_Tree, predict_RF, predict_KNN, predict_Regression, predict_SVM, predict_Neural, predict_Neural_Keras]
xs = [i + 0.5 for i, _ in enumerate(names_alg)]
plt.bar(xs,result_alg, color=('#8B0000','#FF6347','#CD6600','#8B8B00','#458B00','#53868B','#EE7942','#00FF33'))
plt.ylabel("Value")
plt.title("Algorithms")
plt.xticks([i + 0.5 for i, _ in enumerate(names_alg)], names_alg)
plt.show()

