#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


get_ipython().run_line_magic('pip', 'install tensorflow')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns


# # Importing The Dataset

# In[ ]:


dataset = pd.read_csv("../input/framingham-heart-study-dataset/framingham.csv")


# # Analysing The Data

# In[ ]:


dataset.shape


# In[ ]:


dataset.dtypes


# In[ ]:


dataset.info


# # Visualizing the data

# In[ ]:


fig = plt.figure(figsize = (8,8))
ax = fig.gca()
dataset.hist(ax=ax)
plt.show()


# In[ ]:


fig, ax = plt.subplots()
ax.hist(dataset["TenYearCHD"],color = "yellow")
ax.set_title(' To predict heart disease')
ax.set_xlabel('TenYearCHD')
ax.set_ylabel('Frequency')


# In[ ]:


data = np.random.random([100,4])
sns.violinplot(data=data, palette=['r','g','b','m'])


# # Separating the dependent and independent variables

# In[ ]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[ ]:


np.isnan(X).sum()


# In[ ]:


np.isnan(y).sum()


# # Taking Care of Missing Values

# In[ ]:


from sklearn.impute import SimpleImputer
si = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X = si.fit_transform(X)


# In[ ]:


y.shape


# In[ ]:


np.isnan(X).sum()


# In[ ]:


np.isnan(y).sum()


# In[ ]:


dataset.isna().sum()


# # Splitting into Training and test Data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# # Normalising The data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


X_train


# In[ ]:


y_train


# In[ ]:


np.isnan(X_train).sum()


# In[ ]:


np.isnan(y_train).sum()


# # Preparing ANN Model with two layers

# In[ ]:


ann = tf.keras.models.Sequential()


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))


# In[ ]:


ann.add(tf.keras.layers.Dense(units = 1,activation='sigmoid'))


# In[ ]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


model = ann.fit(X_train,y_train,validation_data=(X_test,y_test), batch_size = 32,epochs=100)


# In[ ]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Model Accuracy Visualisation

# In[ ]:


plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


# # Model Loss Visualisation

# In[ ]:


plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# # Calculating Different Metrics

# In[ ]:


print(classification_report(y_test, y_pred))


# # Using MLP Classifier for Prediction

# In[ ]:


from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# # Visualiaing The MLP Model After Apllying the PCA method

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


def visualization_train(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title("%s Model on training data" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
def visualization_test(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title("%s Test Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()


# In[ ]:


visualization_train(model= 'MLP')


# # Saving a machine learning Model

# In[ ]:


import joblib
joblib.dump(ann, 'ann_model.pkl') 
joblib.dump(sc, 'sc_model.pkl') 


# In[ ]:


knn_from_joblib = joblib.load('mlp_model.pkl') 
sc_model = joblib.load('sc_model.pkl') 


# # Saving a tensorflow model

# In[ ]:


get_ipython().system('pip install h5py')


# In[ ]:


ann.save('ann_model.h5')


# In[ ]:


model = tf.keras.models.load_model('ann_model.h5')

