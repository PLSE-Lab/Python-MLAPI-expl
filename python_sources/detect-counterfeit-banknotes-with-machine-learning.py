#!/usr/bin/env python
# coding: utf-8

# # Detect counterfeit Banknotes with machine learning

# <strong>Original Data Source:</strong>
# Flury, B. and Riedwyl, H. (1988). Multivariate Statistics: A practical approach. London: Chapman & Hall, Tables 1.1 and 1.2, pp. 5-8.

# <img src="https://i.imgur.com/vzYAmpP.png" alt="search Conterfeit">

# ### ==> Goal: Detect counterfeit banknotes based on their size. 

# # Table of contents
# 
# 
# [<h3>1. Content of the dataset</h3>](#1)
# [<h3>2. Predictions</h3>](#2)
# .... [2.1. Logistic Regression](#21)<br>
# .... [2.2. Random Forrest](#22)<br>
# .... [2.3. Decision Tree](#23)<br>
# .... [2.4. Neural Network](#24)<br>
# .... [2.5. SVC](#25)<br><br>
# [<h3>3. Clustering</h3>](#3)
# .... [3.1. KMeans with SVD](#31)<br>
# .... [3.2. KMeans with PCA](#32)<br><br>
# [<h3>4. Comparison of the models</h3>](#4)

# # 1. Content of the dataset<a class="anchor" id="1"></a>
# 

# The dataset includes information about the shape of the bill, as well as the label. It is made up of 200 banknotes in total, 100 for genuine/counterfeit each.<br/>
# 
# <strong>Attributes:</strong>
# - conterfeit: Whether a banknote is counterfeit (1) or genuine (0)
# - Length: Length of bill (mm)
# - Left: Width of left edge (mm)
# - Right: Width of right edge (mm)
# - Bottom: Bottom margin width (mm)
# - Top: Top margin width (mm)
# - Diagonal: Length of diagonal (mm)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/swiss-banknote-conterfeit-detection/banknotes.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


sns.heatmap(df.isnull())
plt.title("Missing values?", fontsize = 18)
plt.show()


# There is no missing value in the dataset.

# In[ ]:


# Pairwise relationships depending on counterfeit
sns.pairplot(df, hue = "conterfeit")
plt.show()


# In[ ]:


sns.heatmap(df.corr(), annot = True, cmap="RdBu")
plt.title("Pairwise correlation of the columns", fontsize = 18)
plt.show()


# # 2. Predictions<a class="anchor" id="2"></a>

# In the part, we will first separate the dataset in a training-set and a test-set. With the train-set, we will train the model and later we will the accuracy of the predictions of different models on the test-set.

# In[ ]:


# Shuffle the dataset
df = df.reindex(np.random.permutation(df.index))

X = df.drop(columns = "conterfeit")
y = df["conterfeit"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_train = st.fit_transform(X_train)


# ## 2.1. Logistic Regression<a class="anchor" id="21"></a>

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

pred = model.predict(st.transform(X_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class_report = classification_report(y_test, pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n", class_report)
print("Confusion Matrix\n",conf_matrix)
print("\nAccuracy\n",acc)

results = []
results.append(("LogisticRegression",class_report, conf_matrix, acc))


# ## 2.2. Random Forrest<a class="anchor" id="22"></a>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

pred = rfc.predict(st.transform(X_test))

class_report = classification_report(y_test, pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n", class_report)
print("Confusion Matrix\n",conf_matrix)
print("\nAccuracy\n",acc)

results.append(("RandomForestClassifier",class_report, conf_matrix, acc))


# ## 2.3. Decision Tree<a class="anchor" id="23"></a>

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

pred = dtc.predict(st.transform(X_test))

class_report = classification_report(y_test, pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n", class_report)
print("Confusion Matrix\n",conf_matrix)
print("\nAccuracy\n",acc)

results.append(("DecisionTreeClassifier",class_report, conf_matrix, acc))


# ## 2.4. Neural Network<a class="anchor" id="24"></a>

# In[ ]:


import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train.values, epochs = 50, verbose = 0)


# In[ ]:


pred = model.predict(st.transform(X_test))
pred = [int(round(t)) for t in pred.reshape(1,-1)[0]]

class_report = classification_report(y_test, pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n", class_report)
print("Confusion Matrix\n",conf_matrix)
print("\nAccuracy\n",acc)

results.append(("Neural Network",class_report, conf_matrix, acc))


# # 2.5. SVC<a class="anchor" id="25"></a>

# In[ ]:


from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)

pred = svc.predict(st.transform(X_test))

class_report = classification_report(y_test, pred)
conf_matrix = confusion_matrix(y_test,pred)
acc = accuracy_score(y_test,pred)

print("Classification report:\n\n", class_report)
print("Confusion Matrix\n",conf_matrix)
print("\nAccuracy\n",acc)

results.append(("SVC",class_report, conf_matrix, acc))


# # 3. Clustering<a class="anchor" id="3"></a>

# Now we'll use the unsupervised learning algorithm KMeans to find clusters in the dataset without using the counterfeit column to see if it will be capable to separate well the dataset in two clusters.
# 

# ## 3.1. KMeans with SVD<a class="anchor" id="31"></a>

# In[ ]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 2, random_state = 0)

transf = svd.fit_transform(X)

plt.scatter(x = transf[:,0], y = transf[:,1])
plt.title("Dataset after transformation with SVD", fontsize = 18)
plt.show()


# In[ ]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters = 2)
c = km.fit_predict(transf)

plt.scatter(x = transf[:,0], y = transf[:,1], c = c)
plt.title("Clustering with Kmeans after SVD", fontsize = 18)
plt.show()


# In[ ]:


plt.scatter(x = transf[:,0], y = transf[:,1], c = y)
plt.title("Original labels after SVD", fontsize = 18)
plt.show()


# ## 3.2. KMeans with PCA<a class="anchor" id="32"></a>

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state = 0)

transf = pca.fit_transform(X)

plt.scatter(x = transf[:,0], y = transf[:,1])
plt.title("Dataset after transformation with PCA", fontsize = 18)
plt.show()


# In[ ]:


km = KMeans(n_clusters = 2)
c = km.fit_predict(transf)

plt.scatter(x = transf[:,0], y = transf[:,1], c = c)
plt.title("Clustering with Kmeans after PCA", fontsize = 18)
plt.show()


# In[ ]:


plt.scatter(x = transf[:,0], y = transf[:,1], c = y)
plt.title("Original labels after PCA", fontsize = 18)
plt.show()


# # 4. Comparison of the models<a class="anchor" id="4"></a>

# In[ ]:


labels  = []
height = []
for i in range(len(results)):
    labels.append(results[i][0])
    height.append(results[i][-1])
    
plt.figure(figsize = (12,6))    
ax = sns.barplot(labels,height)
ax.set_xticklabels(labels, fontsize = 18, rotation = 90)
plt.title("Comparison of the models", fontsize = 18)
plt.ylabel("Prediction accuracy")
plt.show()


# <strong>==> All of the models give good predictions.</strong>

# <img src="https://i.imgur.com/5VIHT6R.png" alt = "good">
