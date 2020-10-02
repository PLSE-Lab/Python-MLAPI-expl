#!/usr/bin/env python
# coding: utf-8

# **Hello everyone. This is a detailed notebook with Iris dataset and deep learning. I hope it will be helpful and if yes, please vote up! Thank you :-)**

# We start with loading essential libraries for data processing and visulization.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #vizualization
import matplotlib.pyplot as plt #vizualization
from matplotlib import cm

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Firstly let's load the dataset as a DateFrame, show the first 5 rows of the dataset and check the data in case of any nulls:

# In[ ]:


iris = pd.read_csv("../input/Iris.csv")
iris.head(5)
iris.info()


# There is no nulls so it is quite a clean dataset which can be processed.
# Now we are going to explore more information about columns and unique types of irises:

# In[ ]:


iris.columns


# In[ ]:


iris['Species'].unique()


# ## Exploratory analysis - visualization

# We are going to use very convinient tool to visualize data and dependencies between inputs, pairplot:

# In[ ]:


sns.pairplot(data=iris[iris.columns[1:6]], hue='Species')
plt.show()


# We could not repeat this graph, but I think it would be usefull to notice that we see a clear correlation between Petal features and type of species. 

# In[ ]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='yellow', label='Virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal length depending on Width")
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()


# Lets build a heatmap with input as the correlation matrix calculted by * iris.corr()*

# In[ ]:


plt.figure(figsize=(8,5)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r') 
plt.show()


# The Petal Width and Length are highly correlated, what we can not say about Sepal Width and Length.
# Now let's build an Andrews curve:

# In[ ]:


plt.subplots(figsize = (10,8))
from pandas.tools import plotting

cmap = cm.get_cmap('summer') 
plotting.andrews_curves(iris.drop("Id", axis=1), "Species", colormap = cmap)
plt.show()


# 

# We firstly drop out unecessary column, axis=1. Then process normalization for features vector values and check it with showing first 5 lines:

# In[ ]:


iris.drop('Id',axis=1, inplace=True) #
df_norm = iris[iris.columns[0:4]].apply(lambda x:(x - x.min())/(x.max() - x.min()))
df_norm.sample(n=5)


# Encoding Species labels for use as a target in Neural Network and concatenating feature vectors and target vector in one.

# In[ ]:


target = iris[['Species']].replace(iris['Species'].unique(), [0,1,2])
df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)


# ## Train the model

# In[ ]:


import keras

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler, LabelBinarizer


# Selecting the target as **y**, and feature vectors as **X** :

# In[ ]:


X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)


# Dividing data into train set and test set. We will do it 80/20.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)


# Creating a model, adding a layer after layer

# In[ ]:


model = Sequential()
model.add(Dense( 12, input_dim=4, activation = 'relu'))
model.add(Dense( units = 15, activation= 'relu'))
model.add(Dense( units = 8, activation= 'relu'))
model.add(Dense( units = 10, activation= 'relu'))
model.add(Dense( units = 3, activation= 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 120, validation_data = (x_test, y_test))


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train','Test'])
plt.show()


# The accuracy of our model is almost 97.5%. You can try to perform a better score by tunig the model.
# ### Thank you for reading my notebook!

# In[ ]:




