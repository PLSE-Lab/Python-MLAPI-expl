#!/usr/bin/env python
# coding: utf-8

# Ok so here is where you say things about yourself and about what you are going to do, for now I'm going to use some visualizations and test some Machine Learning algorithms using python..
# 
# #1. Import the libraries needed

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="whitegrid", color_codes=True) #Sets the background of visualizations to white
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as pl


# #2. Load the files using pandas
# After loading the data from the csv file, explore it using ***.head()*** and ***.info()*** method

# In[ ]:


df = pd.read_csv('../input/Iris.csv')
print(df.head())
print(df.info())


# #3. Explore the data

# ##Lets visualize all the possible correlations between the variables and the "Species" column

# In[ ]:


#Sepal Lenght
df["Species"].value_counts()
x=df["Species"]
y=df["SepalLengthCm"]
sns.stripplot(x=x, y=y, data=df, jitter=True)
sns.plt.title('Species vs Sepal Length (cm)')


# In[ ]:


#Sepal Width
sns.stripplot(x=df["Species"], y=df["SepalWidthCm"], data=df, jitter=True)
sns.plt.title('Species vs Sepal Width (cm)')



# In[ ]:


#Sepal Width
sns.stripplot(x=df["Species"], y=df["SepalWidthCm"], data=df, jitter=True)
sns.plt.title('Species vs Sepal Width (cm)')



# #Notice the difference in the plot styles

# In[ ]:


#Petal Length
sns.violinplot(x=df["Species"], y=df["PetalLengthCm"], data=df, inner=None)
sns.swarmplot(x=df["Species"], y=df["PetalLengthCm"], data=df, color="w", alpha=.5)
sns.plt.title('Species vs Petal Length (cm)');


# In[ ]:


#Petal Width
sns.violinplot(x=df["Species"], y=df["PetalWidthCm"], data=df, inner=None)
sns.swarmplot(x=df["Species"], y=df["PetalWidthCm"], data=df, color="w", alpha=.5)
sns.plt.title('Species vs Petal Width (cm)');


# #4. Machine Learning Model Test
# We have a classification problem, so we should use a classification algorithm for machine learning, but which one?  
# Let's evaluate some models and see who gives the best accuracy for predictions

# Now we divide the data for to train the algorithms using the method **.train_split()**

# In[ ]:


validation_size=0.3 #How much of the data are we gonna use to validate the model, this is 30%

# Separates df into training and test dataframes
traindf, testdf = model_selection.train_test_split(df, test_size = validation_size) 

X= traindf.iloc[:,1:-1].values 
Y= traindf.iloc[:,-1].values
Xtest= testdf.iloc[:,1:-1].values
Ytest= testdf.iloc[:,-1].values


# Now that we have separated the data into train and test, now we test the Method, in this case Linear Discriminant Analysis [(Read about it here)][1]
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Linear_discriminant_analysis

# In[ ]:


logReg = LinearDiscriminantAnalysis()
logReg.fit(X,Y)
predictionsLDA = logReg.predict(Xtest);


# Once the calculations are done, it's important to know how accurate our model is, we use the method **accuracy_score()** and pass the arguments as follows:  
# **predictions:**  is the values predicted by the model (a panda series of species)  
# **Ytest:**  is the part of the data we reserved to validate the model, this is real not simulated data (a panda series of species)
# 

# In[ ]:


LDA_accuracy = accuracy_score(predictionsLDA,Ytest)
print("LDA Accuracy: ",LDA_accuracy)


# Let's try another classification algorithm, this time  [**KNeighborsClassifier** ][1]
# 
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# In[ ]:


KNC = KNeighborsClassifier()
KNC.fit(X,Y)
predictionsKNC = KNC.predict(Xtest);


# Again we evaluate the accuracy of the method using **accuracy_scope** and passing the arguments as explained

# In[ ]:


KNC_accuracy = accuracy_score(predictionsKNC,Ytest)
print("KNC Accuracy: ",KNC_accuracy)

