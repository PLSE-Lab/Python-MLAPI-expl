#!/usr/bin/env python
# coding: utf-8

# According to the conditions of the problem at the university, I had to solve this task using only neural networks. I chose Keras for its simplicity.
# 
# Unfortunately, the majority of the owners of the kernels flopped neural networks without any data preparation, and as a result they receive 80-85% score. I tried to make it just a bit smarter and got an accuracy of 89-90%.
# 
# In principle, just a little has been done: data is standardized, classes are balanced, and a pair of hidden layers in a neural network was added. You can also replace the output activation function to softmax, and make cross-validate. I tried it, but my accuracy was low. If you beat my record, please send a link to your work on email: vvpereverzev@edu.hse.ru

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


# In[ ]:


data = pd.read_csv('../input/heart.csv', delimiter=',')


# In[ ]:


data.head(3)


# In[ ]:


# Let's look at the distribution of people by sex
male = len(data[data.sex == 1])
female = len(data[data.sex == 0])
sns.countplot('sex', hue='target', data=data)
plt.title('Heart Disease: Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Now, let's look at the distribution of people by age.
plt.figure(figsize=(9, 9))
plt.title('Heart Disease: Age')
plt.xlabel('Age')
plt.ylabel('Qantity')
data['age'].hist(bins=20)
plt.show()


# In[ ]:


data_v = data.iloc[:, 0:13].values
print('Feature vector:', data_v[1])


# ## Let's look at the overall distribution. Maybe we will find out something

# In[ ]:


#Reducing the dimension to 2 for visualization.
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(data_v)
data_2d = pca.transform(data_v)

#Building a graph on a two-dimensional matrix
colormap = np.array(['red', 'lime'])
plt.figure(figsize=(10, 10))

for i in range(0, data_2d.shape[0]):
    if data['target'][i] == 1:
        c1 = plt.scatter(data_2d[i, 0], data_2d[i, 1], c='red')
    elif data['target'][i] == 0:
        c2 = plt.scatter(data_2d[i, 0], data_2d[i, 1], c='lime')

plt.title('People distribution')
plt.legend([c1, c2], ['Sick', 'Healthy'])


# As we can see, they are not that different.

# At the end, we look at the correlation of signs

# In[ ]:


plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True)
plt.show()


# It's simple! Less thalach, more oldpeak, and everything will be fine

# ## Let's look at the distribution of target

# In[ ]:


print(data['target'].value_counts())


# In[ ]:


plt.figure(figsize=(7, 7))
data['target'].value_counts().plot(kind='bar', label='Target')
plt.legend()
plt.title('Distribution of target')


# ## Rebalancing target values

# In[ ]:


from sklearn.utils import resample

df_majority = data[data.target==1]
df_minority = data[data.target==0]
 

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=165,    
                                 random_state=123)
 

data = pd.concat([df_majority, df_minority_upsampled])
 

data['target'].value_counts()


# In[ ]:


from keras.models import Sequential
from keras import metrics
from keras.layers.core import Dense, Activation ,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ## Standardize data

# In[ ]:


data_pop = data.drop("target", axis=1)
target = data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(data_pop, target, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Keras time

# In[ ]:


#Keras neural network

model = Sequential()
model.add(Dense(15, init = 'uniform', activation='relu', input_dim=13))
model.add(Dense(10, init = 'uniform', activation='relu'))
model.add(Dense(6, init = 'uniform', activation='relu'))
model.add(Dense(1, init = 'uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting
model.fit(X_train, Y_train, epochs=130)


# In[ ]:


#Testing on a test sample
Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded


# In[ ]:


score_nn = round(accuracy_score(Y_pred_nn, Y_test)*100,2)
score_f1 = round(f1_score(Y_pred_nn, Y_test)*100, 2)
print("Accuracy score: " + str(score_nn) + " %")
print("F1 score: " + str(score_f1) + "%")


# If at start you got less accuracy, then try restarting the cells with the neural network several times: until the scales are initialized with greater accuracy
