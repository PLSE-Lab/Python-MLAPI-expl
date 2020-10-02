#!/usr/bin/env python
# coding: utf-8

# **About this notebook**
# 
# This is my first try with kaggle and virtually my first AI project although I have some experience with python. Beforehand I saw a few videos about machine learning, as a result I had some ideas of what I could try but not the first clue about the best strategy to solve this problem.
# 
# In this notebook I tried various solutions from the most naive to more complex. For each approach I tried to improve the performance by testing different parameters. The key idea here is trial and error. I tried the following algoritms:
# 1. k-nearest neighbors
# 2. Support vector classification
# 3. Random forest
# 4. Neural Network
# 
# In this process, I was able to learn a lot and see the accuracy of my submissions grow from 0.62 to 0.78. 
# 
# I think this notebook can be a source of inspiration if you are new to AI and have no idea where to start. It may also be valuable if you already have one solution for this competition and want to browse quickly other approachs.

# **Functions for data preprocessing**
# 
# These function will be used for all approach

# In[ ]:


#First we import a few libraries we will use extensively
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# This function open the training data, drop some parameters that seems less usefull, convert string into integers for Sex and Embarked and fill NaN values

# In[ ]:


from sklearn import model_selection, preprocessing

def open_data(file):
    #Readind training data
    data = pd.read_csv("../input/"+file)

    #Droping Name, Cabin and Ticket as probably irrelevant
    data = data.drop(["Name", "Ticket", "Cabin"], 1)
    
    #Converting string features into integers
    le = preprocessing.LabelEncoder()
    data["Sex"] = le.fit_transform(list(data["Sex"]))
    data["Embarked"] = le.fit_transform(list(data["Embarked"]))

    #Filling NaN values for age with the average value
    data["Age"] = data["Age"].fillna(value = data.Age.mean())
    data["Fare"] = data["Fare"].fillna(value = data.Fare.mean())
    
    return data


# This function separate parameters from labels.

# In[ ]:


def param_label(data):
    data = data.drop(["PassengerId"], 1)
    return data.drop(["Survived"], 1), data[["Survived"]]


# This one generate a random subset we can use to test each approach.

# In[ ]:


def subset_data(X, Y, n):
    return model_selection.train_test_split(X, Y, test_size = n)


# **Naive approachs**
# 
# I began with very naive approachs : what if everyone survive ? No one survive ? survival is random ?

# 1. Everybody survive

# In[ ]:


data = open_data("train.csv")

y_true = data[["Survived"]]
y_test = np.array([1 for i in range(len(y_true))])
print("Accuracy for survived = 1: ",metrics.accuracy_score(y_true, y_test))


# 2. Random survival

# In[ ]:


data = open_data("train.csv")

y_true = data[["Survived"]]
y_test = np.array([random.choice((0, 1)) for i in range(len(y_true))])
print("Accuracy for random survival: ",metrics.accuracy_score(y_true, y_test))


# 3. Everyone die 

# In[ ]:


data = open_data("train.csv")

y_true = data[["Survived"]]
y_test = np.array([0 for i in range(len(y_true))])
print("Accuracy for survived = 1: ",metrics.accuracy_score(y_true, y_test))


# Hey! 0.62 is not bad. Let's summit that and see how we performe...
# 
# The following lines generate a csv to be sumbmitted.

# In[ ]:


#Loading test sample
data_test = open_data("test.csv")

#Setting all values to 0
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], 0] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])

#Saving as csv
solution.to_csv("solution_naive.csv", index=False)


# We got a 0.62679 accuraccy and ranked somewhere in the 11.200... Not great but a few hundreds people did worse.
# 
# Historically approximately two thirds of the passengers died, as a result if we set everyone to 0 we will get an accurracy of approximately 0.6. This will be our reference any approach that get a higher accurracy is somewhat good and any idea that get less than 0.6, well, is just s**t.

# **k-nearest neighbors**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


data = open_data("train.csv")
X, Y = param_label(data)
x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)


# In[ ]:


#Model creation
model = KNeighborsClassifier()
model.fit(x_train, y_train)
acc = metrics.accuracy_score(model.predict(x_test), y_test)

print("Accuracy : " + str(acc))


# That's, hem, slightly better. Maybe we can get a better accuracy by changing the number of neighboors?

# In[ ]:


neighboors = [i for i in range(1, 101)]

averages = []
mins = []
maxs = []

for n in neighboors:
    average_acc = 0
    min_acc = 1
    max_acc = 0
    
    #The accurracy may vary depending on the subset used, so we try 100 times with different subsets to get a better assessment.
    for i in range(100):
        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)
        model = KNeighborsClassifier(n_neighbors = n)
        model.fit(x_train, y_train)
        acc = metrics.accuracy_score(model.predict(x_test), y_test)
        
        average_acc = average_acc + acc
        if acc > max_acc: max_acc = acc
        if acc < min_acc: min_acc = acc
        
    averages = averages + [average_acc/100]
    mins = mins + [min_acc]
    maxs = maxs + [max_acc]
    
#Ploting results
plt.figure(figsize=(24,8))
plt.plot(averages, color = 'r', linewidth=2)
plt.plot(mins, color = 'r', linestyle='--')
plt.plot(maxs, color = 'r', linestyle='--')
plt.xticks(neighboors)


# It seems we will get the best results with the number of neighboors somewhere around 12. Let's try that...

# In[ ]:


#Model training
model = KNeighborsClassifier(n_neighbors = 12)
model.fit(X, Y)

#Results prediction and submission
data_test = open_data("test.csv")
prediction = model.predict(data_test.drop(["PassengerId"], 1))
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_KNN_allfeatures.csv", index=False)


# This time, we get 0.66507. OK, it's better but not really that better.
# 
# What if we reduce the number of features? For instance, if we just keep class and sex.

# In[ ]:


data = open_data("train.csv")
X, Y = param_label(data)
X = X[["Pclass", "Sex"]]
x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

model = KNeighborsClassifier()
model.fit(x_train, y_train)
acc = metrics.accuracy_score(model.predict(x_test), y_test)

print("Accuracy : " + str(acc))


# Whaou! That's a breakthrough. We can get an even better result by adjusting the number of neighbours:

# In[ ]:


model = KNeighborsClassifier(n_neighbors = 8)
model.fit(x_train, y_train)
acc = metrics.accuracy_score(model.predict(x_test), y_test)

print("Accuracy : " + str(acc))


# In[ ]:


#Model training
model = KNeighborsClassifier(n_neighbors = 8)
model.fit(X[["Pclass", "Sex"]], Y)

#Results prediction and submission
data_test = open_data("test.csv")
prediction = model.predict(data_test[["Pclass", "Sex"]])
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_KNN_Class_Sex.csv", index=False)


# This time we get 0.75598! Almost 10 points more than our previous attempt. And we enter the top 10.000. 
# 
# The key for a successful KNN classification seems to choose a limited amount of high-impact features. Maybe we can try to reduce the number of features from the original data set? For instance, instead of having 2 features SibSp and Parch we can have only one FamilyMembers which is the sum of the two previous one?

# In[ ]:


data["FamilyMembers"] = data["SibSp"]+data["Parch"]

X, Y = param_label(data)
X = X[["Pclass", "Sex", "FamilyMembers"]]
x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

model = KNeighborsClassifier(n_neighbors = 8)
model.fit(x_train, y_train)
acc = metrics.accuracy_score(model.predict(x_test), y_test)

print("Accuracy : " + str(acc))


# In[ ]:


#Model training
model = KNeighborsClassifier(n_neighbors = 8)
model.fit(X[["Pclass", "Sex", "FamilyMembers"]], Y)

#Results prediction and submission
data_test = open_data("test.csv")
data_test["FamilyMembers"] = data_test["SibSp"]+data_test["Parch"]
prediction = model.predict(data_test[["Pclass", "Sex","FamilyMembers"]])
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_KNN_Class_Sex_Family.csv", index=False)


# Nope. That wasn't better. We just got 0.72248 this time. Maybe it's time to move to another method?

# **Support vector classification**
# 
# First, let's try the SVM module from sklearn with default parameters:

# In[ ]:


from sklearn import svm

data = open_data("train.csv")
X, Y = param_label(data)
x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

model = svm.SVC()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
    
acc = metrics.accuracy_score(y_predict, y_test)
    
print("Accuracy:",acc)


# This look promising. Maybe we can try with different kernel?

# In[ ]:


kernels = ["rbf", "linear", "sigmoid", "poly"]

for kernel in kernels:
    model = svm.SVC(kernel = kernel)
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    
    acc = metrics.accuracy_score(y_predict, y_test)
    
    print("Accuracy with kernel =", kernel, ": ",acc)


# We get the best results with a linear kernel. Poly give us more or less the same accuracy but is far more time consuming. So let's go with linear:

# In[ ]:


#Model training
model = svm.SVC(kernel = 'linear')
model.fit(X,Y)

#Results prediction and submission
data_test = open_data("test.csv")
prediction = model.predict(data_test.drop(["PassengerId"], 1))
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_SVM.csv", index=False)


# We got 0.76555, our best score so far. Great!
# 
# Let's try to tweak other parameters like C (Penalty parameter) and gamma:

# In[ ]:


cs = [1, 5, 10, 15, 20]
gammas = [0.005, 0.01, 0.02, 0.05, 0.1]

for gamma in gammas:

    for c in cs:

        model = svm.SVC(kernel = 'rbf', C = c, gamma = gamma)
        model.fit(x_train, y_train)
        acc = metrics.accuracy_score(model.predict(x_test), y_test)
        print("Accuracy with gamma = ",gamma,"c = ",c,": ",acc)
        
    print("")


# In[ ]:


#Model training
model = svm.SVC(kernel = 'rbf', gamma =  0.01, C = 10)
model.fit(X,Y)

#Results prediction and submission
data_test = open_data("test.csv")
prediction = model.predict(data_test.drop(["PassengerId"], 1))
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_SVM_para.csv", index=False)


# This version got a disapointing 0.71770 accuracy rate. Let see if we can use another algorithme to improve our best score

# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


data = open_data("train.csv")
X, Y = param_label(data)
x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)


# In[ ]:


model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
y_predict
acc = metrics.accuracy_score(y_predict, y_test)

print(acc)


# Great start. Maybe we can improve by changing the number of trees in our forest?

# In[ ]:


trees = [5, 10, 20, 50, 100]

averages = []
mins = []
maxs = []

for tree in trees:
    average_acc = 0
    min_acc = 1
    max_acc = 0
    
    for i in range(100):
        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

        model = RandomForestClassifier(n_estimators = tree)
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        y_predict
        acc = metrics.accuracy_score(y_predict, y_test)

        average_acc = average_acc + acc
        if acc > max_acc: max_acc = acc
        if acc < min_acc: min_acc = acc

    averages = averages + [average_acc/100]
    mins = mins + [min_acc]
    maxs = maxs + [max_acc]
    
#Ploting results
plt.figure(figsize=(24,8))
plt.plot(averages, color = 'r', linewidth=2)
plt.plot(mins, color = 'r', linestyle='--')
plt.plot(maxs, color = 'r', linestyle='--')


# It seems the results are nor really improving when we add more than 20 trees. What abot changing the max depth of a tree?

# In[ ]:


depths = [1, 2, 5, 10, 15, 20]

averages = []
mins = []
maxs = []

for depth in depths:
    average_acc = 0
    min_acc = 1
    max_acc = 0
    
    for i in range(100):
        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)

        model = RandomForestClassifier(n_estimators = 20, max_depth = depth)
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        y_predict
        acc = metrics.accuracy_score(y_predict, y_test)

        average_acc = average_acc + acc
        if acc > max_acc: max_acc = acc
        if acc < min_acc: min_acc = acc

    averages = averages + [average_acc/100]
    mins = mins + [min_acc]
    maxs = maxs + [max_acc]
    
#Ploting results
plt.figure(figsize=(24,8))
plt.plot(averages, color = 'r', linewidth=2)
plt.plot(mins, color = 'r', linestyle='--')
plt.plot(maxs, color = 'r', linestyle='--')


# 10 seems to be a good guess. Let's try that:

# In[ ]:


#Model training
model = RandomForestClassifier(n_estimators = 20, max_depth = 10)
model.fit(X,Y)

#Results prediction and submission
data_test = open_data("test.csv")
prediction = model.predict(data_test.drop(["PassengerId"], 1))
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_Random_Forest.csv", index=False)


# We got 0.76076. Not bad.
# 
# In addition RandomForest has a nice tool that allow you to see the relative weight of each features:

# In[ ]:


feature_w = pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_w)


# What if we just use the four most important features? It seems we get consistently better results...

# In[ ]:


#Model training
model = RandomForestClassifier(n_estimators = 20, max_depth = 10)
model.fit(X[["Sex", "Age", "Fare", "Pclass"]],Y)

#Results prediction and submission
data_test = open_data("test.csv")
prediction = model.predict(data_test[["Sex", "Age", "Fare", "Pclass"]])
solution = pd.DataFrame(np.array([[data_test.PassengerId.iloc[i], prediction[i]] for i in range(len(data_test))]),
                        columns=['PassengerId', 'Survived'])
solution.to_csv("solution_Random_Forest_4features.csv", index=False)


# We score at 0.77990 and gain a few 3000 places. Not bad...
# 
# Now, let's move to another approach.

# **Neural Network**

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


data = open_data("train.csv")
X, Y = param_label(data)
x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)


# We will first normalize our data:

# In[ ]:


#I use .describe() to get mean and standard deviation
stats = X.describe()
stats = stats.transpose()

def norm(x):
    return (x - stats['mean']) / stats['std']

normed_x_train = norm(x_train)
normed_x_test = norm(x_test)


# In[ ]:


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', kernel_initializer = 'uniform', input_shape=[len(X.keys())]))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer = 'uniform'))
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer = 'uniform'))
    
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

model = build_model()
model.summary()


# Let see if it's working...

# example_batch = normed_x_train[:10]
# example_result = model.predict(example_batch)
# example_result

# In[ ]:


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['acc'])
    plt.ylim([0,1])
    plt.legend()

    plt.show()


# In[ ]:


model = build_model()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(normed_x_train, y_train, epochs=EPOCHS, 
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# In[ ]:


y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(x_test.shape[0])
metrics.accuracy_score(y_pred, y_test)


# We get approximately 0.8 with train data but far lower with test data. Our neural network is obviously overfitting. In addition, results seems to vary a lot on each run sometimes as low as 0.3 and sometimes above 0.6.
# 
# I modified the code to easily test various architecture:

# In[ ]:


architectures = [[12, 6],
                 [32, 16],
                 [64, 32],
                 [12, 12, 6],
                 [32, 16, 8],
                 [64, 32, 16],
                 [12, 12, 6, 6],
                 [32, 32, 16, 8],
                 [64, 32, 16, 8],
                 [64, 64, 32, 16, 8]]


# In[ ]:


def build_model(architecture):
    model = keras.Sequential()
    model.add(keras.layers.Dense(architecture[0], activation='relu', kernel_initializer = 'uniform', input_shape=[len(X.keys())]))
    
    for i in range(1, len(architecture)):
        n = architecture[i]
        model.add(keras.layers.Dense(n, activation='relu', kernel_initializer = 'uniform'))
        
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer = 'uniform'))
    
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model


# In[ ]:


EPOCHS = 1000

averages = []
mins = []
maxs = []

for architecture in architectures:
    model = build_model(architecture)
    
    average_acc = 0
    min_acc = 1
    max_acc = 0
    
    for i in range(100):
        x_train, x_test, y_train, y_test = subset_data(X, Y, 0.2)
        history = model.fit(normed_x_train, y_train, epochs=EPOCHS, 
                        validation_split = 0.2, verbose=0, callbacks=[early_stop])

        y_pred = model.predict(x_test)
        y_pred = (y_pred > 0.5).astype(int).reshape(x_test.shape[0])
        acc = metrics.accuracy_score(y_pred, y_test)
        
        average_acc = average_acc + acc
        if acc > max_acc: max_acc = acc
        if acc < min_acc: min_acc = acc
            
    averages = averages + [average_acc/100]
    mins = mins + [min_acc]
    maxs = maxs + [max_acc]
    
    print("Accuracy with architecture ", architecture, ": ", average_acc/100)
    
#Ploting results
plt.figure(figsize=(24,8))
plt.plot(averages, color = 'r', linewidth=2)
plt.plot(mins, color = 'r', linestyle='--')
plt.plot(maxs, color = 'r', linestyle='--')


# It's official: my neural network don't pass the test. I tried other architectures but almost never got more that 0.7. I probably did something wrong but I can't figure out what...
# 
# Anyway, that's all for this notebook. What did I learn in this notebook?
# 1. Trying multiple solutions and multiple parameters seems to be an efficient way to get better result even though I admit I used brute force more than intuition.
# 2. The choice of features appears to be crucial to obtain better results
# 3. Despite all the hype, neural network are hard to make work correctly. A simple algoritm with good parameters delivers better results than a neural network with a random architecture.
# 
# If you know other algorithms that I did not try or if you spot a way to improve the accuracy, feel free to share them in comment!
