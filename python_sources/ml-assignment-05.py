#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Assignment 05

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.neural_network as nn
import sklearn.metrics as metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


spiral_df = pd.read_csv("/kaggle/input/SpiralWithCluster.csv")
spiral_df.head()


# # Question 1

# In[ ]:


import seaborn as sns
sns.countplot(spiral_df.SpectralCluster)


# a)	(5 points) What percent of the observations have SpectralCluster equals to 1?

# In[ ]:


count = 0
for i in spiral_df.SpectralCluster:
    if i == 1:
        count+=1
print("Percent of the observations having SpectralCluster equals to 1",count/len(spiral_df.SpectralCluster)*100)


# Get the dependent(target) and independent(predictors) variables seperated from the dataframe

# In[ ]:


target =  spiral_df.SpectralCluster
predictors = spiral_df[["x","y"]]


# In[ ]:


def build_NN(activation_function, number_of_layers, number_of_neurons):
    clf = nn.MLPClassifier(activation = activation_function, hidden_layer_sizes = (number_of_neurons,)*number_of_layers, learning_rate_init=0.1,  max_iter=5000, random_state = 20191108, solver = "lbfgs")
    model = clf.fit(predictors,target)
    
    pred = clf.predict(predictors)
    
    loss = clf.loss_
    pred_proba = pd.DataFrame(data=clf.predict_proba(predictors),columns = ["A0","A1"])
    misclass = misclassification_rate(pred_proba["A1"],target)
    
    return (loss,misclass,clf.n_iter_)


# In[ ]:


def misclassification_rate(pred_proba,target):
    threshold = 0.50
    counter = 0
    answer =[]
    for i in pred_proba:
        if i > threshold:
            answer.append(1)
        else:
            answer.append(0)
    for j in range(len(answer)):
        if answer[j] != target[j]:
            counter += 1
    return (counter/len(answer))            
    


# b)	(15 points) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.  You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1. Your search will be done over a grid that is formed by cross-combining the following attributes: (1) activation function: identity, logistic, relu, and tanh; (2) number of hidden layers: 1, 2, 3, 4, and 5; and (3) number of neurons: 1 to 10 by 1.  List your optimal neural network for each activation function in a table.  Your table will have four rows, one for each activation function.  Your table will have five columns: (1) activation function, (2) number of layers, (3) number of neurons per layer, (4) number of iterations performed, (5) the loss value, and (6) the misclassification rate.

# In[ ]:


result = pd.DataFrame(columns = ['Number of Layers', 'Number of Neurons', 'Loss', 'Misclassification Rate', "Activation Function","Number of Iterations"])

activaton_function = ['relu' , 'identity', 'logistic', 'tanh']

for i in activaton_function:
    for j in np.arange(1,6):
        for k in np.arange(1,11):
            loss, rsquared, iterations = build_NN(activation_function = i, number_of_layers = j, number_of_neurons = k)
            result = result.append(pd.DataFrame([[j, k, loss, rsquared , i, iterations]],columns = ['Number of Layers', 'Number of Neurons', 'Loss', 'Misclassification Rate', "Activation Function","Number of Iterations"]))


# In[ ]:


result.sort_values(by=['Loss','Misclassification Rate'])


# d)	(5 points) Which activation function, number of layers, and number of neurons per layer give the lowest loss and the lowest misclassification rate?  What are the loss and the misclassification rate?  How many iterations are performed?

# In[ ]:


relu = result[result["Activation Function"] == "relu"]
identity = result[result["Activation Function"] == "identity"]
logistic = result[result["Activation Function"] == "logistic"]
tanh = result[result["Activation Function"] == "tanh"]


# In[ ]:


a = relu[relu["Loss"] == relu["Loss"].min()]
b = identity[identity["Loss"] == identity["Loss"].min()]
c = logistic[logistic["Loss"] == logistic["Loss"].min()]
d = tanh[tanh["Loss"] == tanh["Loss"].min()]
a.append([b,c,d])


# Model with minimum Loss value

# In[ ]:


optimal_clf = nn.MLPClassifier(activation = "relu", hidden_layer_sizes = (8,)*4, learning_rate_init=0.1,  max_iter=5000, random_state = 20191108, solver = "lbfgs",verbose=True)
model = optimal_clf.fit(predictors,target)
pred_proba = pd.DataFrame(data=optimal_clf.predict_proba(predictors),columns = ["A0","A1"])
missclass = misclassification_rate(pred_proba["A1"],target)
pred = optimal_clf.predict(predictors)
spiral_df['NLPpredictions'] = pred
pred_proba


# In[ ]:


spiral_df.head()


# e)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue) from the optimal MLP in (d).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.color_palette("RdBu_r", 7)
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')
colors = ['red','blue']
for i in range(2):
    suBData = spiral_df[spiral_df['NLPpredictions']==i]
    plt.scatter(suBData.x,suBData.y,c = colors[i],label=i)
    #plt.legend()
plt.title("Scatterplot according to Cluster Values Predicted by optimal neural network")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.legend()


# f)	(5 points) What is the count, the mean and the standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP in (d) by value of the SpectralCluster?  Please give your answers up to the 10 decimal places.

# In[ ]:


lsit = []
for i in pred_proba["A1"]:
    if i > 0.5:
        lsit.append(i)
        
print("The mean of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP is",np.mean(lsit))
print("The standard deviation of the predicted probability Prob(SpectralCluster = 1) from the optimal MLP is",np.std(lsit))

pd.DataFrame(data = lsit, columns=["p"]).describe()
    


# c)	(5 points) What is the activation function for the output layer?

# In[ ]:


print("The activation function for the output layer is",optimal_clf.out_activation_)


# In[ ]:


pred_proba["A1"].describe()


# ## Question 2

# In[ ]:


from sklearn.svm import SVC
svm_clf = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
svm_clf.fit(predictors,target)
svm_pred = svm_clf.predict(predictors)
#metrics.accuracy_score(svm_pred,target)


# b)	(5 points) What is the misclassification rate?

# In[ ]:


pred_proba_result = pd.DataFrame(data=svm_clf.predict_proba(predictors),columns = ["A0","A1"])
pred_proba_result["A0"]
missclass = misclassification_rate(pred_proba_result["A1"],target)
print("The miscalssification rate before transformation is",missclass)


# In[ ]:


svm_clf = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
svm_clf.fit(predictors,target)
spiral_df["SVMClusters"] = svm_clf.predict(predictors)


# c)	(5 points) Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the predicted SpectralCluster (0 = Red and 1 = Blue).  Besides, plot the hyperplane as a dotted line to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes.

# In[ ]:


w = svm_clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm_clf.intercept_[0]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(xx, yy, 'k--')


for i in range(2):
    subdata = spiral_df[spiral_df["SVMClusters"]==i]
    plt.scatter(subdata.x,subdata.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot according to Cluster Values")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")


# a)	(5 points) What is the equation of the separating hyperplane?  Please state the coefficients up to seven decimal places.

# In[ ]:


print ('THe equation of the seperating hyperplane is')
print (svm_clf.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")


# In[ ]:


def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)


# In[ ]:


trainData = pd.DataFrame(columns = ["radius","theta"])
trainData['radius'] = np.sqrt(spiral_df['x']**2 + spiral_df['y']**2)
trainData['theta'] = (np.arctan2(spiral_df['y'], spiral_df['x'])).apply(customArcTan)


# (d) (10 points) Please express the data as polar coordinates.  Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster variable (0 = Red and 1 = Blue).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes. 

# In[ ]:


trainData['class']=spiral_df["SpectralCluster"]
trainData.head()


# In[ ]:


colur = ['red','blue']
for i in range(2):
    subdata = trainData[trainData["class"]==i]
    plt.scatter(subdata.radius,subdata.theta,label = (i),c = carray[i])
    
plt.title("Scatterplot of Polar Co-ordinates")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()
plt.grid()


# In[ ]:


def customArcTan (z):
    theta = np.where(z < 0.0, 2.0*np.pi+z, z)
    return (theta)


# (e) (10 points) You should expect to see three distinct strips of points and a lone point.  Since the SpectralCluster variable has two values, you will create another variable, named Group, and use it as the new target variable. The Group variable will have four values. Value 0 for the lone point on the upper left corner of the chart in (d), values 1, 2,and 3 for the next three strips of points. 
#  
# Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black).  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes. 
# 
#  

# In[ ]:


x = trainData["radius"]
y = trainData['theta'].apply(customArcTan)
svm_dataframe = pd.DataFrame(columns = ['Radius','Theta'])
svm_dataframe['Radius'] = x
svm_dataframe['Theta'] = y

group = []

for i in range(len(x)):
    if x[i] < 1.5 and y[i]>6:
        group.append(0)
        
    elif x[i] < 2.5 and y[i]>3 :
        group.append(1)
    
    elif 2.75 > x[i]>2.5 and y[i]>5:
        group.append(1)
        
    elif 2.5<x[i]<3 and 2<y[i]<4:
        group.append(2)      
        
    elif x[i]> 2.5 and y[i]<3.1:
        group.append(3)
        
    elif x[i] < 4:
        group.append(2)
        

svm_dataframe['Class'] = group
colors = ['red','blue','green','black']
for i in range(4):
    sub = svm_dataframe[svm_dataframe.Class == i]
    plt.scatter(sub.Radius,sub.Theta,c = colors[i],label=i)
plt.grid()
plt.title("Scatterplot with four Groups")
plt.xlabel("Radius")
plt.ylabel('Theta Co-ordinate')
plt.legend()
#svm_dataframe


# (f) (10 points) Since the graph in (e) has four clearly separable and neighboring segments, we will apply the Support Vector Machine algorithm in a different way.  Instead of applying SVM once on a multi-class target variable, you will SVM three times, each on a binary target variable. 
# 
# SVM 0: Group 0 versus Group 1 
# SVM 1: Group 1 versus Group 2 
# SVM 2: Group 2 versus Group 3 
# Please give the equations of the three hyperplanes. 
# 
# (g) (5 points) Please plot the theta-coordinate against the radius-coordinate in a scatterplot.  Please color-code the points using the new Group target variable (0 = Red, 1 = Blue, 2 = Green, 3 = Black). Please add the hyperplanes to the graph. To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes. 
# 
# 

# In[ ]:


#SVM to classify class 0 and class 1
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 0]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 1])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(1, 2)
yy = a * xx - (svm_1.intercept_[0])/w[1] 

print ('THe equation of the hypercurve for SVM 0 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h0_xx = xx * np.cos(yy[:])
h0_yy = xx * np.sin(yy[:])

carray=['red','blue','green','black']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to classify class 1 and class 2
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 1]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 2])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(1, 4)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('THe equation of the hypercurve for SVM 1 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h1_xx = xx * np.cos(yy[:])
h1_yy = xx * np.sin(yy[:])


#Plot ther hyperplane
plt.plot(xx, yy, 'k--')

#SVM to. classify class 2 and class 3
svm_1 = SVC(kernel = "linear", random_state=20191108, decision_function_shape='ovr',max_iter=-1,probability = True)
x = svm_dataframe[svm_dataframe['Class'] == 2]
x = x.append(svm_dataframe[svm_dataframe['Class'] == 3])
td = x[['Radius','Theta']]
svm_1.fit(td,x.Class)

w = svm_1.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(2, 4.5)
yy = a * xx - (svm_1.intercept_[0])/w[1] 
print ('THe equation of the hypercurve for SVM 2 is')
print (svm_1.intercept_[0], " + (", w[0], ") X +(" ,w[1],") Y = 0")

h2_xx = xx * np.cos(yy[:])
h2_yy = xx * np.sin(yy[:])


#Plot ther hyperplane
plt.plot(xx, yy, 'k--')


for i in range(4):
    sub = svm_dataframe[svm_dataframe.Class == i]
    plt.scatter(sub.Radius,sub.Theta,c = carray[i],label=i)
plt.xlabel("Radius")
plt.ylabel("Theta Co-Ordinate")
plt.title("Scatterplot of the polar co-ordinates with 4 diffrent classes seperated by 3 hyperplanes")
plt.legend()


# (h) (10 points) Convert the observations along with the hyperplanes from the polar coordinates back to the Cartesian coordinates. Please plot the y-coordinate against the x-coordinate in a scatterplot.  Please color-code the points using the SpectralCluster (0 = Red and 1 = Blue). Besides, plot the hyper-curves as dotted lines to the graph.  To obtain the full credits, you should properly label the axes, the legend, and the chart title.  Also, grid lines should be added to the axes. 
#  
# Based on your graph, which hypercurve do you think is not needed? 
# 
#  
# 
#  

# In[ ]:


carray=['red','blue']
fig, ax = plt.subplots(1, 1)
ax.grid(b=True, which='major')

plt.plot(h0_xx, h0_yy, 'k--')
plt.plot(h1_xx, h1_yy, 'k--')
plt.plot(h2_xx, h2_yy, 'k--')

for i in range(2):
    subdata = spiral_df[spiral_df["SpectralCluster"]==i]
    plt.scatter(subdata.x,subdata.y,label = (i),c = carray[i])
plt.legend()
plt.title("Scatterplot of the cartesian co-ordinates seperated by the hypercurve (one hypercurve removed)")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")


# By looking at the plot, we can say that we dont need the smallest hypercurve in the centre in our plot. It does not provide any beneficial information to us and can be dropped
