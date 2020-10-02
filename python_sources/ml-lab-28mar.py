import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings

sns.set_style("darkgrid")

sns.set(rc={'figure.figsize':(8.7,6.27)})

warnings.filterwarnings('ignore') 

pd.options.display.max_columns=999 


import os
print(os.listdir("../input"))

data = pd.read_csv('../input/Iris.csv')

#data.head()

#data.info()

#data.describe()

#data['Species'].value_counts()


rows, col = data.shape
print("Rows : %s, column : %s" % (rows, col))


snsdata = data.drop(['Id'], axis=1)
g = sns.pairplot(snsdata, hue='Species', markers='x')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)




mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}

X = data.drop(['Id', 'Species'], axis=1).values 
y = data.Species.replace(mapping).values.reshape(rows,1) 

X = np.hstack(((np.ones((rows,1))), X))







np.random.seed(0)
theta = np.random.randn(1,5)

print("Theta : %s" % (theta))




iteration = 10000
learning_rate = 0.003 
J = np.zeros(iteration) 





for i in range(iteration):
    J[i] = (1/(2 * rows) * np.sum((np.dot(X, theta.T) - y) ** 2 ))
    theta -= ((learning_rate/rows) * np.dot((np.dot(X, theta.T) - y).reshape(1,rows), X))

prediction = np.round(np.dot(X, theta.T))

ax = plt.subplot(111)
ax.plot(np.arange(iteration), J)
ax.set_ylim([0,0.15])
plt.ylabel("Cost Values", color="Green")
plt.xlabel("No. of Iterations", color="Green")
plt.title("Mean Squared Error vs Iterations")
plt.show()



ax = sns.lineplot(x=np.arange(iteration), y=J)
plt.show()




ax = plt.subplot(111)

ax.plot(np.arange(1, 151, 1), y, label='Orignal value', color='red')
ax.scatter(np.arange(1, 151, 1), prediction, label='Predicted Value')

plt.xlabel("Dataset size", color="Green")
plt.ylabel("Iris Flower (1-3)", color="Green")
plt.title("Iris Flower (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")

ax.legend()
plt.show()



accuracy = (sum(prediction == y)/float(len(y)) * 100)[0]
print("The model predicted values of Iris dataset with an overall accuracy of %s" % (accuracy))