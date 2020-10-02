import numpy as np # linear algebra
import pandas as pd # data processing 
import matplotlib.pyplot as plt # to plot graph
import matplotlib.animation as animation # for animation
#Load the dataset
data = pd.read_csv("../input/kc_house_data.csv")
data.head()
data.shape
#grab data ,preditor variable & add column of 1's for gradident descent
x = data['sqft_living']
y = data['price']
x = (x - x.mean()) / x.std()
x
x = np.c_[np.ones(x.shape[0]), x] 
x
#GRADIENT DESCENT
alpha = 0.01 #Step size
iterations = 1000 #No. of iterations
m = y.size #No. of data points
np.random.seed(123) #Set the seed
theta = np.random.rand(2) #Pick some random values to start with
theta
#GRADIENT DESCENT
def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas,past_costs

#Pass the relevant variables to the function and get the new values back...
past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
theta = past_thetas[-1]
#Print the results...
print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))

#Plot the cost function...
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()

