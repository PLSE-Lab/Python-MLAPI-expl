###################### Implementing Feature Scaling #######################

from sklearn.preprocessing import MinMaxScaler

def feature_Scaling(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return (scaler.transform(data))
    
######################### Visualising the points  ############################

from numpy import *
import matplotlib.pyplot as plt

def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

######################### Utilising Algorithm Code ###########################

from sklearn.linear_model import LinearRegression
    
def Regression(X,Y):   
    regr = LinearRegression()
    regr.fit(X,Y)
    return regr

############################## Main Code #######################################

points = genfromtxt("../input/data.csv", delimiter=",")

points = feature_Scaling(points)

plt.plot(points[:,0], points[:,1], '.')

X=points[:,0].reshape(-1,1)
Y=points[:,1].reshape(-1,1)

X=feature_Scaling(X)
Y=feature_Scaling(Y)

regr = Regression(X,Y)

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

print('Score based on the R squraed error is:',regr.score(X,Y))

plt.savefig("graph.png")

################################################################################
