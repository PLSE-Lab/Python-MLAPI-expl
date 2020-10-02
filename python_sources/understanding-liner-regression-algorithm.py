################################ PARAMETERS ####################################

'''
fit_intercept : boolean, optional, default True
whether to calculate the intercept for this model. 
If set to False, no intercept will be used in calculations 
(e.g. data is expected to be already centered).

normalize : boolean, optional, default False
This parameter is ignored when fit_intercept 
is set to False. If True, the regressors X will 
be normalized before regression by subtracting the 
mean and dividing by the l2-norm. If you wish to standardize, 
please use sklearn.preprocessing.StandardScaler before calling 
fit on an estimator with normalize=False.

copy_X : boolean, optional, default True
If True, X will be copied; else, it may be overwritten.

n_jobs : int or None, optional (default=None)
The number of jobs to use for the computation. 
This will only provide speedup for n_targets > 1 
and sufficient large problems. None means 1 unless in 
a joblib.parallel_backend context. -1 means using all processors. 
See Glossary for more details.
'''

######################### Visualising the points  ############################

from numpy import *
import matplotlib.pyplot as plt

def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

######################### Utilising Algorithm Code #############################

from sklearn.linear_model import LinearRegression
    
def Regression(X,Y):   
    regr = LinearRegression()
    regr.fit(X,Y)
    return regr
    
#################### Splitting Training and Testing Data #######################

def split_data(x,y):
    from sklearn.model_selection import train_test_split
    return train_test_split(x, y, test_size=0.1, random_state=42) # testing with 10% of data

############################## Main Code #######################################

points = genfromtxt("../input/data.csv", delimiter=",")
plt.plot(points[:,0], points[:,1], '.')

X=points[:,0].reshape(-1,1)
Y=points[:,1].reshape(-1,1)

x_train, x_test, y_train, y_test = split_data(X,Y)

regr = Regression(x_train,y_train)

plot_best_fit((regr.intercept_)[0],(regr.coef_)[0][0])

print('Score based on the R squraed error is:',regr.score(x_test,y_test)*100,'%')

plt.savefig("graph.png")

################################################################################





