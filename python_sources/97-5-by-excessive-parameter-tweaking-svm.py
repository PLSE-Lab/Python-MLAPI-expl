import pandas as pd
import numpy as np
from sklearn.cross_validation import  train_test_split
import math
from sklearn import preprocessing, svm
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from mpl_toolkits.mplot3d import axes3d
import math

start_time = time.time()

df =  pd.read_csv('../input/data.csv', header=0)
df[df.columns[1]] = df[df.columns[1]].map( {'B': 0, 'M': 1} ).astype(int)

df.replace('?', -99999, inplace=True)
df.drop(['id','Unnamed: 32'], 1, inplace=True)
X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])
Features=X
Labels=y

#The core of the script is a 3 level function.The lowest level, runs the number of SVM fits on a set combination of C and Gamma, each time newly splitting the data.
# Accuracy is recorded.

def Level_three_simulations(X_est,y_est,C_index,Gamma_index):
    accuracies=[]
    for i in range(number_of_simulations):
        X = preprocessing.scale(X_est)
        X_train, X_test, y_train, y_test = train_test_split(X, y_est, test_size=0.25)
        clf = svm.SVC(C=C_values[C_index],gamma=Gamma_values[Gamma_index], kernel='rbf')
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
    return accuracies

#Level two loops over the values of C and averages the accuracies
def Level_two_C(X,y,Gamma_index):
    C_dependent_acc=[]
    for C_index in range(len(C_values)):
        accuracies=Level_three_simulations(X,y,C_index,Gamma_index)
        mean_acc = (sum(accuracies) / len(accuracies))
        C_dependent_acc.append(mean_acc)
    return C_dependent_acc

# Level three loops over the values of Gamma and returns an array Z containing the mean accuracies of all tested combinations
def Level_one_Gamma(X,y):
    for Gamma_index in  range(len(Gamma_values)):
        C_dependent_acc = Level_two_C(X,y,Gamma_index)
        Z[Gamma_index]=C_dependent_acc
    return Z


#The main purpose of the callibration loop is to identify areas of the tested parameter space that did not contribute to maximum accuracies on the Gamma and C dimension
# respectively. Those areas are cut off and the new limits of the parameter space are fed to a new cycle of the callibration loop. If the previously identified overall
# maximum happens to lie on the edge of the space, the tested parameter is extented in this direction.


def callibration_loop(C_min,C_max,Gamma_min,Gamma_max):
    start_time = time.time()
    global C_values, Gamma_values,Z, number_of_simulations,current_loop

    C_range = np.linspace(C_min, C_max, 11) #C and Gamma range always spans an 11x11 grid
    Gamma_range = np.linspace(Gamma_min, Gamma_max, 11)
    # The C and Gamma dimensions are on a logarithimcal scale
    C_values = [math.pow(10, x / 1) for x in C_range]
    Gamma_values = [math.pow(10, x / 1) for x in Gamma_range]
    print('Loop Nr:',current_loop)
    print('The current loop covers C from',C_values[0],'to',C_values[-1],'.')
    print('The current loop covers Gamma from',Gamma_values[0],'to',Gamma_values[-1],'.')

    Y = np.array(Gamma_values)
    X = np.array(C_values)
    X, Y = np.meshgrid(X, Y)
    Z = X * Y
    # Calling the simulation functions
    Z = Level_one_Gamma(Features, Labels)
    maxi = np.amax(Z)               #Value of the maximum accurcay
    indices = np.where(Z == Z.max())# and its indices
    opt_Gamma=Gamma_values[indices[0][0]]
    opt_C=C_values[indices[1][0]]
    opt_dict[maxi]=[opt_Gamma,opt_C]    #Saving it to a dictionary
    opt_list.append(maxi)


    max_coord_C = np.argmax(Z, axis=1)
    C = []
    Gamma = []
    for index in range(len(max_coord_C)):
        C.append(C_values[max_coord_C[index]])# Identification of the levels of C that 'hosted' a maximum along the Gamma dimension
        Gamma.append(Gamma_values[index]) #These values only play a role in the 3D plotting after last cycle

    max_coord_C = np.argmax(Z, axis=0)
    C1 = []
    Gamma1 = []
    for index in range(len(max_coord_C)):
        Gamma1.append(Gamma_values[max_coord_C[index]])# Identification of the levels of Gamma that 'hosted' a maximum along the C dimension
        C1.append(C_values[index]) #These values only play a role in the 3D plotting after last cycle

    #All levels of Gamma and C that did not 'host' a maximum are cut off and the new maximum and minimum values for the following cycle are defined
    C_min_index=C_values.index(min(C))
    C_max_index=C_values.index(max(C))
    Gamma_min_index=Gamma_values.index(min(Gamma1))
    Gamma_max_index=Gamma_values.index(max(Gamma1))

    C_min=C_range[C_min_index]
    C_max=C_range[C_max_index]
    Gamma_min=Gamma_range[Gamma_min_index]
    Gamma_max=Gamma_range[Gamma_max_index]

    current_loop=current_loop+1

    print('The best combination of the Parameters Gamma and C are: Gamma = ', opt_Gamma, ', C=',opt_C)
    print('The accuracy obtained with this combination is', maxi * 100,
          '%. The out-of-sample accuracy on a Cross Validation set is likely to be lower.')
    print()
    # In case the gloabal maximum was on the edge of the space, limits of this dimension are extended in this direction
    if indices[0][0] == 0 or indices[0][0] == len(Gamma_values) - 1 or indices[1][0] == 0 or indices[1][0] == len(C_values) - 1:
        print(
            'Warning: At least one of the estimated optimal paramters lies on the margin of the tested space. There is a chance the optimal combination is missed.')
        print('The next loop will extend the tested range in that direction.')
    if indices[0][0] ==0:
        Gamma_min=Gamma_min-1
    if indices[0][0]==len(Gamma_values)-1:
        Gamma_max=Gamma_max+1
    if indices[1][0]==0:
        C_min=C_min-1
    if indices[1][0]== len(C_values) - 1:
        C_max=C_max+1
    print("--- %s seconds ---" % (time.time() - start_time))
    print('')
    print('')
    print('')
    if current_loop<number_of_loops+1:
        callibration_loop(C_min, C_max, Gamma_min, Gamma_max)
#Now as a set number of cycle has been looped through, the last parameter space is plotted in 3D. Also another graph is produced showing the progression of the maximum
# accuracy throughout the loops. Finally these maxima are subjected to another accuracy testing, using 10 times the simulations as in the regular loops. This helps to
# protect from overfitting by counteracting effects of randomness. After this treatment, the Parameter combination with the highest accuracy is presented as
# recommendation
    else:
        #A 3D Figure shows the accuaries of each parameter combination. 2 lines show how these maxima behave as function of either one of the parameter.
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')  # set the 3d axes
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=0.3, cmap=cm.BuPu)
        ax.plot(C, Gamma, np.amax(Z, axis=1), label='Path along Gamma')
        ax.plot(C1, Gamma1, np.amax(Z, axis=0), color='g', label='Path along C')
        ax.scatter(C_values[indices[1][0]], Gamma_values[indices[0][0]], maxi, marker='*', color='r',
                   label='Maximum Accuracy')
        ax.set_title('Parameter Space C-Gamma')
        ax.set_xlabel('C')
        ax.set_ylabel('Gamma')
        ax.set_zlabel('Accuracy')
        ax.legend()
        # Another simple graph shows the behavior of maximum accuracies with the progression of the loops
        fig1 = plt.figure(2)
        ax1 = fig1.gca()
        ax1.plot(list(range(number_of_loops)),opt_list)
        plt.show()

        #All previously identified maxima are subjected to another testing with 10 times the number of simulations
        norms = sorted([n for n in opt_dict])
        number_of_simulations = number_of_simulations*10
        opt_dict1={}
        for i in norms:
            Gamma_values=[opt_dict[i][0]]
            C_values=[opt_dict[i][1]]
            Z=Level_one_Gamma(Features,Labels)
            opt_dict1[Z.mean()]=[opt_dict[i][0],opt_dict[i][1]]
        norms = sorted([n for n in opt_dict1])
        opt_choice1 = opt_dict1[norms[-1]]
        print('Most accurate Parameter combination: Gamma=',opt_choice1[0],', C=',opt_choice1[1],', attaining an accuracy of:',100*norms[-1],'%.')
        print("--- Total Time: %s seconds ---" % (time.time() - start_time_total))

# All parameters are defined for the first loop.
start_time_total = time.time()
opt_dict={}
opt_list=[]
Gamma_max=math.log(10/X.shape[1],10)    #the initial Gamma ranges from a tenth to tenfold of 1/Nr.of features
Gamma_min=math.log(0.1/X.shape[1],10)
C_max=math.log(10,10)   #The intial C ranges from 0.1 to 10.
C_min=math.log(0.1,10)
current_loop=1
number_of_loops=10
number_of_simulations = 15
callibration_loop(C_min, C_max, Gamma_min, Gamma_max)