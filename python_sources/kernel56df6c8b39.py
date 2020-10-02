import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import csv
import math
from math import radians, sin, cos, acos
# Global variables
phase = "train"  # phase can be set to either "train" or "eval"

""" 
You are allowed to change the names of function arguments as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.

"""

def get_features(file_path):
	# Given a file path , return feature matrix and target labels 
	# feature matrix consists of normalized distance and num_people
    divisions = 2
    cols = pd.read_csv(file_path,sep=",", nrows=1).columns
    if file_path=='test.csv':
        phi = pd.read_csv(file_path, sep=",")
    else:    
        phi = pd.read_csv(file_path, sep=",", usecols=cols[:-1])
    data = np.array(phi)
    
    train_data_t = data.T
    
    y = pd.read_csv(file_path, sep=",", usecols=cols[-1:])
    taxi_fare = np.array(y)
    R = 6373

    lat1 = (data.T[2]) #pickup latitude
    lat2 = (data.T[4]) #dropoff lat
    lon1 = (data.T[1]) #pickup long
    lon2 = (data.T[3]) #dropoff long
    dlon = list((lon2 - lon1)/2)
    dlat = list((lat2 - lat1)/2)   
    dist = 6371.01 * np.arccos(np.sin(list(lat1))*np.sin(list(lat2)) + np.cos(list(lat1))*np.cos(list(lat2))*np.cos(list(lon1-lon2)))
    dist = dist.reshape(dist.shape[0],1)
    print(np.var(dist)) 
    ndist = (dist-np.mean(dist))/np.sqrt(np.var(dist))
    
    num_people=data.T[-1] 
    
    num_people= num_people.reshape(dist.shape[0],1)
    nnum_people= (num_people-np.mean(num_people))/np.sqrt(np.var(num_people))
    X = np.hstack((dist,num_people))
    taxi_fare= taxi_fare.reshape(dist.shape[0],1)
    #ny = (taxi_fare-np.mean(taxi_fare))/np.var(taxi_fare)
    return X  ,taxi_fare
    
def limit_training_data(file_path,size):
    # Given a file path , return feature matrix and target labels 
    # feature matrix consists of normalized distance and num_people
    divisions = 2
    cols = pd.read_csv(file_path,sep=",", nrows=1).columns
    
    phi = pd.read_csv(file_path, sep=",", usecols=cols[:-1])
    data = np.array(phi)
    original_size = data.shape[0]
    data = data[:size,:]
    train_data_t = data.T
    
    y = pd.read_csv(file_path, sep=",", usecols=cols[-1:])
    taxi_fare = np.array(y)
    taxi_fare = taxi_fare[:size,:]
    R = 6373

    lat1 = (data.T[2]) #pickup latitude
    lat2 = (data.T[4]) #dropoff lat
    lon1 = (data.T[1]) #pickup long
    lon2 = (data.T[3]) #dropoff long
    dlon = list((lon2 - lon1)/2)
    dlat = list((lat2 - lat1)/2)

    num = taxi_fare.shape[0]
    
    dist = 6371.01 * np.arccos(np.sin(list(lat1))*np.sin(list(lat2)) + np.cos(list(lat1))*np.cos(list(lat2))*np.cos(list(lon1-lon2)))
    dist = dist.reshape(dist.shape[0],1) 
    ndist = (dist-np.mean(dist))/np.sqrt(np.var(dist))
    
    num_people=data.T[-1] 
    
    num_people= num_people.reshape(dist.shape[0],1)
    nnum_people= (num_people-np.mean(num_people))/np.sqrt(np.var(num_people))
    X = np.hstack((ndist,nnum_people))
    taxi_fare= taxi_fare.reshape(taxi_fare.shape[0],1)
    #ny = (taxi_fare-np.mean(taxi_fare))/np.var(taxi_fare)
    return X  ,taxi_fare

def get_features_basis1(file_path):
	# Given a file path , return feature matrix and target labels 
	
    divisions = 2
    cols = pd.read_csv(file_path,sep=",", nrows=1).columns
    
    phi = pd.read_csv(file_path, sep=",", usecols=cols[:-1])
    data = np.array(phi)
    
    train_data_t = data.T
    
    y = pd.read_csv(file_path, sep=",", usecols=cols[-1:])
    taxi_fare = np.array(y)
    R = 6373
    lat1 = (data.T[2]) #pickup latitude
    lat2 = (data.T[4]) #dropoff lat
    lon1 = (data.T[1]) #pickup long
    lon2 = (data.T[3]) #dropoff long
    dlon = list((lon2 - lon1)/2)
    dlat = list((lat2 - lat1)/2)

    dist = 6371.01 * np.arccos(np.sin(list(lat1))*np.sin(list(lat2)) + np.cos(list(lat1))*np.cos(list(lat2))*np.cos(list(lon1-lon2)))
    
    datetime = np.char.split(list(data.T[0]), sep =' ')
    hour=[]
    for i in datetime:
        hour.append(int(int(i[1].split(":")[0])/divisions))
    X=np.empty((int(24/divisions)+1))
    first = True
    for i,k in enumerate(hour):
        if first:
            X = np.zeros((int(24/divisions)+1))
            X[0] = 1
            first = False   
        else:
            X = np.vstack((X, np.hstack((np.ones(1),np.zeros(int(24/divisions))))))
            X[i][k+1] = 1
    dist = dist.reshape(X.shape[0],1) 
    num_people=data.T[-1] 
    num_people= num_people.reshape(X.shape[0],1) 
    X = np.hstack((X,np.hstack((dist,num_people))))
	
    return X, y

def get_features_basis2(file_path):
	# Given a file path , return feature matrix and target labels 
    divisions = 2
    hour_slot=4
    cols = pd.read_csv(file_path,sep=",", nrows=1).columns

    phi = pd.read_csv(file_path, sep=",", usecols=cols[:-1])
    data = np.array(phi)

    train_data_t = data.T

    y = pd.read_csv(file_path, sep=",", usecols=cols[-1:])
    taxi_fare = np.array(y)
    R = 6373
    lat1 = (data.T[2]) #pickup latitude
    lat2 = (data.T[4]) #dropoff lat
    lon1 = (data.T[1]) #pickup long
    lon2 = (data.T[3]) #dropoff long
    dlon = list((lon2 - lon1)/2)
    dlat = list((lat2 - lat1)/2)

    a = np.sin(dlat)**2 + np.cos(list(lat1)) * np.cos(list(lat2)) * np.sin(dlon)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    dist = 6371.01 * np.arccos(np.sin(list(lat1))*np.sin(list(lat2)) + np.cos(list(lat1))*np.cos(list(lat2))*np.cos(list(lon1-lon2)))

    datetime = np.char.split(list(data.T[0]), sep =' ')
    hour=[]
    minute=[]
    for i in datetime:
        hour.append(int(int(i[1].split(":")[0])/divisions))
    for i in datetime:
        minute.append(int(int(i[1].split(":")[1])*hour_slot/60))
    X=np.zeros(int(24/divisions)+hour_slot+1)
    for i,k in enumerate(hour):
        X = np.vstack((X, np.hstack((np.ones(1),np.zeros(int(24/divisions)), np.zeros(hour_slot)))))
        X[i][k+1] = 1
    for i,k in enumerate(minute):
        X[i][k+13] = 1
    

    return X, y

def compute_RMSE(phi, w , y) :
	# Root Mean Squared Error
    m = phi.shape[0]
    error = np.sum((y-phi.dot(w))**2)
    return math.sqrt(error)/m

def RMSE_size(phi, w , y, size) :
    # Root Mean Squared Error
    return compute_RMSE(phi[:size,:],w,y)

def generate_output(phi_test, w):
    
    l=phi_test.dot(w)

    out = open('output.csv', 'w')
    out.write('ID,Fare\n')
    for idi,i in enumerate(l):
        for column in i:
            out.write(str(idi)+','+str(column))
            out.write('\n')
        
    out.close()
	# writes a file (output.csv) containing target variables in required format for Kaggle Submission.
def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)
	
def gradient_descent(phi, y) :
	# Mean Squared Error
    m = phi.shape[0]
    w= np.random.rand(phi.shape[1]).reshape((phi.shape[1],1))
    w_old = np.zeros(phi.shape[1]).reshape((phi.shape[1],1))
    
    learning_rate = 0.001

    while np.sum((w_old-w)**2)>0.000000001:
        prediction = np.dot(phi, w)
        w_old = w
        w = w - (1/m)*learning_rate*((phi.T).dot(prediction-y))
        
    return w

def sgd(phi, y) :
	# Mean Squared Error

    m = phi.shape[0]
    w= np.random.rand(phi.shape[1]).reshape((phi.shape[1],1))
    w_old = np.zeros(phi.shape[1]).reshape((phi.shape[1],1))
    
    learning_rate = 0.001
    num_divisions = 10
    phi_split=np.array_split(phi, num_divisions)
    print(phi_split[0].shape[0])
    y = y[:phi_split[0].shape[0],:]
    while np.sum((w_old-w)**2)>0.000000001:
        for subphi in phi_split:
            prediction = np.dot(subphi, w)
            w_old = w
            print(subphi.T.shape)
            print(prediction.shape)
            print(y.shape)
            #w = w - (1/m)*learning_rate*((subphi.T).dot(prediction-y))
            if np.sum((w_old-w)**2)<0.000000001:
                break
    return w


def pnorm(phi, y, p) :
	# Mean Squared Error
    lamda=10
    w = gradient_descent(phi,y)
    w_t=np.sum(np.power(w,p))
    if p==2:
        wi=np.sqrt(w_t)
    elif p==4:
        wi=np.sqrt(np.sqrt(w_t))
  
    return lamda*wi	

	
def main():
    '''
    The following steps will be run in sequence by the autograder.
    '''
    ######## Task 1 #########
    phase = "train"
    phi, y = get_features('train.csv')
    w1 = closed_soln(phi.astype('float'), y.astype('float'))
    w2 = gradient_descent(phi, y)
    phase = "eval"
    phi_dev, y_dev = get_features('dev.csv')
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)

    print('1a: ')
    print(abs(r1-r2))

    w3 = sgd(phi, y)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 2 #########
    w_p2 = pnorm(phi, y, 2)  
    w_p4 = pnorm(phi, y, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)

    num_obs = [10000,15000,20000,30000,60000]
    y =[]
    for obs in num_obs:
         phi_k, y_k = limit_training_data('train.csv',obs)
         w_k = gradient_descent(phi_k, y_k)
         y.append(compute_RMSE(phi_k,w_k,y_k))
    plt.plot(num_obs, y) 

    # # naming the x axis 
    plt.xlabel('x - axis') 
    # naming the y axis 
    plt.ylabel('y - axis') 

    # giving a title to my graph 
    plt.title('My first graph!') 

    # function to show the plot 
    plt.show() 

    phi_test, y_test = get_features('test.csv')

    print(phi_test)
    generate_output(phi_test,w2)
    # ######## Task 3 #########
    phase = "train"
    phi1, y = get_features_basis1('train.csv')
    phi2, y = get_features_basis2('train.csv')
    phase = "eval"
    phi1_dev, y_dev = get_features_basis1('dev.csv')
    phi2_dev, y_dev = get_features_basis2('dev.csv')
    w_basis1 = pnorm(phi1, y, 2)  
    w_basis2 = pnorm(phi2, y, 2)  
    rmse_basis1 = compute_RMSE(phi1_dev, w_basis1, y_dev)
    rmse_basis2 = compute_RMSE(phi2_dev, w_basis2, y_dev)
    print('Task 3: basis1')
    print(rmse_basis1)
    print('Task 3: basis2')
    print(rmse_basis2)

    main()
