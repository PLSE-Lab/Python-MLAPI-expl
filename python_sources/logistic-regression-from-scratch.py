#Logistic_Regression From Scratch

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



#Calculate Hypothesis using Sigmoid Function
def sigmoid(z):

	sig = 1/(1+np.exp(-z))
	return sig



#Initializing Weight and Bias
def initialize_parameter(dim):
	
	w = np.full((1, dim), 0.1)
	b = 0.1

	return w, b



#Forward_Propogation for Hypothesis
def forward_prop(datax, w, b):

	z = np.dot(datax, w.T)+b
	A = sigmoid(z)

	return A



#Calculating Cost for our Model
def compute_cost(datay, a):

	J = -np.sum(np.dot(datay.T, np.log(np.abs(a))) + np.dot((1-datay).T, np.log(np.abs(1-a))))/400.0

	return J



#Calculate Derivative for Weight and Bias
def back_prop(datax, datay, a):

	dw = np.dot((a-datay).T, datax)/len(datay)
	db = np.sum(a-datay)/len(datay)

	return dw, db



#Update Weight and Bias
def gradient_update(w, b, dw, db, lr_rate):

	w = w - lr_rate*dw
	b = b - lr_rate*db

	return w, b



#Prediction for testing data
def predict(test, w, b):

	pred = forward_prop(test, w, b)

	pred[pred>0.5] = int(1)
	pred[pred<=0.5] = int(0)

	return pred



#Calculate Accuracy of Model
def accuracy(predictions, y):

	acc = np.sum(predictions == y)/predictions.shape[0]
	return acc




#Main Logistic_Regression function
def Logistic_Regression():

	#Collecting Data
	data = pd.read_csv("FilePath")

	
	#Convert Gender {male : 1, female : 0}
	data['Gender_C'] = pd.get_dummies(data['Gender'], drop_first=True)

	
	#Collect Features and Scalling them
	x = data[['Gender_C', 'Age', 'EstimatedSalary']]

	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)


	#Collecting Target Valriable {purchased : 1, not-purchasd : 0}
	y = data[['Purchased']]

	
	#Separate data into training and testing data
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

	x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1])
	y_train = np.array(y_train).reshape(y_train.shape[0], y_train.shape[1])

	x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1])
	y_test = np.array(y_test).reshape(y_test.shape[0], y_test.shape[1])


	#Initializing Wight, Bias, Learning_rate, Iteration
	w, b = initialize_parameter(x_train.shape[1])
	num_iteration = 1000000
	learning_rate = 0.05


	#Cost_List
	cost_list = []


	#Run our Model
	for i in range(num_iteration):

		#(1) : Forward Propogation - Calculate Hypothesis
		A = forward_prop(x_train, w, b)

		#(2) : Calculating Cost of Model
		cost = compute_cost(y_train, A)

		#(3) : Gradent Descent - Calculate derivative of Weight and Bias
		dw, db = back_prop(x_train, y_train, A)

		#(4) : Update the Weight and Bias
		w, b = gradient_update(w, b, dw, db, learning_rate)

		#(5) : Printing Cost, Weight and Bias
		if i%100000 == 0:
			cost_list.append(cost)
			print("After iteration- {0}..cost : {1}  W : {2}  B : {3}".format(i, cost, w, b))



	#Final Cost, Weight and Bias
	print()
	print("Final Cost : ", cost_list[-1])
	print("Final Weght : " ,w)
	print("Fina Bias : ", b)
	print()


	#Calculating Accuracy
	predictions = predict(x_test, w, b)
	acc = accuracy(predictions, y_test)
	print("Accuracy : {0} {1}".format(acc*100,'%'))




if __name__=="__main__":

	Logistic_Regression()