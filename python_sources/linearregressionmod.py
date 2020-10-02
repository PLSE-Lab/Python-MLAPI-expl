import numpy as np
import matplotlib.pyplot as plt

def compute_error_for_given_points(b, m, points):
	# It compute how bad our line is also known as LOSS.
	# We compute it as Least Square Sum (Recidual)
	# Calculating Varience
	# There is no error at begning
	totalError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y- (m * x + b))**2
	return totalError/ float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
	#gradient descent
	#Form Upward Concave Bell Curve
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gradient += -(y - ( m_current * x + b_current))
		m_gradient += - x * ( y - ( m_current * x + b_current) )
	b_gradient *= (2/N)	
	m_gradient *= (2/N)
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b,new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iteration):
	b = starting_b
	m = starting_m
	prev_err = 0.0
	curr_err = compute_error_for_given_points(b, m, points)
	while curr_err!=prev_err:
		prev_err = curr_err
		b, m = step_gradient(b, m, np.array(points), learning_rate)
		curr_err = compute_error_for_given_points(b, m, points)
		print(curr_err,b,m)
	return [b,m]	


def run():
	points = np.genfromtxt('/kaggle/input/lrdata.csv', delimiter=',')
	#hyperparameters
	#L_R : How fast model learn , if too low then it is slow to converge and if high then never converge
	learning_rate = 0.0001 
	# y = mx + b (slope formula)
	# start with 0 
	initial_b = 0
	initial_m = 0
	# Number of iteration we run for training.
	num_iteration = 1000 # 1000 becuase of small dataset
	# Optimal value 
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iteration)
	print(b)
	print(m)
	plt.plot(points[:,0],points[:,1],'bo')
	plt.plot(points[:,0],(m*points[:,0]+b),color='r')
	plt.xlabel('Hours')
	plt.ylabel('Marks')
	plt.show()

if __name__ == '__main__':
	run()