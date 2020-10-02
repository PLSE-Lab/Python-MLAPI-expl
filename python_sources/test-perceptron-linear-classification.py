import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
data_count = 100

real_w = np.array([1.0, 1.0])
real_b = 0.0

func = lambda x, w, b: (x*w).sum(-1) + b 
lr = 1e-1
max_iter = 20
show_img = False

def get_classify_dataset():
	x_dataset = np.random.normal(0, 1, (data_count, 2))
	y_dataset = np.where(func(x_dataset, real_w, real_b) > 0, 1, -1)
	
	return x_dataset, y_dataset

def sorting(x_dataset, y_dataset):
	sort_indices = y_dataset.argsort()

	return x_dataset[sort_indices], y_dataset[sort_indices]

def draw_line(w, b):
	x = np.array([-2, 2])
	y = - x * w[0] - b / w[1]

	plt.title('y = {} x + {}'.format(w, b))
	plt.plot(x, y, ls=':', c='k')

def main():
	w = np.random.random(2)
	b = np.random.random(1)

	x_dataset, y_dataset = get_classify_dataset()
	sort_x, sort_y = sorting(x_dataset, y_dataset)

	positive_index = np.where(sort_y > 0)[0][0]

	for _ in range(max_iter):
		for x, y in zip(x_dataset, y_dataset):
			plt.clf()
			if y * func(x, w, b) <= 0:
				w += lr * (y * x)
				b += lr * y

			if show_img:
				draw_line(w, b)
				plt.scatter(sort_x[:positive_index, 0], sort_x[:positive_index, 1], c='b')
				plt.scatter(sort_x[positive_index:, 0], sort_x[positive_index:, 1], c='r')
				plt.scatter(x[0], x[1], marker='*', c='k')
				plt.xlim(min(x_dataset[:,0]), max(x_dataset[:,0]))
				plt.ylim(min(x_dataset[:,1]), max(x_dataset[:,1]))
				plt.pause(1e-6)
			else:
				print('y = {} x + {}'.format(w, b))

if __name__ == '__main__':
	main()