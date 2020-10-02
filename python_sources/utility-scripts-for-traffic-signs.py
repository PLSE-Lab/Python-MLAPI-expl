"""
Related Paper:
Sichkar V. N., Kolyubin S. A. Effect of various dimension convolutional layer filters on traffic sign classification accuracy. Scientific and Technical Journal of Information Technologies, Mechanics and Optics, 2019, vol. 19, no. 3, pp. DOI: 10.17586/2226-1494-2019-19-3-546-552

Full-text available on ResearchGate here:
https://www.researchgate.net/publication/334074308_Effect_of_various_dimension_convolutional_layer_filters_on_traffic_sign_classification_accuracy

Test online with custom Traffic Sign here: https://valentynsichkar.name/traffic_signs.html
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from math import sqrt, ceil
import pickle
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.


"""
Getting Grid from Set of Images
"""
# Preparing function for ploting set of examples
# As input it will take 4D tensor and convert it to the grid
# Values will be scaled to the range [0, 255]
def convert_to_grid(x_input):
    N, H, W, C = x_input.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C)) + 255
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = x_input[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = 255.0 * (img - low) / (high - low)
                next_idx += 1
            x0 += W + 1
            x1 += W + 1
        y0 += H + 1
        y1 += H + 1

    return grid


"""
Loading dataset data2.pickle with RGB examples
"""
def load_ts():
    # Opening file for reading in binary mode
    with open('../input/traffic-signs-preprocessed/data2.pickle', 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # dictionary type

    # Making channels come at the end
    data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
    data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    # Showing loaded data from file
    for i, j in data.items():
        if i == 'labels':
            print(i + ':', len(j))
        else: 
            print(i + ':', j.shape)

    # x_train: (86989, 32, 32, 3)
    # y_train: (86989, 43)
    # x_test: (12630, 32, 32, 3)
    # y_test: (12630,)
    # x_validation: (4410, 32, 32, 3)
    # y_validation: (4410, 43)
    # labels: 43
    
    # Returning 81 examples
    return data['x_train'][:81, :, :, :]


"""
Showing some examples
"""
def visualize_ts():
    # Visualizing some examples of training data
    examples = load_ts()
    print(examples.shape)  # (81, 32, 32, 3)

    # Plotting some examples
    fig = plt.figure()
    grid = convert_to_grid(examples)
    plt.imshow(grid.astype('uint8'))
    plt.axis('off')
    plt.gcf().set_size_inches(15, 15)
    plt.title('Some examples of training data', fontsize=18)

    # Showing the plot
    plt.show()

    # Saving the plot
    fig.savefig('training_examples.png')
    plt.close()


"""
Implementing functions if main script is run
"""
if __name__== "__main__":
  visualize_ts()
