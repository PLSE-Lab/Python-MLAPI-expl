#!/usr/bin/env python
# coding: utf-8

# # A Reservoir Computing Approach to Generalized Learning
# Implementing the experiment from "*Similarity Learning and Generalization with Limited Data: A Reservoir Computing Approach*", Krishnagopal, Aloimonos, and Girvan (https://www.researchgate.net/publication/328691984_Similarity_Learning_and_Generalization_with_Limited_Data_A_Reservoir_Computing_Approach)
# 
# In the reservoir computing model, a "reservoir" of neurons is recurrently connected by a (sparse) set of randomly generated weights. The reservoir weights are fixed at the time of their creation. From the paper abstract, "... the reservoir acts as a nonlinear filter projecting the input into a higher dimensional space in which the relationships are separable." Because reservoir computing networks do not use back-propogation learning, the neuron activation function need not be differentiable. In the Liquid State Machine implementation, 3rd generation spiking neurons are commonly used. However, in the Echo State Network reservoir implementation used in this experiment, the neurons are continuous output perceptrons.
# 
# The nonlinear projected space is linearly separable, evoking superficial similarities to Support Vector Machines, and the "readout" weight matrix is directly computed using Ridge Regression.
# 
# In this experiment an Echo State Network attempts to classify the relationship between two images. The MNIST digit data set is used and various transformations are applied to produce versions of the digits. A digit image along with one of the transformed versions is applied to the network and the network is taught to recognize the relationship between the two images. 
# 
# The experiment is remarkable on two counts. First, the network learns the transformation types using only half the randomly chosen digits, for instance using only digits 1,3,4,6, and 9, but performs as well when tested using the other half of the digits, 0,2,5,7, and 8 which the network never saw during training. This demonstrates that the network is learning the relationships between the images and is not merely memorizing pairs of images. By way of contrast, a deep Siamese network similarly trained performs as well when tested with digits it is trained with, but poorly when tested with the other digits. Second, the network requires relatively little training data, just 250 examples in this experiment.
# 
# The network is implemented in pure python using just Numpy and takes less than 30 seconds to run.

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd # only used to read the MNIST data set
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


# ### Program Constants
# 
# There are five different comparison types. Four of them are transformations of different instances of the same digit. The fifth is a comparison of one digit with an instance of a different digit.
# 

# In[ ]:


IMG_ROWS, IMG_COLS = 28,28    # number of rows and columns in mnist images

IMG_PAIR_LABELS = ['similar', 'rot90', 'zoom', 'blur', 'different']    # image pair comparison labels
N_IMG_PAIR_LABELS = len(IMG_PAIR_LABELS)                               # number of comparison types
IMG_PAIR_LABEL_ONE_HOT = np.eye(N_IMG_PAIR_LABELS, dtype=np.float)     # array of one-hot label encodings


# In[ ]:


N_TEACH_IMG_PAIRS = 50    # number of image pairs per comparison type to teach with. Total teaching comparision = 5*50 = 250
N_TEST_IMG_PAIRS = 200    # number of image pairs per comparison type to test with. Total testing comparisons = 5*200 = 1000


# We choose 5 random digits to teach the network and use the other 5 digits to test with.

# In[ ]:


np.random.random(42)
ALL_DIGITS = np.arange(10)
TEACH_DIGITS = np.random.choice(ALL_DIGITS, size=5, replace=False)
TEST_DIGITS = np.setdiff1d(ALL_DIGITS, TEACH_DIGITS)

print('teach digits:', TEACH_DIGITS)
print('test  digits:', TEST_DIGITS)


# ### create_weights
# Input and reservoir weight matrices are populated with random values and never changed. The weight matices may be (usually are) sparse. It is important that **reservoir** weights demonstrate the "echo state property", which means that the reservoir is stable under all input conditions, that the reservoir transient response dies out. Stability is guaranteed when the reservoir weight matrix **spectral radius** is less than 1.0. The matrix spectral radius is defined as the maximum matrix eigenvalue.
# 
# This method creates weight matrices of arbitrary shape with optional sparsity and spectral radius. (NOTE - reservoir weight matrices are, by definition, square matrices.)

# In[ ]:


def create_weights(shape, low=-1.0, high=1.0, sparsity=None, spectral_radius=None):
    w = (high - low)*np.random.ranf(shape[0]*shape[1]).reshape(shape) + low      # create the weight matrix
    if not sparsity is None:                                                     # if sparsity is defined
        s = np.random.ranf(shape[0]*shape[1]).reshape(shape) < (1.0 - sparsity)  #     create a sparse boolean matrix
        w *= s                                                                   #     set weight matrix values to 0.0
    if not spectral_radius is None:                                              # if spectral radius is defined
        sp = np.max(np.abs(np.linalg.eig(w)[0]))                                 #     compute current spectral radius
        w *= (spectral_radius)/sp                                                #     adjust weight matrix to acheive specified spectral radius
    return w


# ### operate_reservoir
# Given two images and the input and reservoir weights, we operate the reservoir with the following steps:
# * initialize the reservoir to all zeros
# * for each column in the images
#     * form the input vector
#         * augment each column by 1 to include the bias value
#         * transform (dot product) the augmented columns from each image using the same input weights
#         * concatenate the transformed columns to form the input vector
#     * update the reservoir using the reservoir weights
#     * add the input vector to the transformed reservoir
#     * form the reservoir state variable by collecting interim reservoir values

# In[ ]:


def operate_reservoir(w_u, w_x, img_1, img_2):
    n_cols = img_1.shape[1]    # number of image columns = number teaching steps
    x_size = w_x.shape[0]      # reservoir size
    u_size = w_u.shape[1]      # input vector size

    x = np.zeros((x_size,1))    # start with zero reservoir value for each image pair
    for col in range(n_cols):   # step through image columns
        img_col_1 = img_1[:,col].reshape((u_size-1,1))
        img_col_2 = img_2[:,col].reshape((u_size-1,1))
        #
        # this technique is somewhat different than what is described in either paper
        #
        u_1 = np.dot(w_u, np.vstack((1, img_col_1)))    # stack up bias (=1) for img_1 column and transform by input weights (1/2 reservoir size)
        u_2 = np.dot(w_u, np.vstack((1, img_col_2)))    # stack up bias (=1) for img_2 column and transform by input weights (1/2 reservoir size)
        u_total = np.concatenate((u_1, u_2))            # join the transformed inputs to get to 100% reservoir size
        x = np.tanh(u_total + np.dot(w_x, x))           # update the reservoir
        #
        # collect reservoir to reservoir state
        #
        if col == 0:                              # if this is first column
            x_state = (x)                         #     initialize state
        else:                                     # subsequently
            x_state = np.hstack((x_state, (x)))   #     append latest reservoir to state

    return x_state


# ### display_reservoir_damping
# This method displays the transient behavior of the reservoir by initializing the reservoir to random values and operating the reservoir over a number of time steps. Display the values of 20 randomly chosen reservoir neurons.

# In[ ]:


def display_reservoir_damping(w, n_steps):
    x_size = w.shape[0]
    x = (2.0*np.random.ranf(x_size) - 1.0).reshape(x_size,1)    # initialize reservoir to random values
    for i in range(n_steps):                                    # for a given number of steps
        x = np.tanh((np.dot(w,x)))                              #     operate the reservoie
        if i == 0:                                              #     and collect the interim states
            M = x
        else:
            M = np.hstack((M,x))

    n_plots = min(M.shape[0], 20)                               # format the plot area
    rows = int(np.floor(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots/rows))
    x_size = M.shape[0]
    if n_plots > 20:                                            # if reservoir size is greater than 20
        cells = np.random.choice(np.arange(x_size), 20)         #    select 20 random neurons for display
    else:
        cells = np.arange(x_size)

    n_steps = M.shape[1]
    x_axis = np.arange(n_steps, dtype=np.int)
    fig, _axs = plt.subplots(nrows=rows, ncols=cols)
    for p in range(n_plots):
        idx = cells[p] 
        row = p//cols
        col = p % cols
        _axs[row][col].plot(x_axis, M[idx, :])
    fig.tight_layout()
    plt.show()


# ### Choose an Image Pair for Teaching/Testing
# In the course of teaching and testing, two images are repeatedly chosen at random. This method chooses two images at random for comparison using the image pair label as a guide for what transformation (if any) to apply.

# In[ ]:


def choose_img_pair(digits, img_pair_label):
    digit_1 = np.random.choice(digits)                     #     pick a random base digit
    digit_idx_1 = np.random.choice(len(images[digit_1]))   #     pick a random instance of the base digit
    img_1 = images[digit_1][digit_idx_1, :, :]             #     get the base digit image
    if img_pair_label == 'different':
        # choose a digit different from digit_1
        digit_2 = np.random.choice([digit for digit in ALL_DIGITS if digit != digit_1])
        digit_idx_2 = np.random.choice(len(images[digit_2]))             # pick a random instance of the comparison digit
        img_2 = images[digit_2][digit_idx_2, :, :]                       # get the comparison image
    else:
        digit_idx_2 = np.random.choice(len(images[digit_1]))    # pick a random instance of the base digit
        img_2 = images[digit_1][digit_idx_2, :, :]              # get the comparison digit
        if img_pair_label == 'similar':
            screen_1 = np.random.rand(IMG_ROWS, IMG_COLS)           # create a random screen to choose which pixels to change
            screen_2 = np.random.rand(IMG_ROWS, IMG_COLS)           # create a random screen of values to change the selected pixels to
            img_2[screen_1 > 0.80] = screen_2[screen_1 > 0.80]*0.3  # add random noise at random locations
        elif img_pair_label == 'rot90':
            img_2 = np.fliplr(img_2.T)                              # rotate the image 90 degrees
        elif img_pair_label == 'zoom':
            img_2 = img_2.repeat(2,axis=0).repeat(2,axis=1)[IMG_ROWS//2:IMG_ROWS//2 + IMG_ROWS, IMG_COLS//2:IMG_COLS//2 + IMG_COLS]
        elif img_pair_label == 'blur':                              # blur the image using the convolution kernel
            img_2 = convolve2d(img_2, KERNEL, boundary='wrap', mode='same')
        else:
            raise ValueError('illegal image pair label')
    return img_1, img_2


# ### Data Preparation
# Now that we're done with the preliminaries, code execution begins here with data perparation. For this network, we don't need separate teaching and testing data sets. We read in the MNIST train data and create two numpy arrays, one for labels and one for images, then create a dictionary where the keys are the digits 0-9 and the values are arrays of 28x28 pixel images. The image data is scaled to values in the range [0.0 .. 1.0]

# In[ ]:


df = pd.read_csv('../input/train.csv', sep=',')
lbl = df['label'].values
img = df.loc[:, df.columns != 'label'].values
images = {}
for digit in range(10):
    images[digit] = np.array([((image - np.min(image))/np.ptp(image)).reshape((28,28)) for image in img[lbl == digit]], dtype=np.float)


# ### Display Image Transformation Examples (optional)
# The following code block may be executed to display examples of the image transformations used in the experiment.

# In[ ]:


#
# these are examples of the image transformations - this code box may be safely skipped
#
img_base = images[3][0]
img_noisy = images[3][1]                                     # create the 'similar' (noisy) version
screen_1 = np.random.rand(IMG_ROWS, IMG_COLS)                #     create a random screen to choose which pixels to change
screen_2 = np.random.rand(IMG_ROWS, IMG_COLS)                #     create a random screen of values to change the selected pixels to
img_noisy[screen_1 > 0.80] = screen_2[screen_1 > 0.80]*0.3   #     add the random noise in at random locations
img_rot90 = images[3][2]                                     # create the 'rot90' version
img_rot90 = np.fliplr(img_rot90.T)                           #     rotate the image 90 degrees
img_zoom = images[3][3]                                      # create the zoomed version 
img_zoom = img_zoom.repeat(2,axis=0).repeat(2,axis=1)[IMG_ROWS//2:IMG_ROWS//2 + IMG_ROWS, IMG_COLS//2:IMG_COLS//2 + IMG_COLS]
img_blur = images[3][4]                                      # create the blurred version
img_blur = convolve2d(img_blur, np.ones((6, 6), dtype="float") * (1.0 / 36.0), boundary='wrap', mode='same')
img_different = images[5][0]

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6)
ax1.imshow(img_base, cmap='gray')
ax1.set_title('Base')

ax2.imshow(img_noisy, cmap='gray')
ax2.set_title('Similar')

ax3.imshow(img_rot90, cmap='gray')
ax3.set_title('Rotated')

ax4.imshow(img_zoom, cmap='gray')
ax4.set_title('Zoomed')

ax5.imshow(img_blur, cmap='gray')
ax5.set_title('Blurred')

ax6.imshow(img_different, cmap='gray')
ax6.set_title('Different')

plt.tight_layout()
plt.show()


# ### Initialize Some Experiment Paramters
# RC networks are inherently recurrent and applicable to time- or sequence-series data. In this experiment we "serialize" the images by feeding them to the reservoir one column at a time. The input vector size is therefore the number of rows in the image, plus one for a bias.
# 
# We set the reservoir size (number of reservoir neurons) to be 1000.
# 
# The output vector size is the size of the one-hot representation of the image comparison class label.

# In[ ]:


U_SIZE = IMG_ROWS + 1       # image is input column-wise, input vector is size of row (+1 for bias)
X_SIZE = 1000               # reservoir size
O_SIZE = N_IMG_PAIR_LABELS  # image pair one-hot label size


# The reservoir connection matrix is normally sparse - 90% in this case. A spectral radius < 1.0 guarantees reservoir stability. 

# In[ ]:


SPARSITY = 0.9          # sparsity of reservoir node connections
SPECTRAL_RADIUS = 0.5   # determines reservoir echo state property


# ### Create and Initialize Weight Matrices
# Create the reservoir weight matrix with specified sparsity and spectral radius. Create the input weight matrix with full connectivity and no spectral radius constraints. The input weight matrix is not recurrent and so does not have a stability requirement. The same input weight matrix is used to transform the columns of both images being compared. The output of the input  weight matrix is only half the size of the reservoir, however the outputs of both image column transformations are concatenated to form a single input vector the same size as the reservoir which allows it to be added to the reservoir state during reservoir operation.
# 
# Optionally display the reservoir transient behavior.

# In[ ]:


w_x = create_weights(shape=(X_SIZE, X_SIZE), low=-1.0, high=1.0, sparsity=SPARSITY, spectral_radius=SPECTRAL_RADIUS)    # 1000 by 1000
w_u = create_weights(shape=(X_SIZE//2, U_SIZE), low=-1.0, high=1.0)                                                     # 500 x 29
display_reservoir_damping(w_x, 25)


# Create the "blur" convolution kernal.

# In[ ]:


KERNEL_SIZE = 6
KERNEL = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype="float") * (1.0 / (KERNEL_SIZE * KERNEL_SIZE))


# ### Teach
# Teaching the reservoir is done with the following steps
# * iterate through each of the 5 comparison types, 'similar', 'rot90', 'zoom', 'blur', and 'different'
#     * for the specified number of examples
#         * select a base and comparison image instance (use a different digit for the 'different' comparison type, a different instance of the same digit for all others)
#         * transform the comparison image using the current comparison type (except for the 'different' comparison type)
#         * operate the reservoir for the image pair
#         * accumulate the resulting reservoir state
#         * accumulate the corresponding comparison type (label) state
# 

# In[ ]:


for img_pair_label in IMG_PAIR_LABELS:                         # for each comparison type
    img_pair_label_idx = IMG_PAIR_LABELS.index(img_pair_label)
    for img_pair_teach in range(N_TEACH_IMG_PAIRS):            # for the number of teaching examples
        img_1, img_2 = choose_img_pair(TEACH_DIGITS, img_pair_label)
        x_state = operate_reservoir(w_u=w_u, w_x=w_x, img_1=img_1, img_2=img_2)  # x_state 1000 x 28, y_state 5 x 28
        #
        # create label state for image pair
        #
        y_state = np.repeat(IMG_PAIR_LABEL_ONE_HOT[img_pair_label_idx], repeats=IMG_COLS).reshape(N_IMG_PAIR_LABELS, IMG_COLS)
        #
        # append the current (x_size by n_cols) reservoir state and (n_labels by n_cols) label state to total teaching matricies
        #
        if img_pair_label_idx == 0 and img_pair_teach == 0:  # if this is first time through the loop
            X = x_state                                      #     init the reservoir state
            Y = y_state                                      #     and output state
        else:                                                # subsequently
            X = np.hstack((X, x_state))                      #     accumulate reservoir state by stacking
            Y = np.hstack((Y, y_state))                      #     accumulate corresponding labels by stacking


# ### Compute Readout Weight Matrix
# Use Ridge Regression to compute the readout weight matrix matching collected reservoir state to collected label state. Ridge regression is used to add some regularization to situations where data points tend to be co-linear. 
# 
# \begin{equation}
# W_{out} = YX^T(XX^T + \gamma I)^{-1}
# \end{equation}

# In[ ]:


err = 1e-8
w_out = np.dot(np.dot(Y, X.T), np.linalg.inv(np.dot(X, X.T) + err*np.eye(X_SIZE)))


# ### Test
# Test the network using digits not used in the teaching phase. This demonstrates that the network is learning the relationship between the images and not just memorizing image pairs.
# 
# Operate the reservoir as before in the teaching phase using the number of examples specified for testing. Instead of collecting label state, compute the label state using the readout weight matrix. The label is computed as each image column is applied to the reservoir. The collected labels for each image pair are then normalized by taking the mean of each label value and dividing by the sum of the means. The label is then selected as the index of maxium label value.
# 
# Collect the results into two arrays representing the true and predicted labels.

# In[ ]:


y_true = []
y_pred = []
for img_pair_label in IMG_PAIR_LABELS:                              # for each comparison type
    img_pair_label_idx = IMG_PAIR_LABELS.index(img_pair_label)
    for img_pair_test in range(N_TEST_IMG_PAIRS):                   # for each testing example
        img_1, img_2 = choose_img_pair(TEST_DIGITS, img_pair_label)
        x_state = operate_reservoir(w_u=w_u, w_x=w_x, img_1=img_1, img_2=img_2)  # x_state 1000 x 28, y_state 5 x 28
        #
        # predict label
        #
        y_state = np.tanh(np.dot(w_out, x_state))   # compute the label values for all image columns
        y_mean = y_state.mean(axis=1)               # compute the mean of each value in the label
        y_norm = y_mean/sum(y_mean)                 # normalize by dividing by the sum of the means
        pair_label_pred = np.argmax(y_norm)         # the predicted label is the index of maxium value

        y_true.append(img_pair_label_idx)
        y_pred.append(pair_label_pred)


# ### Results
# Compute and display the **precision**, **recall**, and **f1_score** for each classification as well as the **aggregate f1_score**. Note that in this case, the recall scores are the same as the accuracy scores for each class.

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print()
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f_score = f1_score(y_true, y_pred, average=None)
print('label      precision\trecall\tf1_score')
print('----------------------------------------')
for img_pair_label in IMG_PAIR_LABELS:    # for image pair label
    img_pair_label_idx = IMG_PAIR_LABELS.index(img_pair_label)
    print('{:11s}{:4.4f}\t{:4.4f}\t{:4.4f}'.format(img_pair_label, precision[img_pair_label_idx], recall[img_pair_label_idx], f_score[img_pair_label_idx]))
print()
f_score_aggregate = f1_score(y_true, y_pred, average='micro')
print('f1_score aggregate: {:4.4f}'.format(f_score_aggregate))

