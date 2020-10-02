#!/usr/bin/env python
# coding: utf-8

# # Deep Learning 01: Logistic Regression
# 
# This is the first notebook of hands on by following the deep learning specialization course in Couresera. Logistic regression is a fundamental unit in deep neural network. In this notebook, we will use a single unit of logistic regression to categorize real image data. The images are Gaussian filtered retina scan to detect diabetic retinopathy. In the end we will see that with a simple logistic regression, we can detect the diabetic retinopathy on the images with an accuracy around 85%.

# In[ ]:


# Initialize the notebook
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(17)


# ## 1) Data pre-processing
# 
# The images are 224x224 pixels RGB images. We load the images into 224x224x3 numpy arrays. For simple classification, we will classify only "No_DR" (no diabetic retinopathy) and others. We label "No_DR" is 0 and the others are 1.

# In[ ]:


# Load images and create label vector
root_path = "../input/diabetic-retinopathy-224x224-gaussian-filtered/"
images_path = root_path + "gaussian_filtered_images/gaussian_filtered_images/"
images = [] # images
Y = [] # label vector

dir_names = [dir_name for dir_name in os.listdir(images_path) if os.path.isdir(images_path + dir_name)]
for dir_name in dir_names:
    if dir_name == "No_DR": # Create a label
        y = 0 
    else:
        y = 1
    for file_name in os.listdir(images_path + dir_name):
        images.append(plt.imread(images_path + dir_name + "/" + file_name))
        Y.append(y)
    print("Finish reading " + dir_name + " condition images.")
    print("Total image read: " + str(len(images)))
print("End of the data")
Y = np.reshape(np.array(Y), (1, -1)) # Make Y to be a row vector


# Let us see some detail and some images that we loaded.

# In[ ]:


# Print some detail
print("Size of each image: " + str(images[0].shape))
print("Size of labels: " + str(Y.shape))
print("Number of DR (y = 1): " + str(np.sum(Y)))
print("Number of No_DR (y = 0): " + str(Y.shape[1] - np.sum(Y)))

# Randomly show images
n = 4 # show 4 images
fig, axes = plt.subplots(nrows = 1, ncols = n, figsize=(16, 16))
rand_idx = np.random.randint(0, len(images), n)
for i in range(n):
    axes[i].imshow(images[rand_idx[i]])
    axes[i].title.set_text("Image #" + str(rand_idx[i]) + "\n y label = " + str(Y[0][rand_idx[i]]))
    axes[i].axis("off")
fig.show()


# For simplicity and less time consuming, we will use grayscale images intead of RGB images for classification. We turn the RGB images to grayscale images and therefore each image is 224x224 2D array.

# In[ ]:


# Function to convert RGB to grayscale
def rgb_to_gray(image):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(image, rgb_weights)

gray_images = list(map(rgb_to_gray, images)) # Convert RGB images to grayscale images

# Print the image size
print("Size of each grayscale image: " + str(gray_images[0].shape))

# Show grayscale images
fig, axes = plt.subplots(nrows = 1, ncols = n, figsize=(16, 16))
for i in range(n):
    axes[i].imshow(gray_images[rand_idx[i]], cmap='gray')
    axes[i].title.set_text("Image #" + str(rand_idx[i]) + "\n y label = " + str(Y[0][rand_idx[i]]))
    axes[i].axis("off")
fig.show()


# We will use all pixels as features for one sample. Therefore each sample will have 224x224 = 50176 features. For convinience, we flatten the features into one column for each sample and stack all samples together horizontally. Let us denote $n_x$ as a number of features and $m$ as a number of samples. Then the dataset $X$ and the label $Y$ can be written as
# 
# $X = \begin{bmatrix}
# x_1^{(1)} & x_1^{(2)} & x_1^{(3)} & \cdots & x_1^{(m)} \\
# x_2^{(1)} & x_2^{(2)} & x_2^{(3)} & \cdots & x_2^{(m)} \\
# x_3^{(1)} & x_3^{(2)} & x_3^{(3)} & \cdots & x_3^{(m)} \\
# \vdots & \vdots & \vdots & & \vdots \\
# x_{n_x}^{(1)} & x_{n_x}^{(2)} & x_{n_x}^{(3)} &  & x_{n_x}^{(m)}
# \end{bmatrix},
# \quad \quad
# Y = \begin{bmatrix}
# y^{(1)} & y^{(2)} & y^{(3)} & \cdots & y^{(m)}
# \end{bmatrix} $
# 
# where the superscript $(i)$ means the $i^{th}$ sample and the lowerscript $j$ means the $j^{th}$ feature.

# In[ ]:


# Prepare dataset matrix X
X = np.array(gray_images).reshape(len(gray_images), -1).T # flatten and reshape
print("Size of data set (n_x, m): " + str(X.shape))
print("Size of label (1, m): " + str(Y.shape))


# Now our data are ready for logistic regression.

# ## 2) Logistic regression
# 
# Logistic regression is a basic unit in deep nerual network. Let us start with the first two steps:
# * linear transformation with parameters called weight ($w$) and bias ($b$)
# * Calculate activation function, for a simple logistic regression, we will use a sigmoid function as an activation function.

# ### 2.1) Linear transformation
# 
# The first step of doing logistic regression is doing a linear transformation for each sample using a weight vector ($w$) and a bias scalar($b$),
# 
# $z^{(i)} = w^T x^{(i)} + b$
# 
# where the weight vector is $(n_x, 1)$ column vector. The vectorization form of the linear transformation can be expressed easily as
# 
# $Z = w^T X + b$
# 
# where we utilize the broadcasting feature of Python for $b$ here.
# 
# The weight and the bias are the parameters we have to find so that it can predict the output as well as possible. To begin with, we have to initiate the weight vector and the bias. There is a few ways to initiate the parameters. Here, the weight vector and the bias are initiated to be zero.

# In[ ]:


# Initialize w and b function
def initialize_parameters(n_x):
    w = np.zeros((n_x, 1))
    b = 0
    return w, b


# ### 2.2) Sigmoid function
# 
# For a simple logistic regression, we will use a sigmoid function as an activation function. Sigmoid function is defined as
# 
# $ f(z) = \displaystyle\frac{1}{1 + e^{-z}} $

# In[ ]:


# Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))


# Here is a plot of the sigmoid function

# In[ ]:


# Plot of sigmoid function
z = np.linspace(-4, 4, 101)
plt.figure(figsize = (5, 3))
_ = plt.plot(z, sigmoid(z))
plt.title("Sigmoid(z)")
plt.xlabel("z")
plt.grid()
plt.show()


# In logistic regression, the input of the sigmoid function is the linear transformation of the features and the output is a probability that the label will be 1.

# ### 2.3) Forward and backward propagation
# 
# To find appropriate parameters $w$ and $b$, we have to minimize a cost function,
# 
# $J = -\displaystyle\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
# 
# where $y$ is a true label from the training set and $a$ is a value of the activation function. This cost function comes from that we want to maximize likelihood estimation.
# 
# To minimize the cost function, we use a common algorithm called "gradient decent" which an update rule can be described as
# 
# $ w = w - \alpha \displaystyle \frac{\partial J}{\partial w}$
# 
# $ b = b - \alpha \displaystyle \frac{\partial J}{\partial b}$
# 
# where $\alpha$ is a learning rate. We can run the altorithm as many iterations as we want to minimize the cost function as much as possible. The partial derivatives for logistic regression can be derived as
# 
# $\displaystyle \frac{\partial J}{\partial w} = \displaystyle \frac{1}{m}X(A-Y)^T$
# 
# $\displaystyle \frac{\partial J}{\partial b} = \displaystyle \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$
# 
# where $A$ is a row vector of the activation values. Derivation of the partial derivatives can be found in the last section of the notebook.
# 
# Overall, the algorithm can be implemented as follow.
# 1. For a given weight $w$, bias $b$ and dataset $X$, calculate the prediction value. This is a forward propagation.
# 2. Calculate the cost function
# 3. Calculate the derivatives. This is a backward propagation.
# 4. Update the weight and bias.
# 
# We can repeat step one to four as much as we like and check that how the cost function evolves. The cost function should decrease with more iterations.
# 
# For convenience, we implement step one to three as a function called "propagate" and another function to run the step one to four iteratively called "logistic_regression".

# In[ ]:


# Forward and backward propagation
def propagate(X, Y, w, b):
    m = Y.shape[1]
    grads = {}
    A = sigmoid(w.T @ X + b) # forward propagation
    cost = -1/m * (np.sum (Y * np.log(A)) + np.sum((1 - Y) * np.log(1 - A))) # cost function
    
    # backward propagation
    grads['dw'] = 1/m * X @ (A - Y).T
    grads['db'] = 1/m * np.sum(A - Y, axis = 1, keepdims = True)
    
    return A, grads, cost   

# Logistic regression function
def logistic_regression(X, Y, learning_rate = 0.0006, num_iter = 200, print_cost = True):
    w, b = initialize_parameters(X.shape[0]) # initailize the parameters
    costs = []
    
    # logistic regression
    for i in range(num_iter):
        A, grads, cost = propagate(X, Y, w, b)
        if print_cost and i % 20 == 0:
            print("Iteration #" + str(i) + "\tCost value = " + str(cost))
        costs.append(cost)
        w -= learning_rate * grads['dw']
        b -= learning_rate * grads['db']
    
    # compute the cost of the final parameter
    A, grads, cost = propagate(X, Y, w, b)
    print("Final cost value = " + str(cost))
    costs.append(cost)
    
    return w, b, costs


# ## 3) Train the model
# Now we are ready. let us split the dataset into train/test set to be 75:25 roughly.

# In[ ]:


# Split the data to train/test set
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size = 0.25, random_state = 5)
X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T

# Print some detail
m = Y.shape[1]
m_train, m_test = Y_train.shape[1], Y_test.shape[1]
y1_train, y1_test = np.sum(Y_train), np.sum(Y_test)
print("number of train samples: " + str(m_train) + "(" + "{0:.2f}".format(m_train/m*100) + "%)")
print("number of DR cases in train samples: " + str(y1_train) + "(" + "{0:.2f}".format(y1_train/m_train*100) + "%)")
print("number of test samples: " + str(m_test) + "(" + "{0:.2f}".format(m_test/m*100) + "%)")
print("number of DR cases in test samples: " + str(y1_test) + "(" + "{0:.2f}".format(y1_test/m_test*100) + "%)")


# We train the model with 200 iterations (with the train dataset) and learning rate = 0.0006.

# In[ ]:


w, b = initialize_parameters(X.shape[0]) # initailize the parameters
w, b, costs = logistic_regression(X_train, Y_train, learning_rate = 0.0006, num_iter = 200)
_ = plt.plot(costs)
plt.xlabel("number of iterations")
plt.ylabel("Cost value")
plt.grid()
plt.show()


# We got the learnt parameters. Let us use the parameters to predict the output. If the output is greater than 0.5, it will predict 1 (DR) and vise versa. The accuracies are calculated on both training dataset and test set.

# In[ ]:


# Predict function
def predict(X, Y, w, b):
    A, _, _ = propagate(X, Y, w, b)
    A[A >= 0.5] = 1
    A[A < 0.5] = 0
    diff = np.abs(A - Y)
    acc = 1 - np.sum(diff)/diff.shape[1]
    return A, diff, acc

yhat_train, _, acc_train = predict(X_train, Y_train, w, b) # Accuracy on train set
print("Accuracy on train set: " + "{0:.2f}".format(acc_train*100) + "%")

yhat_test, _, acc_test = predict(X_test, Y_test, w, b) # Accuracy on test set
print("Accuracy on test set: " + "{0:.2f}".format(acc_test*100) + "%")


# As we can see, we get the model for diabetic retinopathy detection using Gaussian filtered retina images with accuracy around 85%

# ## 4 Derivatives derivation
# 
# In this section, we will roughly show how the derivatives below in back propagation can be derived. Let us introduce a lost function as
# 
# $ \mathcal{L}^{(i)} = y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)}) $.
# 
# 
# Recall that $a^{(i)}$ is a sigmoid function of $z^{(i)}$ which the derivative can be found as
# 
# 
# $ \displaystyle \frac{\partial a^{(i)}}{\partial z^{(i)}} = \frac{e^{-z^{(i)}}}{(1 + e^{-z^{(i)}})^2} = \frac{1 + e^{-z^{(i)}} - 1}{(1 + e^{-z^{(i)}})^2} = (1/a^{(i)} - 1)\cdot(a^{(i)})^2 = a^{(i)}(1 - a^{(i)})$.
# 
# We also know that $z^{(i)}$ is a linear transfrom of the input features. The derivatives respect to the parameters are
# 
# $ \displaystyle \frac{\partial z^{(i)}}{\partial b} = 1$
# 
# $ \displaystyle \frac{\partial z^{(i)}}{\partial w_j} = x_j^{(i)}$
# 
# ### 4.1) Proof of the derivative respect to $b$
# 
# Let us compute
# 
# $\displaystyle \frac{\partial \mathcal{L}^{(i)}}{\partial b} = \frac{\partial \mathcal{L}^{(i)}}{\partial a^{(i)}} \cdot \frac{\partial a^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial b}= \left( \frac{y^{(i)}}{a^{(i)}} - \frac{1-y^{(i)}}{1-a^{(i)}}\right) \cdot a^{(i)}(1 - a^{(i)}) \cdot 1 = y^{(i)} - a^{(i)}$
# 
# and therefore,
# 
# $\displaystyle \frac{\partial J}{\partial b} = -\displaystyle\frac{1}{m}\sum_{i=1}^{m}\frac{\partial \mathcal{L}^{(i)}}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})$
# 
# 
# ### 4.2) Proof of the derivative repect to $w$
# 
# Let us compute
# 
# $\displaystyle \frac{\partial \mathcal{L}^{(i)}}{\partial w_j} = \frac{\partial \mathcal{L}^{(i)}}{\partial a^{(i)}} \cdot \frac{\partial a^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}= \left( \frac{y^{(i)}}{a^{(i)}} - \frac{1-y^{(i)}}{1-a^{(i)}}\right) \cdot a^{(i)}(1 - a^{(i)}) \cdot x_j^{(i)} = x_j^{(i)}(y^{(i)} - a^{(i)})$
# 
# and 
# 
# $\displaystyle \frac{\partial J}{\partial w_j} = -\displaystyle\frac{1}{m}\sum_{i=1}^{m}\frac{\partial \mathcal{L}^{(i)}}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m x_j^{(i)}(a^{(i)}-y^{(i)})$.
# 
# To get the vectorization of the derivative, we can show that
# 
# $\displaystyle \frac{\partial J}{\partial w} = 
# \begin{bmatrix}
# \partial J/\partial w_1 \\
# \partial J/\partial w_2 \\
# \partial J/\partial w_3 \\
# \vdots \\
# \partial J/\partial w_{n_x}
# \end{bmatrix} = \displaystyle \frac{1}{m}
# \begin{bmatrix}
# \sum_{i=1}^m x_1^{(i)} (a^{(i)}-y^{(i)})\\
# \sum_{i=1}^m x_2^{(i)} (a^{(i)}-y^{(i)}) \\
# \sum_{i=1}^m x_3^{(i)} (a^{(i)}-y^{(i)}) \\
# \vdots \\
# \sum_{i=1}^m x_{n_x}^{(i)} (a^{(i)}-y^{(i)})
# \end{bmatrix} = \displaystyle \frac{1}{m} X(A - Y)^T$
