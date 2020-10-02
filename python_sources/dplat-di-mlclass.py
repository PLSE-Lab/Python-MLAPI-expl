#!/usr/bin/env python
# coding: utf-8

# ## Linear Regression Examples
# 
# Note: this notebook works only with Python >= 3.6.
# 
# (Credit to https://tutorials.technology/tutorials/19-how-to-do-a-regression-with-sklearn.html)

# ### First example: simple linear regression
# 
# First, we're going to generate some pseudo-random data that is roughly linear.  Scikit-learn provides a function that generates this kind of data.

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=300, n_features=1, n_informative=1, noise=10, random_state=7)
plt.scatter(X, y, color='blue')
plt.title('Generated pseudo-data', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()


# Now, we'll split the x and y data into training sets and test sets.  We'll take out 20% of our data and reserve it for test data.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(f"Number of data points in X_train: {len(X_train)}")
print(f"Number of data points in X_test: {len(X_test)}")
print(f"Number of data points in y_train: {len(y_train)}")
print(f"Number of data points in y_test: {len(y_test)}")


# We'll plot the training data and testing data separately so you can see the distribution:

# In[ ]:


p, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 5))
ax1.scatter(X_train, y_train, color='blue')
ax1.set_title('Training data', fontsize=16)
ax2.scatter(X_test, y_test, color='blue')
ax2.set_title('Testing data', fontsize=16)
p.show()


# Now, we'll create the LinearRegression object and train it with our training data.  Note, we are not using the test data at all yet.

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)   # Note, X_test and y_test are NOT referenced here!


# Now, predict the test values from this new model, and plot the resulting line:

# In[ ]:


y_pred = lr.predict(X_test)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.title('Linear regression result', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()


# We can find out the exact parameters used for this line formula:

# In[ ]:


from IPython.display import display, Math
display(Math(r'y = {:.2f}x + {:.2f}'.format(lr.coef_[0], lr.intercept_)))


# Now let's plot the line and the test data together:

# In[ ]:


plt.scatter(X_test, y_test, color='blue')
max_x = max(X_test)
min_x = min(X_test)
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.title('Fitted linear regression', fontsize=16)
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.show()


# Not bad. Remember, the training routine **never saw** any of these test data points.
# 
# We have a formal way of quantifying the quality of predictions, by using the mean squared error and coefficient of determination (R^2):

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))


# That mean squared error looks huge, but remember the scale of the y axis we are dealing with.  The R^2 value indicates a good fit (where 1.0 is perfect).

# ### Second example: a dataset with 100 features!
# 
# So far, all of this could easily be done in Excel.  Let's try something harder! How about our example where we had 100 variables and 20,000 data points?  Let's generate some data in a similar way. Since plotting a 100-dimensional space is quite difficult, we'll plot only 9 '2-D projections' of the first 9 dimensions below.
# 
# We'll use "make_sparse_uncorrelated" to generate our data, which has the interesting property that only the first four features are useful (i.e. linear).

# In[ ]:


from sklearn.datasets import make_sparse_uncorrelated

X, y = make_sparse_uncorrelated(n_samples=20000, n_features=100, random_state=7)

fig, ax = plt.subplots(3,3, figsize = (10,10))
for i in range(3):
    for j in range(3):
        ax[i,j].set_title(f"Dimension {i*3+j+1}")
        ax[i,j].scatter(X[:,i*3+j], y, s=1)
plt.tight_layout()
plt.show()


# (Why do these look like blobs, and not lines?  Remember, we are projecting a 100-dimensional dataset onto 2 dimensions. We shouldn't expect to see a line in one of these slices.)
# 
# Now, split the data as before.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(f"Number of data points in X_train: {len(X_train)}")
print(f"Number of data points in X_test: {len(X_test)}")
print(f"Number of data points in y_train: {len(y_train)}")
print(f"Number of data points in y_test: {len(y_test)}")


# Let's train! This time we'll use a stochastic linear regression model, since we have so many data points and there's a risk that the resultant matrix would be non-invertible.

# In[ ]:


from sklearn.linear_model import SGDRegressor
lr = SGDRegressor(verbose=2)
lr.fit(X_train, y_train)   # Note, X_test and y_test are NOT referenced here!


# Now let's take a look at our MSE and R^2.

# In[ ]:


y_pred = lr.predict(X_test)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# ### Third example: Underfitting, overfitting and "just right"
# 
# This example involves generating some noisy data that approximates a sin curve, and increasing the degree of the polynomial and see how it affects the MSE.

# In[ ]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

def true_fun(X):
    return np.sin(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()


# ### Fourth example: Linear regression applied to face prediction
# 
# We can even apply linear regression, as well as other models, to pixel data.  Let's see if we can train different types of models on a selection of faces.  Given the upper half of a new face, can the model predict the bottom half?

# In[ ]:


from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV

# Load the faces datasets
data = fetch_olivetti_faces()
targets = data.target

# assemble training and test sets
data = data.images.reshape((len(data.images), -1))
train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

# Pick 5 random people from the test set to render later
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]
print("All the training faces:")

# render some sample faces
image_shape = (64, 64)
plt.figure(figsize=(10,30))
for i in range(len(train)):
    sub = plt.subplot(30, 10, i+1)
    sub.set_axis_off()
    sub.imshow(train[i].reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")
plt.tight_layout()
plt.show()


# In[ ]:


# cut the pictures in half
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2] # Upper half of the faces
y_train = train[:, n_pixels // 2:] # Lower half of the faces
X_test = test[:, :(n_pixels + 1) // 2] # Upper half of the faces
y_test = test[:, n_pixels // 2:] # Lower half of the faces
print(f"There are {len(X_train)} training values")
print(f"There are {len(X_test)} testing values")

# render some sample faces
image_shape = (32, 64)
plt.figure(figsize=(25,40))
#plt.suptitle("Sample Faces", size=16)
for i in range(len(X_test)):
    sub = plt.subplot(len(X_test)/10+1, 10, i+1)
    sub.set_axis_off()
    sub.imshow(X_test[i].reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")
plt.show()


# In[ ]:


# Define estimators
ESTIMATORS = {
    "Linear regression": LinearRegression(),
    "RidgeCV": RidgeCV(),
    "Random Forest": RandomForestRegressor(n_estimators=20, max_depth=10, max_features=10),
    "Neural network": MLPRegressor(hidden_layer_sizes=(100, 30, 10, 5)),
}

# Fit estimators
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    print(f"Now training: {name}")
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
    print(f"Mean squared error: {mean_squared_error(y_test, y_test_predict[name])}")
    print(f"R2: {r2_score(y_test, y_test_predict[name], multioutput='variance_weighted')}")
    print()

# Plot the completed faces
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)
for i in range(n_faces):
    # render the true face
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")
    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")
    
    # render the generated faces
    for j, est in enumerate(ESTIMATORS):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)
        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")
plt.show()


# **Moral:** the mean squared errors and r^2 values don't necessarily tell you the whole story. Always double-check the results!
