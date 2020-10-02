#!/usr/bin/env python
# coding: utf-8

# # **Careful Linear Regression for Beginners 1**

# ## Introduction
# 
# In this notebook and the following one, we will attack the house pricing dataset using nothing but linear regression. Of course, we all know that to do well in the competition, our best bet is to use some works-well-out-of-the-box boosting package and model averaging. However, there are already lots of good kernels that go over such methods. Perhaps more fundamentally, to do well in the competition, we would also likely have to wade deep into the data. Once again, there are a number of great kernels that go over this for the ames housing data set.
# 
# The reason for this kernel is then not to win the competition, or even to explore the data set in full. Instead, it is to really understand the machine learning techniques we will be applying, namely linear regression. To do this, we will take a *careful* approach (note I don't say rigorous!) and really try to understand the motivation for what we are doing at each step. 
# 
# So, the *careful* part of the title refers to the fact that this kernel will include some theory sections going over the ideas behind the methods we use. The *beginner* part refers to the fact that the techniques we will use are very simple to implement with the help of sklearn, and are approriate for those just starting out.
# 
# Unfortunately, there is a slight tension between the *careful* and *beginner* parts as the former requires us to do a bit a maths, for something very easy to execute. Because of this, I have tried to write the notebook in such as way that if you are only interested in getting started with the techniques, and aren't too interested in maths, then that's no problem. In particular, the theory sections contain a summary at the end with no maths. So, if you are only interested in learning to get started with data science then you can safely skip the theory or read the summary. Additionally, I have tried to add a fair amount of comments to the code, so even data science beginners should be able to have a go following along. Of course, if you like maths then please get stuck into the theory sections.
# 
# So, the goals of the two kernels are, on the *beginner* side:
# 
# * To learn the basics of importing data, exporting data, using sklearn to model and make predictions etc
# * To see how you can implement a simple model with reasonable performance (top 16% on leaderboard in the next notebook)
# * To give you the tools to explore (you can try editing some of the functions to do better, probably much better)
# 
# And on the *careful* side the goals are:
# 
# * To understand the fundamental probablistic ideas behind linear regression, including with regularisation (we will look at this in the next notebook, specifically ridge regression)
# * To complement the other great kernels that focus on data analysis or the use of advanced methods
# 
# In this first kernel, we will go over the basic implementation of linear regression, as well as the underlying theory. Then, in the next kernel, we will look at implementing linear regression with regularisation and, of course, discuss some of the theory behind it.

# ## Still to do
# 
# These notebook is still underdevelopment, so there is lots to do... Some things that need finishing are in this one are:
# 
# * In model 2, add more comments about filling missing values/preprocessing code generally
# * Anything else people want and I have time for (so suggestions welcome)...
# 
# If like the basic idea of the notebook and want to keep it going, please leave some comments with any errors, ideas, questions or things you would like to see (explained better...) and I'll try to add them as I go. Thanks!

# # **Model 1: Linear Regression with Living Area (0.296 on LB)**

# ### Importing The Data and Loading Useful Packages

# First, let's just import anything we want along with our data. To get started we will only work with the 'GrLivArea' feature. Beginners see the comments in the code...

# In[ ]:


### First, import some useful stuff
import numpy as np #  a package for linear algebra 
import pandas as pd #  for handling data
import matplotlib.pyplot as plt # for making plots


# In[ ]:


### Next, load our data into pandas 
path_to_training_data = '../input/train.csv' #This syntax says to go up a level (../) then down into input
path_to_test_data = '../input/test.csv'

train = pd.read_csv(path_to_training_data) # This is the training data
test = pd.read_csv(path_to_test_data) # this loads the test data


# In[ ]:


### For illustration in this intro, we will just use the 'GrLivArea' feature
### To select a feature, use the [] syntax and fill in the name of the feature
X = np.array(train['GrLivArea']) # The design matrix - a (N,D) matrix where rows are data points [here D=1]
N = X.shape[0] 
D = 1
X = np.reshape(X,(N,D)) #Keep it as a matrix, though here it is just a vector since D=1

### Extract the sale-price from the dataframe
y = np.array(train['SalePrice'])
y = np.reshape(y,(N,1))
y = np.log(y) # take the log of the sale price as our target; see "Bayesian decision theory" below

### Now create the design matrix for the test data. It is important that it is structured in the same
###        way as our training data matrix. This is no issue with only one feature...
X_test = np.array(test['GrLivArea'])
N_test = X_test.shape[0]
X_test = np.reshape(X_test,(N_test,D))


# In[ ]:


### Let's plot the living-area and log-sales-price to see what we are up against

fig1 = plt.figure(figsize=(10,6)) # create the figure object
plt.plot(X,y,'ro') # plot the X data versus y data in red circles
plt.xlabel('Living Area') # label x-axis
plt.ylabel('Log Sale Price') # label y-axis
plt.show()


# Now, with some data prepped, please prepare yourself for some theory. Or simply skip down to the summary to get on with it...

# ## Theory 1: Decision theory for Linear Regression

# ### **Bayesian Decision Theory**
# 
# In this competition, our task is to build a model to predict house sale prices, $y$, from a vector of features $\mathbf{x}$. To do this, we will use the training data set we have been given, $\mathcal{D}$, that consists of $N$ pairs of sale prices and feature vectors. 
# 
# To make predictions, we must construct a decision function $\delta(\mathbf{x})$ which takes in a feature vector and produces a sale price $y$ (more generally a decision function might produce an action to take $a$). 
# 
# Of course, we do not want to choose just any decision function, but the one that is "best" for the task in hand. Bayesian decision theory tells us that this is the function that minimises the (posterior) expected loss. 
# 
# The loss function, $L(y,\delta(\mathbf{x}))$ measures how accurate our decision function's prediction is for a given data point. For this competition, the loss function is the square-error for the log-sale-price,
# 
# \begin{align}
# L(y,\delta(\mathbf{x})) = \left( \log(y) - \delta(x)\right)^{2} .
# \end{align}
# 
# We will then be assessed using the root-mean of these, i.e. the root-mean-square-error (RMSE), taken over the test set. 
# 
# To make what follows easier, let's now redefine the target of our problem, $y$, to be the log-sale-prices. Then the loss function is simply the usual quadratic loss,
# 
# \begin{align}
# L(y,\delta(\mathbf{x})) = \left( y - \delta(x)\right)^{2} .
# \end{align}
# 
# With the loss function settled, we are now ready to given the solution to our problem: When we are given a new observation in the form of a feature vector, $\mathbf{x}_0$ (for model one this is just a specific living area) then the corresponding log-sale-price $y_{0}$ is unknown. Therefore, we construct (the hard bit!) a probability for this unknown quantity, $p(y_{0}|\mathbf{x}_{0})$. Then, for a chosen decision function, the expected loss for this data point, which we denote $\rho(\mathbf{x}_{0}, \delta(\mathbf{x}_{0}))$, is given by summing/integrating over all possibly log-sale-prices, weighted by the probability of occuring,
# 
# \begin{align}
# \rho(\mathbf{x}_{0}, \delta(\mathbf{x}_{0})) &= \int dy ~ L(y,\delta(\mathbf{x}_{0})) ~ p(y|\mathbf{x}_{0}) \\
# &= \int dy ~ \left( y - \delta(\mathbf{x}_{0})\right)^{2} ~ p(y|\mathbf{x}_{0}) \\
# &= \int dy ~ \left( y^{2} - 2 y \delta(\mathbf{x}_{0}) + \delta(\mathbf{x}_{0})^{2}\right) ~ p(y|\mathbf{x}_{0}) \\
# &= E[y^{2}] - 2 E[y] y_{0} + y_{0}^{2} , \\
# \end{align}
# 
# where in the last line we have used the notation $y_{0} = \delta(\mathbf{x}_{0})$ for the value of the decision function taken at the point $\mathbf{x}_{0}$, to emphasise that this is just a number (equal to the predicted log-sale-price)
# 
# Now, our task is to choose the value of the decision function at this point, i.e. to choose our prediction $y_{0}$, so that the expected loss above is minimised. This can be done by differentiating the above expression with respect to $y_{0}$ and setting the expression equal to zero. If you work through the maths this gives;
# 
# \begin{align}
# y_{0} = E[y] .
# \end{align}
# 
# This is our result! What does it mean? It means that, for any test-point $\mathbf{x}_{0}$ we come up against, we should calculate the expected value of the log-sale-price, $y$, using our conditional probability $p(y|\mathbf{x}_{0})$ and return that as our answer. Note that is always the best thing to do no matter how we construct $p(y|\mathbf{x}_{0})$. Of course, this means that the whole task is to come up with a way to model $p(y|\mathbf{x}_{0})$...

# ### **The Linear Regression Model for $p(y|\mathbf{x})$**
# 
# While we will be looking at linear regression in this set of notebooks, first let's think generally about probabilties like $p(y|x)$, and how the idea of data comes into play when we build them.
# 
# From a Bayesian perspective, probabilites are simply an extension of logic; they are plausible reasoning about statements. Of course, when we reason about the plausibility of something we always bring a lot of assumptions to the table, even if we don't think about them explicitly. Therefore, in some sense, all probabilites are really conditional on a lot of "background" information, which we denote $B$. So really, what we want to calculate is,
# 
# $p(y | x, B) .$
# 
# Of course, most of this background information will be useless in predicting a sales price, so we can ignore most of this information. However, what certaintly is important for our reasoning is the data set we have! Therefore, we can consider instead the (somewhat simplified problem) of modelling
# 
# $p(y | x, \mathcal{D}) .$
# 
# Once we have this, we can use it to make predictions via the expected value of $y$, as discussed above.
# 
# One way we could quite directly construct $p(y | x, \mathcal{D})$ is to simply enumerate all the instances of $(x,y)$ pairs in our data set, before normalising appropriately. This is an example of a non-parameteric model, since we are just using the data directly. However, in these notebooks we will take a different approach and build a parametric model (namely a linear regression model). 
# 
# In a parametric model, all the relevant information from the data for predicting $y$ is assumed to be captured in some set of parameters, written in a parameter vector $\mathbf{\theta}$. Therefore, we can expand $p(y | x, \mathcal{D}) $ as
# 
# \begin{align}
# p(y | x, \mathcal{D})  &= \int d\mathbf{\theta} ~ p(y, \theta | x, \mathcal{D}) \\
# &= \int d\mathbf{\theta} ~ p(y | x, \mathbf{\theta}, \mathcal{D})p(\mathbf{\theta} | x, \mathcal{D}) \\
# &= \int d\mathbf{\theta} ~ p(y | x, \mathbf{\theta})p(\mathbf{\theta} | \mathcal{D}) ,
# \end{align}
# 
# where in the final line we have used our assumption that the data is irrelevant for our reasoning beyond the information contained in $\mathbf{\theta}$, and the fact that knowing the current test point $x$ has no impact on our assessment of the plausibility of $\mathbf{\theta}$.
# 
# Now, with this expression, we are ready to introduce the linear regression model. The linear regression model is a parameteric model of the form,
# 
# \begin{align}
# p(y | x, \mathbf{\theta}) = \mathcal{N}\left(y | \mathcal{x}^{T}\mathcal{w}, \sigma^{2}\right) . 
# \end{align}
# 
# In words, we predict the probability of a log-sale-price $y$ using a normal distribution, $\mathcal{N}(\mu,\sigma^{2})$, where the parameters are the weights $\mathcal{w}$ for each feature and the variance of the Gaussian $\sigma^{2}$ (i.e. $\theta = (\mathbf{w},\sigma^{2})$). Note that the mean (expected value) of the Gaussian, $\mu = E[y] = \mathcal{x}^{T}\mathcal{w}$ does not depend on the parameter $\sigma^{2}$, only on the weights. For one feature, we can write explicitly that,
# 
# \begin{align}
# \mu = \mathbf{w}^{T}\mathbf{x} = w_{0} x_{0} + w_{1} x_{1} = w_{0} + w_{1} x_{1} ,
# \end{align}
# 
# where we define $x_{0} := 1$ for convenience. Note also that, for a Gaussian, the mean is also equal to the mode (defined as the point of maximum probability).
# 
# *[while I will be loose with language, actually $p(y|x)$ is a probability density, not a probability, since the probabiity for any continunous variable to take on an exact value is zero. The probability density is proportional to the probability for finding the sale price to be in some small region around the values $y$, so some of the intuition from probabilites holds. However, rather than sum to one, probability densities integrate to one (the area under the curve equals one)]*

# ### **Decision Theory for Linear Regression**
# 
# From decision theory we know that for a given test point we should predict the mean of $p(y|\mathbf{x}_{0},\mathcal{D})$. In the case of linear regression, this problem takes of a particularly simple form. 
# 
# To see this, we insert the linear regression expression for $p(y | x, \mathbf{\theta})$ into our equation for the optimal action, $y_{0} = E[y]$, and simplify:
# 
# \begin{align}
# y_{0} &= E[y] \\
# &= \int dy ~y \left[p(y|\mathbf{x}_{0})\right] \\
# &= \int dy ~ y \left[ \int d\mathbf{w} d\sigma^{2} ~ \mathbf{N}\left(y | \mathbf{x}_{0}^{T}\mathcal{w}, \sigma^{2}\right) p(\mathbf{w},\sigma^{2}|\mathcal{D}) \right] \\
# &=  \int d\mathbf{w} d\sigma^{2} ~ \left[ \int dy ~ y ~\mathbf{N}\left(y | \mathbf{x}_{0}^{T}\mathcal{w}, \sigma^{2}\right) \right] p(\mathbf{w},\sigma^{2}|\mathcal{D})  \\
# &= \int d\mathbf{w} d\sigma^{2} ~ \left( \mathbf{x}_{0}^{T}\mathbf{w} \right) p(\mathbf{w},\sigma^{2}|\mathcal{D}) \\
# &= \int d\mathbf{w} ~ \left( \mathbf{x}_{0}^{T}\mathbf{w} \right) p(\mathbf{w}|\mathcal{D}) \\
# &= \mathbf{x}_{0}^{T}E[\mathbf{w}] .
# \end{align}
# 
# So, for linear regression, the optimal prediction to make for any test point $\mathbf{x}_{0}$ is given by taking the dot product of this feature vector with the *mean weight vector*, when the mean is taken with respect to the (marginal posterior) distribution for the weights, $p(\mathbf{w}|\mathcal{D})$. Therefore, in linear regression, our whole task comes down to constructing the mean of $p(\mathbf{w}|\mathcal{D})$!
# 
# The probability $p(\mathbf{w} | \mathcal{D})$ tells us how likely a given set of weights are, based on the data we have. In the next theory section, we will go into more detail about how to calculate this. However, a simple approxmation we can make is just to assume that the value of $\mathbf{w}$ is certain, i.e. the probability is zero for anything other than a given "true" value. This approximation will be reasonable if we have a lot of data that selects out some special set of parameters that are much more likely than any others. 
# 
# To proceed for now, we will then use the approximation, which seems quite reasonable from a geometric perspective (and indeed is from a statistical perspective as we will see), that, given the data we have, the only possible weights are those that minimise the residual-sum-of-squares (RSS), see the code for how this is defined/calculated.
# 
# Let us call the weights that minimise the RSS $\hat{\mathbf{w}}$. Then, our approximation for the weight probability density is,
# 
# \begin{align}
# p(y | x, \mathbf{w}) \approx \hat{\delta}_{\mathbf{w}, \hat{\mathbf{w}}} ~,
# \end{align}
# 
# where $\hat{\delta}_{a, b}$ is the kronecker delta function, equal to one if $a=b$ and zero otherwise. Of course, the expected value of a probability distribution where only one value is possible is just that value! Therefore, our optimal prediction is simply
# 
# \begin{align}
# y_{0} = \mathbf{x}_{0}^{T} \hat{\mathbf{w}} .
# \end{align}
# 
# So, after a lot of work, we have found that all we need to do for now is combine the weights we obtain by fitting (minimising the RSS) with the provided feature vector, $\mathbf{x}_{0}$, to make our optimal prediction for the log-sale-price.
# 
# While this result is very simple, it is suprisingly general. We will see this more in the next theory section but, for now, let's conclude by using sklearn to build our model.

# ## Theory 1: Summary (for Busy People)
# 
# - for the quadratic loss function that we are assessed on in this competition, we should make predictions by taking the dot product of a feature vector of a test point and the vector of mean (expected) weights
# - a simple approximation for the probability distribution of weights is just to take a single value. This is a good approximation when the distibution of weights is highly peaked, as will often happen with a lot of data.
# - The weights that minimise the residual sum of squares are often equal to the mean of the distribution of weights, so simply using the weights that minimise the residual sum of squares is a good option generally.

# ## **Fitting Model 1 with Sklearn**
# 
# Now, with the theory out of the way, let's fit our linear regression model and see how we do on the leaderboard.

# In[ ]:


def residual_sum_of_squares(y_predictions,y_values):
    """ This function calculates the residual sum of squares. 
        A residual is the true y value minus the predicted value.
        Squaring the residuals and adding them up gives the RSS
    """
    residuals = y_values - y_predictions # make the vector of residuals via elementwise subtraction
    residuals_squared = residuals**2 # elementwise squaring
    RSS = np.sum(residuals_squared)
    
    return RSS

def root_mean_square_error(y_predictions,y_values):
    """ This function calculates the RSME. 
        This is given by taking the root of the RSS divided by the number of data points
    """
    
    N = len(y_values) # The number of data points
    RSS = residual_sum_of_squares(y_predictions,y_values) # see above
    MSE = RSS/N # The mean squared error
    RMSE = np.sqrt(MSE)
    
    return RMSE


# In[ ]:


from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression() # This will find the weights that minimise the RSS
ls_solution= linear_regression.fit(X,y) # ls stands for "least-squares" i.e. it minimises the RSS
y_predictions = ls_solution.predict(X) # this calculates the y-predictions
RMSE = root_mean_square_error(y_predictions,y)


# In[ ]:


# Now let's plot the best-fit line through our data 
plt.figure(figsize=(10,6))
plt.plot(X,y,'ro')
plt.plot(X,ls_solution.predict(X),'b-') # blue and red contrast well for the colourblind. Please not green and red.
plt.title('RMSE = {:.4f}'.format(RMSE)) # the format method on a string is useful, it places the given value ...
plt.ylabel('log-sale-price')    # .. into the {}. The :.4f syntax in {} tells it to keep 4 dp.
plt.xlabel('Gr Live Area')
plt.show()


# In[ ]:


### make predictions on test-set

test_predictions = ls_solution.predict(X_test)
test_predictions = np.exp(test_predictions) # undo the log for submission!

### Now make submission dataframe and export
sub = pd.DataFrame() # a blank submission dataframe
sub['Id'] = test['Id'] # the Id of the test points, otherwise it won't know which prediction is which!
sub['SalePrice'] = test_predictions # the sale-price, which we are predicting
sub.to_csv('submission_liv_area_linear_regression.csv',index=False) #export as a csv
## To actually submit this to the competition, you can view the kernel outside the editor, go to output ... 
### ..., find the csv file, select it and click submit to competition


# On the leaderboard this gets an RMSE of 0.296, which isn't too bad considering the model we used (only one feature!). Note it is also quite close to the RMSE on our training data, though a bit higher.

# # Model 2: Linear Regression with all Features (0.141 on LB)

# For our second model we will again use the linear regression model from before, but now using the all the features. Of course, this should do much better. 
# 
# However, it does mean we need to deal with problems such as missing values and categorical data. Since we are focussing on the machine learning side of things in this notebook, I have adapted the work from
# 
# https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
# 
# to do this, so credit goes to juliencs and please see that kernel for details!
# 
# What I have tried to do is break up the processes of handling missing values, encoding categorical variables etc into some functions. Hopefully this means that you can come back and try editing them for yourself e.g. try filling in the missing values in a different ways. Enjoy!
# 

# In[ ]:


def handle_missing_values(df):
    """
    This function handles the missing values
    The methods are taken from:
        https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    see there for details! (the comments are also taken from that kernel)
    """

    # Handle missing values for features where median/mean or most common value doesn't make sense
    # Alley : data description says NA means "no alley access"
    df.loc[:, "Alley"] = df.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
    df.loc[:, "BedroomAbvGr"] = df.loc[:, "BedroomAbvGr"].fillna(0)
    # BsmtQual etc : data description says NA for basement features is "no basement"
    df.loc[:, "BsmtQual"] = df.loc[:, "BsmtQual"].fillna("No")
    df.loc[:, "BsmtCond"] = df.loc[:, "BsmtCond"].fillna("No")
    df.loc[:, "BsmtExposure"] = df.loc[:, "BsmtExposure"].fillna("No")
    df.loc[:, "BsmtFinType1"] = df.loc[:, "BsmtFinType1"].fillna("No")
    df.loc[:, "BsmtFinType2"] = df.loc[:, "BsmtFinType2"].fillna("No")
    df.loc[:, "BsmtFullBath"] = df.loc[:, "BsmtFullBath"].fillna(0)
    df.loc[:, "BsmtHalfBath"] = df.loc[:, "BsmtHalfBath"].fillna(0)
    df.loc[:, "BsmtUnfSF"] = df.loc[:, "BsmtUnfSF"].fillna(0)
    # CentralAir : NA most likely means No
    df.loc[:, "CentralAir"] = df.loc[:, "CentralAir"].fillna("N")
    # Condition : NA most likely means Normal
    df.loc[:, "Condition1"] = df.loc[:, "Condition1"].fillna("Norm")
    df.loc[:, "Condition2"] = df.loc[:, "Condition2"].fillna("Norm")
    # EnclosedPorch : NA most likely means no enclosed porch
    df.loc[:, "EnclosedPorch"] = df.loc[:, "EnclosedPorch"].fillna(0)
    # External stuff : NA most likely means average
    df.loc[:, "ExterCond"] = df.loc[:, "ExterCond"].fillna("TA")
    df.loc[:, "ExterQual"] = df.loc[:, "ExterQual"].fillna("TA")
    # Fence : data description says NA means "no fence"
    df.loc[:, "Fence"] = df.loc[:, "Fence"].fillna("No")
    # FireplaceQu : data description says NA means "no fireplace"
    df.loc[:, "FireplaceQu"] = df.loc[:, "FireplaceQu"].fillna("No")
    df.loc[:, "Fireplaces"] = df.loc[:, "Fireplaces"].fillna(0)
    # Functional : data description says NA means typical
    df.loc[:, "Functional"] = df.loc[:, "Functional"].fillna("Typ")
    # GarageType etc : data description says NA for garage features is "no garage"
    df.loc[:, "GarageType"] = df.loc[:, "GarageType"].fillna("No")
    df.loc[:, "GarageFinish"] = df.loc[:, "GarageFinish"].fillna("No")
    df.loc[:, "GarageQual"] = df.loc[:, "GarageQual"].fillna("No")
    df.loc[:, "GarageCond"] = df.loc[:, "GarageCond"].fillna("No")
    df.loc[:, "GarageArea"] = df.loc[:, "GarageArea"].fillna(0)
    df.loc[:, "GarageCars"] = df.loc[:, "GarageCars"].fillna(0)
    # HalfBath : NA most likely means no half baths above grade
    df.loc[:, "HalfBath"] = df.loc[:, "HalfBath"].fillna(0)
    # HeatingQC : NA most likely means typical
    df.loc[:, "HeatingQC"] = df.loc[:, "HeatingQC"].fillna("TA")
    # KitchenAbvGr : NA most likely means 0
    df.loc[:, "KitchenAbvGr"] = df.loc[:, "KitchenAbvGr"].fillna(0)
    # KitchenQual : NA most likely means typical
    df.loc[:, "KitchenQual"] = df.loc[:, "KitchenQual"].fillna("TA")
    # LotFrontage : NA most likely means no lot frontage
    df.loc[:, "LotFrontage"] = df.loc[:, "LotFrontage"].fillna(0)
    # LotShape : NA most likely means regular
    df.loc[:, "LotShape"] = df.loc[:, "LotShape"].fillna("Reg")
    # MasVnrType : NA most likely means no veneer
    df.loc[:, "MasVnrType"] = df.loc[:, "MasVnrType"].fillna("None")
    df.loc[:, "MasVnrArea"] = df.loc[:, "MasVnrArea"].fillna(0)
    # MiscFeature : data description says NA means "no misc feature"
    df.loc[:, "MiscFeature"] = df.loc[:, "MiscFeature"].fillna("No")
    df.loc[:, "MiscVal"] = df.loc[:, "MiscVal"].fillna(0)
    # OpenPorchSF : NA most likely means no open porch
    df.loc[:, "OpenPorchSF"] = df.loc[:, "OpenPorchSF"].fillna(0)
    # PavedDrive : NA most likely means not paved
    df.loc[:, "PavedDrive"] = df.loc[:, "PavedDrive"].fillna("N")
    # PoolQC : data description says NA means "no pool"
    df.loc[:, "PoolQC"] = df.loc[:, "PoolQC"].fillna("No")
    df.loc[:, "PoolArea"] = df.loc[:, "PoolArea"].fillna(0)
    # SaleCondition : NA most likely means normal sale
    df.loc[:, "SaleCondition"] = df.loc[:, "SaleCondition"].fillna("Normal")
    # ScreenPorch : NA most likely means no screen porch
    df.loc[:, "ScreenPorch"] = df.loc[:, "ScreenPorch"].fillna(0)
    # TotRmsAbvGrd : NA most likely means 0
    df.loc[:, "TotRmsAbvGrd"] = df.loc[:, "TotRmsAbvGrd"].fillna(0)
    # Utilities : NA most likely means all public utilities
    df.loc[:, "Utilities"] = df.loc[:, "Utilities"].fillna("AllPub")
    # WoodDeckSF : NA most likely means no wood deck
    df.loc[:, "WoodDeckSF"] = df.loc[:, "WoodDeckSF"].fillna(0)

    return df


# In[ ]:


def handle_numerical_categories(df):
    """
    This function converts things like numerical dates to categories
    The methods/comments are taken from:
        https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    see there for details!
    """
    # Some numerical features are actually really categories
    df = df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                           50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                           80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                           150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                           "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                       7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                          })  
    
    return df


# In[ ]:


def encode_ordinal_categorical_data(df):
    """
    The methods/comments are taken from:
        https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    see there for details!
    """
    # Encode some categorical features as ordered numbers when there is information in the order
    df = df.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                           "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                           "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                           "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                           "Min2" : 6, "Min1" : 7, "Typ" : 8},
                           "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                           "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                           "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                           "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                           "Street" : {"Grvl" : 1, "Pave" : 2},
                           "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                         )  
    
    return df


# In[ ]:


def encode_categorical_complete_missing(df):
    """
    The methods/comments are taken from:
        https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset
    see there for details!
    Use one-hot encoding for the categorical variables and use median for any remaining numerical vars
    """
    
    # Differentiate numerical features (minus the target) and categorical features
    categorical_features = df.select_dtypes(include = ["object"]).columns
    numerical_features = df.select_dtypes(exclude = ["object"]).columns
    df_num = df[numerical_features]
    df_cat = df[categorical_features]    
    
    # Handle remaining missing values for numerical features by using median as replacement
    df_num = df_num.fillna(df_num.median())
    
    # Create dummy features for categorical values via one-hot encoding
    df_cat = pd.get_dummies(df_cat)
    
    # Join categorical and numerical features
    df = pd.concat([df_num, df_cat], axis = 1)
    
    return df
    
    


# In[ ]:


def process_input_data(df):
    
    """
    Put all the previous functions together
    """
    
    df = handle_missing_values(df)
    df = handle_numerical_categories(df)
    df = encode_ordinal_categorical_data(df)
    df = encode_categorical_complete_missing(df)

    return df


# With the functions defined, let's apply them to our datasets and prep them for modelling.

# In[ ]:


### Reload the data (just in case I accidently changed them before..) 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
## Drop the Id columns, which are not used for prediction, but save test_ids for submission!
train.drop("Id", axis = 1, inplace = True)
test_ids = test['Id']
test.drop("Id", axis = 1, inplace = True)
# extract the target from training data
y = np.array(train['SalePrice'])
y = np.reshape(y,(y.shape[0],1))
y = np.log(y) # log it again.
# Drop sale price so train only has predictors in it
train.drop("SalePrice", axis = 1, inplace = True)


# In[ ]:


# Process the data using the functions from above
train = process_input_data(train)
test = process_input_data(test)
print('Train shape = {} '.format(train.shape))
print('Test shape = {}'.format(test.shape))
# Note that these do not have the same number of features! We must align them!


# In[ ]:


def align_training_and_test(df_train,df_test):
    """
    This function drops the features that aren't present in both the training and test set,
     which would prevent us from using the model fitted using the training features to the test data
         (which might not have the same features)
                 
    """
    df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)
    
    return df_train, df_test


# In[ ]:


## Align the test and training sets
train,test = align_training_and_test(train,test)
print('Train shape = {} '.format(train.shape))
print('Test shape = {}'.format(test.shape))


# In[ ]:



## Cast the data sets into design matrices 
X = np.array(train)
X_test = np.array(test)
N,D = X.shape


# Now that we have prepped our data, all that is left is to fit to it. After a large chunk of theory that is...

# ## Theory 2: Parameter Inference

# ### The Idea of Estimating Parameters and Priors
# 
# In machine learning, our goal is to make use of data to develop our models. In the simplest case, this means picking a model that is fixed up to a some parameters, and then estimating those parameters using the observed data.
# 
# Starting from our Linear Regression model, the parameters we must estimate are the weights, $\mathbf{w}$, and the variance $\sigma^{2}$, which we can also combine into a single parameter vector, $\mathbf{\theta}$, for convenience.
# 
# Now, what is means to "estimate" these parameters is sometimes a bit obscure. However, what we really mean is that we want to write down a probability distribution for them. For example, if we think $w_{1}$ should be between (or equal to) the values $a$ and $b$, i.e. $a \le w_{1} \le b$, but beyond that have no idea, we could use a uniform distribution for $w_{1}$,
# 
# \begin{align}
# p(w_{0}) = \text{Unif}(w_{0}|a,b) = \frac{1}{b-a}\mathcal{I}(a \le x \le b) , 
# \end{align}
# 
# where $\mathcal{I}(s)$ is the *indicator function*, equal to one is the statement $s$ is true or zero otherwise.
# 
# This type of guess, made without actually considering the data we have, is known as a *prior*. Since we always need to start from somewhere, we always need to choose a prior. However, a good rule is to only put into the prior the information you feel is important, or you are sure of. 
# 
# In many cases, we have no idea what values the parameters should take before seeing the data. Therefore, picking something like the uniform distribution is a good idea for a prior; it says that we don't initially favour any one value of the parameters above any other. Priors of this kind are called uninformative.

# ### The posterior for parameters and Bayes' Theorem
# 
# While the prior is the probability we assign to our parameters before seeing any data, what were are really interested in is the *posterior*, which is the probabilties for the parameter values given the data we have see, $p(\mathbf{\theta} | \mathcal{D})$.
# 
# To actually calculate the posterior, we must express it in terms of things we actually know. This is done by using Bayes' theorem,
# 
# \begin{align}
# p(\mathbf{\theta} | \mathcal{D}) = p(\mathcal{D} |\mathbf{\theta}) \frac{p(\mathbf{\theta})}{p(\mathcal{D})} .
# \end{align}
# 
# Notice how in this expression the right-hand-side contains things we can actually calculate:
# 
# First, there is the prior probability, $p(\mathbf{\theta})$. As mentioned, this is something that we can choose and will be very important when we discuss regularisation.
# 
# Second, there is the *likelihood*, $p(\mathcal{D} |\mathbf{\theta})$. This is just the probability our linear regression model predicts for the data we observe, something we can calculate using the expression for our model.
# 
# Finally, the denominator $p(\mathcal{D})$ can be thought of just as a "normalisation factor" since it doesn't depend on our parameters at all and only serves to ensure that the probabilities of all our possible parameter values sum to one, as they must do, (this means that our probability density must integrate to one).
# 
# Since the denominator just gives a normalsiation factor, we often just specify our posteriors as 
# 
# \begin{align}
# p(\mathbf{\theta} | \mathcal{D}) \propto p(\mathcal{D} |\mathbf{\theta}) p(\mathbf{\theta}).
# \end{align}
# 
# This simplifies notation, and we know the missing factor is just given by the normalisation of probabilites.

# ### The Likelihood for Linear Regression
# 
# To compute the likelihood, we assume that our data are independent of each other, and all drawn from the same distribution (they are i.i.d). We then can calculate the probability of our data by multiplying together the probabilities for each individual data point. Therefore, using the i.i.d assumption, our likelihood is,
# 
# \begin{align}
# p(\mathcal{D} |\mathbf{w}) = p(y_{1} | x_{1},\mathbf{\theta}) p(y_{2}| x_{2} \mathbf{\theta}) p(y_{3} | x_{3},\mathbf{\theta}) ...
# \end{align}
# 
# In linear regression, we model the probability of a data point using the normal, defined above. Therefore, the likelihood is a product of Gaussians, one for each data point,
# 
# \begin{align}
# p(\mathcal{D} |\mathbf{\theta}) = \mathcal{N}\left(y_{1}|\mathbf{x_{1}}^{T}\mathbf{w},\sigma^{2}\right)\mathcal{N}\left(y_{2}|\mathbf{x_{2}}^{T}\mathbf{w},\sigma^{2}\right) ...
# \end{align}
# 
# To combine these into a single expression, we can use the definition of the normal distribution in terms of exponential functions and combine them. The result is,
# 
# \begin{align}
# p(\mathcal{D} |\mathbf{\theta}) = \mathcal{N}\left(\mathbf{y}|\mathbf{X}\mathbf{w},\sigma^{2}\right) ,
# \end{align}
# 
# where $\mathbf{y}$ is the vector of log-sale-prices in the data set and $\mathbf{X}$ is the design matrix, of size $(N,D)$.

# ### Maximum Likelihood Estimation for Linear Regression
# 
# With the likelihood calculated, we are now ready to evalulate our posterior for the parameters in the case of a uniform prior. Since for a unform prior we can write $p(\mathbf{w}) \propto 1$ (in words, the prior has no dependence on the parameters values, i.e., it treats them all as equally likely), then the posterior becomes,
# 
# \begin{align}
# p(\mathbf{\theta} | \mathcal{D}) \propto p(\mathcal{D} |\mathbf{\theta}).
# \end{align}
# 
# So it is just proportional to the likelihood! In the case of linear regression this means that
# 
# \begin{align}
# p(\mathbf{\theta} | \mathcal{D})  = \mathcal{N}\left(\mathbf{y}|\mathbf{X}\mathbf{w},\sigma^{2}\right).
# \end{align}
# 
# So the posterior is just a normal. This is very nice for the following reason; we want to calculate the expected value of the posterior, to find the mean weights for prediction. However, in the case of a normal distribution the mean is equal to the mode. Therefore, all we need to do in this case is find the mode of the likelihood, i.e., we need to find the weights that maximise the likelihood, which is known as maximum likelihood estimation.
# 
# As we will see later in theory section 3 (next notebook), maximising the likelihood is the same as minimising the negative log-likelihood, we which can write as the optimisation problem,
# 
# \begin{align}
# \hat{\mathbf{\theta}} = \argmin_{\mathbf{\theta}} \left( - \log\left[p(\mathcal{D} |\mathbf{w})\right] \right) .
# \end{align}
# 
# This is exactly what we solve when we call the fit function in linear regression! For the weights, minimising the negative log-likelihood is exactly the same as minimising the sum of squared residuals, i.e. solving the problem,
# 
# \begin{align}
# \hat{\mathbf{w}} = \argmin_{\mathbf{w}} \left[ RSS(\mathbf{w},\mathcal{D}) \right],
# \end{align}
# 
# which was our motivation from the first theory section. Now we see that was justified in the special case of uniformative priors and the maximum likeliood principle.

# ## Theory 2: Summary
# * To make predictions, we need to establish the values of our parameters using the provided data
# * Since the parameters are unknown, we can assign probabilites to their values
# * Before seeing the data, we can reasonably take the position that no value is more likely than any other
# * For linear regression, with the above assumption, we find the expected value of the weights - which we use to calculate our prediction - is given by minimising the RSS, just as we have been doing.

# Ok, theory done. Let's fit the model.

# In[ ]:


def negative_log_likelihood(RSS,variance):
    ### combine the RSS with the MLE estimate of variance to compute the NLL   
    NLL = 1/(2*variance)*RSS + y.shape[0]/2*np.log(2*np.pi*variance)
    
    return NLL


# fit the model...

# In[ ]:


linear_regression = LinearRegression()
mle_solution= linear_regression.fit(X,y) 
y_predictions = mle_solution.predict(X) 
RMSE = root_mean_square_error(y_predictions,y)
# RSS = N*(RMSE**2)
# NLL = negative_log_likelihood(RSS,mle_variance)

# print(NLL)
print('Linear regression with all features gives a RMSE = {:.4f}'.format(RMSE))


# export the predictions...

# In[ ]:


predictions = mle_solution.predict(X_test)
predictions = np.exp(predictions)

### Now make subission dataframe and export
sub = pd.DataFrame()
sub['Id'] = test_ids
sub['SalePrice'] = predictions
sub.to_csv('submission_linear_regression.csv',index=False)


# This scores 0.14146 on the LB, top 57%. An improvement, but well off the 0.10513 on our training set. Of course, we will need to deal with this! As you probably know, the issue is that of overfitting and the solution is... regularisation. But thats for the next notebook...
