#!/usr/bin/env python
# coding: utf-8

# ## kernel 0: trivial solution
# #### Establishing some baseline guesses
# 

# Let's start by importing some important tools that we would use on nearly any project:
# - **numpy** for fast math and linear algebra
# - **pandas** for critical data science tasks
# - **matplotlib** for visualizaiton
# - and **os** for working with file and folders.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# #### import the data
# The data for this competition is stored in the directory `../input/`. Let's see what's in there.

# In[ ]:


os.listdir('../input/')


# For now, we'll just import `inspections_train.csv` and name it `d`.

# In[ ]:


d = pd.read_csv('../input/inspections_train.csv')


# It might be helpful to print the shape (rows x columns) and columns names of the [**dataframe**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) we've imported. We might also want to see the top 5 rows of data. These operations are all pretty easy:

# In[ ]:


print(d.shape)

print(d.columns)


# In[ ]:


d.head()


# Now let's import some basic tools from [**scikit-learn**](http://scikit-learn.org), the most widely used open source python toolkit for machine learning. We then use `train_test_split` to randomly partition our data into two groups: train and test. We specify a test size of 25%, which is the amount of data that we hold out in order to evaluate our work.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

x_train0, x_test0 = train_test_split(d, test_size=0.25)


# It's an important best practice to only use the training data when we're evaluating a problem. This simulates a real-world scenario where you don't know what the rest of the data looks like ahead of time. Using this assumption, let's just see the average probability that restaurant passes inspection in our training data.

# In[ ]:


percent_passed = x_train0.passed.mean()
print(f"Percent of inspections with 'A' grade: {100*percent_passed:.2f}%")


# Ok, so we know that restaurants pass the inspection about $\frac{2}{3}$ of the time. That's a great start. What if we make the simplest possible solution and assign a probability of $1.0$ to every test instance? We'll evaluate the quality of our answer using **log loss**.

# In[ ]:


test_solution0 = np.ones(x_test0.passed.shape)
loss0a = log_loss(x_test0.passed.values, test_solution0)
print(f'log loss: {loss0a:.3f}')


# Our baseline for assiging 100% every time is $11.2$. What if we had guessed $0.0$ instead? It's actually a lot worse- more than twice as bad!

# In[ ]:


test_solution0 = np.zeros(x_test0.passed.shape)
loss0b = log_loss(x_test0.passed.values, test_solution0)
print(f'log loss: {loss0b:.3f}')


# Why is guessing $0.0$ for every answer worse than guessing $1.0$? Because there are many more $1$ values in the solution that $0$. We can actually create a much better constant-valued prediction using a little bit of intuition. If we want to minimize our **log loss**, which constant value would be optimal? Let's try the mean percentage of restaurants that pass inspection!

# In[ ]:


test_solution0 = np.ones(x_test0.passed.shape) * percent_passed
loss0c = log_loss(x_test0.passed.values, test_solution0)
print(f'log loss: {loss0c:.3f}')


# That is a massive improvement! We can actually demonstrate analytically that this is pretty much the best value we can choose by plotting the log loss of every constant-valued prediction from $0.01$ to $0.99$. The black dot shows the value of our guess- looks pretty close to the minimum of this curve!
# <br><br>
# **Note:** It's ok if you don't understand exactly what's going on in the code below. We are basically just defining a function (`probability_soln`) that returns a bunch of predictions that are all the same value. We then plot the results of using that function to output predictions spaced evenly from 0.01 to 0.99, and then plotting the results. Why don't we use 0.00 and 1.00? Because the log loss function does some weird things at those values that we should try to avoid.

# In[ ]:


def probability_soln(shape, val): return np.ones(shape) * val
prediction_values = np.linspace(0.01, 0.99, 100)
plt.plot(prediction_values,
        [log_loss(x_test0.passed, probability_soln(x_test0.shape[0], _)) for _ in prediction_values])
plt.scatter([percent_passed], [loss0c], color='black')
plt.xlabel('probability'); plt.ylabel('log loss')
plt.title('Log loss at different prediction values')
plt.show()


# We've made a significant improvement in our loss score just by fine-tuning our guess... but we're still making the same prediction for each instance, not taking into account the features. We can make much better guesses once we start to dig into the data.

# ### Submitting our solution
# We've developed a way to generate solutions. Now we need to generate solutions for each row in the test data, which we find in inspections_test.csv. The steps are:

# In[ ]:


# load the test data
test_data = pd.read_csv('../input/inspections_test.csv')

# take just the `id` columns (since that's all we need)
submission = test_data[['id']].copy()

# create a `Predicted` column. in this case it's just the `percent_passed` that we calculated earlier
submission['Predicted'] = percent_passed

# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here
submission.columns = ['Id', 'Predicted']

# write the submission to a csv file so that we can submit it after running the kernel
submission.to_csv('submission0.csv', index=False)

# let's take a look at our submission to make sure it's what we want
submission.head()

