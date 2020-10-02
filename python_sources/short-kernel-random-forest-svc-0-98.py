#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Load Data**
# 
# Let's load the files and check for any missing values in both datasets. 

# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print("Dataset ready")


# In[ ]:


for i in df_test.isnull().any():
    if i == True:
        sys.exit("There is unlabeled data")
print("No missing values found")


# In[ ]:


for i in df_train.isnull().any():
    if i == True:
        sys.exit("There is unlabeled data")
print("No missing values found")


# ** Split the Data**
# 
# Next, I'll split the dataset into x and y data and check if the shape of the test data and train data is the same.

# In[ ]:


y = df_train["label"]
x = df_train.drop("label", axis=1)


# In[ ]:


x.shape


# In[ ]:


df_test.shape


# **Transform**
# 
# Now I divide the arrays by 255. I do this so that the values are normalized, i.e. between 0 and 1 and distortion is reduced. 
# 
# Then I just have a sample displayed to see if the dataset works. For this you have to convert the array from 1x784 to 28x28, because the image has this size. 

# In[ ]:


x = x / 255.0
df_test = df_test / 255.0

print("Completed")


# In[ ]:


plt.imshow(df_test.values.reshape(-1, 28, 28, 1)[40][:,:,0])


# ** Training ** 
# 
# Now I split the training data into a train and test sample. After this I'll train the model with a RFC. 

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=2)

print("Completed")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion = "entropy")
model.fit(x_train, y_train)

print(model.score(x_test, y_test))


# **Hyperparameter Tuning**
# 
# The results are not so bad, but not good either. Let's do some hyperparameter tuning with GridSearchCV.
# Due to the size of the dataset I will only use the first 1000 entries to test the hyperparameters. We can do this because the data is not sorted. But you should note that the results don't have to be 100% correct.
# 
# After this, I'll train the model with the parameters and the whole dataset.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(1, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

forrest = RandomForestClassifier(criterion = "entropy")
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
forest_cv.fit(x_train.values[: 100, :], y_train.values[: 100])
print(forest_cv.best_score_)
print(forest_cv.best_estimator_)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion = "entropy", max_depth=12, min_samples_leaf=1, min_samples_split=6, n_estimators=50)
model.fit(x_train, y_train.values)

print(model.score(x_test, y_test))


# **First Results**
# 
# As you can see the RFC provides good results in an acceptable amount of time and hyperparameter tuning improved the accuracy by 2%. Now let's see if the model would improve with more data. 

# In[ ]:


#Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle

x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(RandomForestClassifier(criterion = "entropy", n_estimators = 50), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))


# **Another Classifier**
# 
# To further improve the model I used another classifier. With some hyper-parameter tuning, which I won't show here because it makes the kernel extremely slow, I was able to get good results. However, it is noticeable that the SVC is much slower than the RFC. 

# In[ ]:


from sklearn.svm import SVC

model = SVC(kernel = "rbf", gamma = 0.021, C = 2.1)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


# In[ ]:


pred = model.predict(df_test)

submission = pd.DataFrame({
    "ImageId": list(range(1,len(pred)+1)),
    "Label": pred
})

submission.to_csv("submision.csv", index=False)

print("Prediction Completed")


# In[ ]:




