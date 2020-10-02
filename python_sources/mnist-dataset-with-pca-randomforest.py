#!/usr/bin/env python
# coding: utf-8

# # MNIST Dataset with PCA+RandomForest
# - Loading Data.
# - Visulization of Images.
# - Preprocessing.
#     - Scaling the data.
#     - Principal Component Analysis.
# - Chosing Number Of Components.
# - RandomForest classifier
#     - Tuning Model
# - Submission

# # Loading Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing,
import warnings    # warning to show or hide
import random     # For random pocess
import matplotlib.pyplot as plt     # Plotting Images, Graph
from IPython.display import display, Markdown

# Ignoring Wornings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Reading Training Data
train_data = pd.read_csv('../input/train.csv')
# Reading Testing Data
test_data = pd.read_csv('../input/test.csv')
print("Training data informataion")
print(train_data.info())
print("Training Data Shape: {0}\nTesting Data Shape: {1}".format(train_data.shape, test_data.shape))


# In[ ]:


# Training Data Columns
train_data.columns


# Separating features and labels from traning data.

# In[ ]:


# Get Featrues and Labels
featrue = train_data.drop(['label'], axis=1)
label = np.asarray(train_data[['label']])
print("Unique Digit Labels: {}".format(np.unique(label)))


# # Visulization of Images
# Lets have visulization of MNIST handwritten digit dataset.

# In[ ]:


# Creating custom subplot function to visualize the image
def plotImage(image_data, label, number_of_plots=3):
    
    # Generate Random number for position
    pos_of_image = [random.randint(0, image_data.shape[0]) for p in range(0, number_of_plots*number_of_plots)]
    
    # Defining figure
    fig = plt.figure(figsize=(10, 10))
    
    # plotting subplots
    for pos, plotnumber in zip(pos_of_image, range(1, (number_of_plots*number_of_plots)+1)):
        ax = fig.add_subplot(number_of_plots, number_of_plots, plotnumber)
        
        # pLotting image
        ax.imshow(image_data[pos], cmap='gray')
        ax.set_title("Digit-{}".format(label[pos][0]), fontsize=14)
        fig.tight_layout()

# Converting the 1-D digit array to 2-D array.
# Using reshape function 784 >> 28,28
image_array = np.reshape(np.asarray(featrue), (len(featrue), 28, 28))

# Calling custom plot funcation
display(Markdown("### MNIST Handwritten Digits"))
plotImage(image_array, label, number_of_plots=5)


# # Preprocessing
# ## Scaling the data
# - Scaling is most important when it comes to data preprocssing. 
# - The independent feature which ranges in 0-1000 values makes the optimization slower i.e. training gets slower. 
# - In our case the digit pixel values ranges from 0-255. Which will make our classifier system slower. 
# - So to imporve the training we normalize the data using Normalizer which will convert the data in range of -1 to 1 or 0 to 1.

# In[ ]:


from sklearn.preprocessing import Normalizer

# Scaling Data with Normalizer
def norm(input_data):
    nm = Normalizer()
    nm.fit(input_data)
    input_data_scale = nm.transform(featrue)
    return input_data_scale, nm

featrue_scale, nm = norm(featrue)


# ## Dimensionality Reduction: Principal Component Analysis
# - The dimensionality reduction is another most important part of data preprocessing.
# - Lets say we have data of vehicles. In which type of vehicle (2-wheeler,4-wheeler), number of tyres (2 tyre,4 tyre) which are co-related to a high degree. Hence, we can reduce the number of features in such problems.
# - In this case we have 784 features. Whcih represnts the pixels values of digit. These feature are correalted so we can readuce the number of features.
# - Approch is done using Principal Component Analysis. 
# - Lets calculate optimum number of feature is which covers 90% of PCA variance.

# In[ ]:


from sklearn.decomposition import PCA

component_with_var = {}
# We will use variacne range instead  of number features. Looping 784 features is slower than 10 varince range.
variance_list = np.arange(0.1, 1.1, 0.1)

for var in variance_list:
    if var < 1.0:
        # Selecting 2nd shape value which is nothing but number of components.
        component_with_var[str(var*100)+' %'] = PCA(var).fit_transform(featrue_scale).shape[1]
    else:
        component_with_var[str(var*100)+' %'] = featrue_scale.shape[1]
component_with_var


# In[ ]:


variance_cover = 0.9    # 90% Of varince cover
# Get component list
component_list = list(component_with_var.values())
# using numpy interp function we will get number of features(components) for the 90% varince coverage.
components = int(np.interp(variance_cover, variance_list, component_list))
print("Optimum number of feature or component is {0}".format(components))


# In[ ]:


plt.plot(variance_list, component_list)
plt.scatter(variance_cover, components, color='red')
plt.annotate('{}% variance covered by \n{} features'.format(variance_cover*100,components), 
             xy=(variance_cover, components), xytext=(0.3,400),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=14
            )
plt.show()


# From above calculation we found that up to 229 we can reduce the features. Which covers 90% of variance.

# In[ ]:


# Lets reduce the dimension of given dataset
def pca_transform(input_data, components):
    pca = PCA(n_components=components)
    pca.fit(input_data)
    input_data_pca = pca.transform(input_data)
    return input_data_pca, pca
feature_pca, pca_model = pca_transform(featrue_scale, components)


# # Train-Test-Split
# Lets split the training data into train-test data for cross validations

# In[ ]:


from sklearn.model_selection import train_test_split
# Train data 80%, Test data 20%
X_train, X_test, Y_train, Y_test = train_test_split(feature_pca, label, test_size=0.2)


# # RandomForsest Classifier

# In[ ]:


# Calling Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Taking effective params 
rfr_grid = {
    "n_estimators":[100, 200],
    "max_depth":[2, 5, 10, None]
}


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fitting model\nclf = GridSearchCV(RandomForestClassifier(),param_grid=rfr_grid,verbose=5)\nclf.fit(X_train, Y_train)')


# In[ ]:


# Best params are
clf.best_params_


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Fitting with best params\nrfr = RandomForestClassifier(\n    n_estimators=200\n)\nrfr.fit(X_train,Y_train)')


# In[ ]:


# Train score
rfr.score(X_train,Y_train)


# In[ ]:


# ValidationTest Score
rfr.score(X_test, Y_test)


# Validation Testing scores 95% 

# # Submission

# In[ ]:


# Test data
test_data.head(3)


# In[ ]:


# Preprocessing Test data
test_scale = nm.transform(test_data)
test_pca = pca_model.transform(test_scale)


# In[ ]:


# Predicted Digit
predicted_digit = rfr.predict(test_pca)


# In[ ]:


# Submission
sub =  pd.DataFrame(range(1,len(test_data)+1),columns=['ImageId'])
sub['Label'] = predicted_digit
sub.to_csv('pca_rfr_digit.csv', index=False)

