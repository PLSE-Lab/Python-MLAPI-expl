#!/usr/bin/env python
# coding: utf-8

# # <center>Forest Cover Classification</center>
# <center>If you liked this kernel and/or found it helpful, please upvote it so others can see it too!</center>

# <img src="https://cdn-images-1.medium.com/max/500/1*QNtpWNresupHbiV7Xe59gg.png">

# # The Forest Cover Type Challenge
# 
# From the competition's <a href="https://www.kaggle.com/c/forest-cover-type-kernels-only">overview page</a>:  
# 
# In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from cartographic variables. The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.
# 
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.
# 
# 

# # Table of Contents
# * [Config and Imports](#config_and_imports)<br>
# * [Feature Overview](#feature_overview)<br>
# * [Load and Explore Data](#load_and_explore_data)<br>
# * [Visualize and Transform Data](#visualize_and_transform_data)<br>
# * [Train Model](#train_model)<br>
# * [Error Analysis](#error_analysis)<br>
# * [Make Predictions on Test Set](#make_predictions)<br>
# * [Next Steps](#next_steps)<br>

# <a id='config_and_imports'></a>
# # Config and Imports

# In[ ]:


# True to spend extra time displaying graphs, False for speedy results
show_plots = True


# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='feature_overview'></a>
# #  Feature Overview
# __Elevation__ - Elevation in meters<br>
# __Aspect__ - Aspect in degrees azimuth<br>
# __Slope__ - Slope in degrees<br>
# __Horizontal_Distance_To_Hydrology__ - Horz Dist to nearest surface water features<br>
# __Vertical_Distance_To_Hydrology__ - Vert Dist to nearest surface water features<br>
# __Horizontal_Distance_To_Roadways__ - Horz Dist to nearest roadway<br>
# __Hillshade_9am (0 to 255 index)__ - Hillshade index at 9am, summer solstice<br>
# __Hillshade_Noon (0 to 255 index)__ - Hillshade index at noon, summer solstice<br>
# __Hillshade_3pm (0 to 255 index)__ - Hillshade index at 3pm, summer solstice<br>
# __Horizontal_Distance_To_Fire_Points__ - Horz Dist to nearest wildfire ignition points<br>
# __Wilderness_Area__ (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation<br>
# __Soil_Type__ (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation<br>
# __Cover_Type__ (7 types, integers 1 to 7) - Forest Cover Type designation<br>

# <a id='load_and_explore_data'></a>
# # Load and Explore Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


train.head()


# <a id='visualize_and_transform_data'></a>
# # Visualize and Transform Data

# **Separate features by type**

# In[ ]:


target_classes = range(1,8)
target_class_names = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine',                       'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']

numerical_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',                     'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',                     'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

categorical_features = [ 'Wilderness_Area', 'Soil_Type' ]


# In[ ]:


# extract target from data
y = train['Cover_Type']
train = train.drop('Cover_Type', axis=1)


# **Plot distribution of target classes**

# In[ ]:


# plot target var
plt.hist(y, bins='auto')
plt.title('Cover_Type')
plt.xlabel('Class')
plt.ylabel('# Instances')
plt.show()


# For each numerical feature, plot the variable distributions in the train and test sets. We want to make sure the distributions are similar in order to perform well on the test set.

# In[ ]:


if show_plots:
    for feature_name in numerical_features:
        plt.figure()
        sns.distplot(train[feature_name], label='train')
        sns.distplot(test[feature_name], label='test')
        plt.legend()
        plt.show()


# Specifically interesting to note that the train and test distributions for Elevation differ significantly. This could be a problem if Elevation is an important feature. Now let's do the same for categorical features (Wilderness and Soil Type).

# In[ ]:


if show_plots:
    # categorical distributions btw train and test set
    train_wilderness_categorical = train['Wilderness_Area1'].copy().rename('Wilderness_Area')
    train_wilderness_categorical[train['Wilderness_Area2'] == 1] = 2
    train_wilderness_categorical[train['Wilderness_Area3'] == 1] = 3
    train_wilderness_categorical[train['Wilderness_Area4'] == 1] = 4

    test_wilderness_categorical = test['Wilderness_Area1'].copy().rename('Wilderness_Area')
    test_wilderness_categorical[test['Wilderness_Area2'] == 1] = 2
    test_wilderness_categorical[test['Wilderness_Area3'] == 1] = 3
    test_wilderness_categorical[test['Wilderness_Area4'] == 1] = 4

    plt.figure()
    sns.countplot(train_wilderness_categorical, label='train')
    plt.title('Wilderness_Area in Train')

    plt.figure()
    sns.countplot(test_wilderness_categorical, label='test')
    plt.title('Wilderness_Area in Test')

    plt.show()


# In[ ]:


soil_classes = range(1,41)

train_soiltype_categorical = train['Soil_Type1'].copy().rename('Soil_Type')
for cl in soil_classes:
    train_soiltype_categorical[train['Soil_Type'+str(cl)] == 1] = cl

test_soiltype_categorical = test['Soil_Type1'].copy().rename('Soil_Type')
for cl in soil_classes:
    test_soiltype_categorical[test['Soil_Type'+str(cl)] == 1] = cl

plt.figure(figsize=(10, 5))
sns.countplot(train_soiltype_categorical, label='train')
plt.title('Soil_Type in Train')

plt.figure(figsize=(10, 5))
sns.countplot(test_soiltype_categorical, label='test')
plt.title('Soil_Type in Test')

plt.show()


# **PCA**

# In[ ]:


pca = PCA(n_components=3)
train_pca = pca.fit_transform(train)
print('Representation of dataset in 3 dimensions:\n')
print(train_pca)


# In[ ]:


if show_plots:
    # graph pca in interactive 3d chart
    # props to Roman Kovalenko's "Data distribution & 3D Scatter Plots" kernel for showing me where to find a good 3d graphing lib

    colors = ['red', 'blue', 'green', 'black', 'purple', 'orange', 'gray']
    # feel free to change the colors up - unfortunately there's usually a tradeoff between aesthetics and readability
    # colors = ['#f45f42', '#f49241', '#db6a0d', '#dba00d', '#ead40e', '#ffb163', '#ea480e']

    traces = []

    # iterate over classes and add each set of points to traces list
    for cl in target_classes:

        # get all 3-d pca vectors that match the current class
        class_pca = train_pca[y[y == cl].index.values]

        class_pca_x = [ pt[0] for pt in class_pca]
        class_pca_y = [ pt[1] for pt in class_pca]
        class_pca_z = [ pt[2] for pt in class_pca]

        trace = go.Scatter3d(
            x=class_pca_x,
            y=class_pca_y,
            z=class_pca_z,
            mode='markers',
            marker=dict(
                color=colors[cl-1],
                size=3
            ),
            name=target_class_names[cl-1]
        )

        traces.append(trace)

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')


# Looks like the classes are pretty intertwined, but there are definitely patterns emerging. Let's handle the data a bit more before training a model!

# **Feature Engineering**

# In[ ]:


# drop uninformative features
train = train.drop('Id', axis=1)


# Now we'll add the following features:
# * Euclidean distance to hydrology
# * Mean distance to amenities (fire points, hydrology, and roadways)
# * Elevation minus vertical distance to hydrology
# 
# Disclaimer: these features were inspired by Lathwal's excellent public kernel. I have left out a few features so you can check his out for the full set!

# In[ ]:


# write a function to transform the train and test sets
# we'll also append an underscore "_" to our engineered feature names to help differentiate them
def add_features(data):
    data['Euclidean_Distance_To_Hydrology_'] = (data['Horizontal_Distance_To_Hydrology']**2 + data['Vertical_Distance_To_Hydrology']**2)**0.5
    data['Mean_Distance_To_Amenities_'] = (data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']) / 3.0
    data['Elevation_Minus_Vertical_Distance_To_Hydrology_'] = data['Elevation'] - data['Vertical_Distance_To_Hydrology']
    return data

train = add_features(train)
test = add_features(test)


# Additionally, I wrote code to convert the angle in degrees (circular, aka 0 = 360) to their cos and sin components - however, the code is commented because it actually lowered performance. Perhaps there was some bias in assigning certain angles which contains information that is lost when it is approximated as cos and sin.

# In[ ]:


# # convert aspect angle in degrees to cos + sin
# train['AspectCos'] = train['Aspect']
# train['AspectSin'] = train['Aspect']

# train['AspectCos'] = train['AspectCos'].apply(lambda x: np.cos(np.deg2rad(x)))
# train['AspectSin'] = train['AspectSin'].apply(lambda x: np.sin(np.deg2rad(x)))

# train = train.drop(['Aspect'], axis=1)


# **Feature Correlations**

# In[ ]:


if show_plots:
    # plot each feature (y axis) with target (x axis)
    plt.figure(figsize=(30, 190))

    # iterate through feature names and assign to pyplot subplot
    for i,feature_name in enumerate(train.columns.values):
        plt.subplot(19,3,i+1)
        sns.violinplot(y, train[feature_name])
        plt.title(feature_name, fontsize=30)

    plt.show()


# In[ ]:


if show_plots:

    # Compute the correlation matrix
    corr = train.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(110, 90))

    # Generate a custom diverging colormap, use the line below to customize your color options
    # sns.choose_diverging_palette()
    cmap = sns.diverging_palette(8,132,99,50,50,9, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns_heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0)

    # since the heatmap is very large, use following line to save to png for close examination
    # sns_heatmap.get_figure().savefig("corr_heatmap.png")


# Since we have so many features, it's really hard to read those features. There's a line of code in the previous cell to save this graphic to a CSV to take a closer look.
# 
# From this graph we can draw a couple conclusions:
# * A couple of soil types show up as white, because they're actually entirely uniform, and thus useless.
# * Almost no correlation among soil types.
# * There is some correlation among the Hillshade times, as well as Slope and Elevation.

# <a id='train_model'></a>
# # Train Model

# Nothing out of the ordinary here. Split into training and holdout sets, train a model, and plot feature importance. 
# 
# Originally I used Random Forest, but I switched to ExtraTreesClassifier because it generalizes better. I use n_estimators=500 for decent time/performance tradeoff.

# In[ ]:


# split data into train and test sets, using constant random state to better quantify our changes
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=1)


# In[ ]:


# train model
model = ExtraTreesClassifier(n_estimators=500)
model.fit(X_train, y_train)


# In[ ]:


# plot feature importance
plt.figure(figsize=(20,20))
plt.barh(X_train.columns.values, model.feature_importances_)
plt.title('Feature Importance')
plt.ylabel('Feature Name')
plt.xlabel('Gini Value')
plt.show()


# In[ ]:


# make predictions on the cross validation set
y_pred = model.predict(X_test)
n_correct = (y_pred == y_test).sum()
n_total = (y_pred == y_test).count()
print('Accuracy:', n_correct/n_total)


# <a id='error_analysis'></a>
# # Error Analysis

# In[ ]:


# table with data points, truth, and pred
errors = X_test.copy()
errors['truth'] = y_test
errors['pred'] = y_pred
errors = errors[errors['truth'] != errors['pred']]


# In[ ]:


print(errors.shape[0], 'errors over',y_pred.shape[0],'predictions')


# In[ ]:


errors.head()


# In[ ]:


errors.describe()


# Let's try subtracting the descriptive statistics for all train data from those of the errors. This may help us spot trends (features that are underperforming).

# In[ ]:


errors.describe() - train.describe()


# In[ ]:


# x: classes y: # errors
error_truths = []
for cl in target_classes:
    error_count = errors[errors['truth'] == cl]['truth'].count()
    error_truths.append(error_count)
    
plt.bar(target_classes, error_truths)
plt.title('Errors by truth class')
plt.xlabel('True Class')
plt.ylabel('# Errors')
plt.show()


# In[ ]:


# x: classes y: # errors
error_preds = []
for cl in target_classes:
    error_count = errors[errors['pred'] == cl]['pred'].count()
    error_preds.append(error_count)
    
plt.bar(target_classes, error_preds)
plt.title('Errors by predicted class')
plt.xlabel('Predicted Class')
plt.ylabel('# Errors')
plt.show()


# Let's make a confusion matrix to visualize where our errors are coming from.

# In[ ]:


cf_matrix = confusion_matrix(errors['truth'], errors['pred'])

cfm_df = pd.DataFrame(cf_matrix, index = [str(cl)+'t' for cl in target_classes],
                  columns = [str(cl)+'p' for cl in target_classes])

ax = plt.axes()
sns.heatmap(cfm_df, annot=True, fmt='g', ax=ax)
ax.set_title('Error Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


# Looks like most of the errors are because of:
# * Class 1 misclassified as Class 2, and vice versa
# * Class 3 missclassified as Class 6
# * Class 2 misclassified as Class 5

# <a id='make_predictions'></a>
# # Make predictions on test set

# In[ ]:


prediction_classes = pd.Series(model.predict(test.drop('Id', axis=1))).rename('Cover_Type')
predictions = pd.concat([test['Id'], prediction_classes], axis=1).reset_index().drop('index', axis=1)
predictions.to_csv('submission.csv', index=False)
predictions.head()


# <a id='next_steps'></a>
# # Next Steps
# 1. Try to minimize misclassifications between:
#     - class 1 and class 2
#     - class 3 and class 6
# 2. Convert soil type descriptions to categorical features
# 3. Downsample train set to closer resemble feature distribution of test set - especially for elevation, which seems to be one of the most important features, but has the highest mismatch

# In[ ]:




