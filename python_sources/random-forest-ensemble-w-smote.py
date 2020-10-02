#!/usr/bin/env python
# coding: utf-8

# # Robot Ensemble Prediction: Kaggle CareerCon 2019
# 
# Credits to jesucristo on Kaggle for their wonderful public kernel (https://www.kaggle.com/jesucristo/1-smart-robots-complete-notebook-0-73) that was instrumental in me learning the ropes; various bits of exploration graphing and the general format of the notebook are greatly inspired by the work there.
# 
# In this notebook, I will be using an Ensemble method prediction after creating derivative features such as normalized parameters, Euler angles, and dot and cross products then balancing the surface classes using SMOTE.
# 
# NOTE: Through iteration, I've found the model to be slightly improved when I don't include dot and cross products, and only use orientation-based parameters. SMOTE resampling seems to (very slightly) improve outcomes, though the difference seems insignificant.
# 
# ## Setup
# 
# #### Initialize libraries and outside packages

# In[52]:


# Helper libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Sci-kit learn libraries
import sklearn as sk
from sklearn import ensemble
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Imbalanced-learn libraries
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE


# #### Load in helper functions, these are quite lengthy!

# In[53]:


# Helper functions, used much later on

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    # Retrieved from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def quaternion_to_euler(x, y, z, w):
    # Retrieved from https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
    # Returns a radian measurement for roll, pitch, and yaw
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def normalize(data):
    # Normalizes the direction-dependent parameters for an input dataset 'data'
    # Specifically, creates unit vectors for orientation, velocity, and acceleration
    data['mod_quat'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2 + data['orientation_W']**2)**.5
    data['norm_orientation_X'] = data['orientation_X']/data['mod_quat']
    data['norm_orientation_Y'] = data['orientation_Y']/data['mod_quat']
    data['norm_orientation_Z'] = data['orientation_Z']/data['mod_quat']
    data['norm_orientation_W'] = data['orientation_W']/data['mod_quat']
    
    data['mod_angular_velocity'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)**.5
    data['norm_velocity_X'] = data['angular_velocity_X']/data['mod_angular_velocity']
    data['norm_velocity_Y'] = data['angular_velocity_Y']/data['mod_angular_velocity']
    data['norm_velocity_Z'] = data['angular_velocity_Z']/data['mod_angular_velocity']
    
    data['mod_linear_acceleration'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**.5
    data['norm_acceleration_X'] = data['linear_acceleration_X']/data['mod_linear_acceleration']
    data['norm_acceleration_Y'] = data['linear_acceleration_Y']/data['mod_linear_acceleration']
    data['norm_acceleration_Z'] = data['linear_acceleration_Z']/data['mod_linear_acceleration']
    return data

def add_euler_angles(data):
    # Derives Euler angles from the quaternion for an input dataset 'data'
    # *Requires normalized quaternion orientations first*
    x = data['norm_orientation_X'].tolist()
    y = data['norm_orientation_Y'].tolist()
    z = data['norm_orientation_Z'].tolist()
    w = data['norm_orientation_W'].tolist()
    eX, eY, eZ = [],[],[]
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        eX.append(xx)
        eY.append(yy)
        eZ.append(zz)
    data['euler_X'] = eX
    data['euler_Y'] = eY
    data['euler_Z'] = eZ
    return data

def add_direction_vectors(data):
    # Derives unit direction vectors from Euler angles in dataset 'data'
    roll = data['euler_X'].tolist()
    pitch = data['euler_Y'].tolist()
    yaw = data['euler_Z'].tolist()
    uX, uY, uZ = [],[],[]
    for i in range(len(roll)):
        xx = math.cos(yaw[i])*math.cos(pitch[i])
        yy = math.sin(yaw[i])+math.cos(pitch[i])
        zz = math.sin(pitch[i])
        uX.append(xx)
        uY.append(yy)
        uZ.append(zz)
    data['orientation_vector_X'] = uX
    data['orientation_vector_Y'] = uY
    data['orientation_vector_Z'] = uZ
    return data

def eng_data(data):
    # Creates engineered features within dataset 'data'
    # Intended for use on the raw X data
    
    # Idea 1: Ratios
    data['ratio_velocity-acceleration'] = data['mod_angular_velocity'] / data['mod_linear_acceleration']
    
    # Idea 2: 
    
    return data

def descriptive_features(features, data, stats):
    # Creates descriptive statistics such as max, min, std. dev, mean, median, etc. from
    # features 'stats' in dataset 'data' and stores these in 'features'
    for col in data.columns:
        if col not in stats:
            continue
        # Base statistics
        colData = data.groupby(['series_id'])[col]
        features[col + '_min'] = colData.min()
        features[col + '_max'] = colData.max()
        features[col + '_std'] = colData.std()
        features[col + '_mean'] = colData.mean()
        features[col + '_median'] = colData.median()
        
        # Derivative statistics
        features[col + '_range'] = features[col + '_max']-features[col + '_min']
        features[col + '_maxOverMin'] = features[col + '_max']/features[col + '_min']
        features[col + '_mean_abs_chg'] = colData.apply(lambda x: 
                                                        np.mean(np.abs(np.diff(x))))
        features[col + '_abs_max'] = colData.apply(lambda x: 
                                                   np.max(np.abs(x)))
        features[col + '_abs_min'] = colData.apply(lambda x: 
                                                   np.min(np.abs(x)))
        features[col + '_abs_avg'] = (features[col + '_abs_min'] 
                                      + features[col + '_abs_max'])/2
    return features

def eng_features(features):
    # Creates engineered features within dataset 'features'
    # Intended for use on the modified X data
    
    # Idea 1: Dot and cross products of unit direction vectors
    # Note: np.dot and np.cross are very slow to perform on large sets of data,
    # minimize iterations used of them
    stat = '_mean'
    Ox = features['orientation_vector_X' + stat]
    Oy = features['orientation_vector_Y' + stat]
    Oz = features['orientation_vector_Z' + stat]
    Vx = features['norm_velocity_X' + stat]
    Vy = features['norm_velocity_Y' + stat]
    Vz = features['norm_velocity_Z' + stat]
    Ax = features['norm_acceleration_X' + stat]
    Ay = features['norm_acceleration_Y' + stat]
    Az = features['norm_acceleration_Z' + stat]
    
    oDv,oDa,vDa = [],[],[]
    oCv_x,oCv_y,oCv_z = [],[],[]
    oCa_x,oCa_y,oCa_z = [],[],[]
    vCa_x,vCa_y,vCa_z = [],[],[]
    for i in range(len(Ox)):
        oDv.append(np.dot([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]]))
        oCv = np.cross([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]])
        oCv_x.append(oCv[0])
        oCv_y.append(oCv[1])
        oCv_z.append(oCv[2])
        oDa.append(np.dot([Ox[i],Oy[i],Oz[i]],[Ax[i],Ay[i],Az[i]]))
        oCa = np.cross([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]])
        oCa_x.append(oCa[0])
        oCa_y.append(oCa[1])
        oCa_z.append(oCa[2])
        vDa.append(np.dot([Vx[i],Vy[i],Vz[i]],[Ax[i],Ay[i],Az[i]]))
        vCa = np.cross([Ox[i],Oy[i],Oz[i]],[Vx[i],Vy[i],Vz[i]])
        vCa_x.append(vCa[0])
        vCa_y.append(vCa[1])
        vCa_z.append(vCa[2])
        
    features['orientation_dot_velocity'] = oDv
    features['orientation_cross_velocity_X'] = oCv_x
    features['orientation_cross_velocity_Y'] = oCv_y
    features['orientation_cross_velocity_Z'] = oCv_z
    features['orientation_dot_acceleration'] = oDa
    features['orientation_cross_acceleration_X'] = oCa_x
    features['orientation_cross_acceleration_Y'] = oCa_y
    features['orientation_cross_acceleration_Z'] = oCa_z
    features['velocity_dot_acceleration'] = vDa
    features['velocity_cross_acceleration_X'] = vCa_x
    features['velocity_cross_acceleration_Y'] = vCa_y
    features['velocity_cross_acceleration_Z'] = vCa_y
    
    return features


# ## Data Exploration
# 
# Load in data and analyze its composition

# In[54]:


x_test = pd.read_csv('../input/X_test.csv')
x_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
submission = pd.read_csv('../input/sample_submission.csv')
print('Train X: {}\nTrain Y: {}\nTest X: {}\nSubmission: {}'.format(x_train.shape,y_train.shape,x_test.shape,submission.shape))


# In[55]:


x_train.head()


# In[56]:


x_train.describe()


# In[57]:


y_train.head()


# In[58]:


y_train.describe()


# In[59]:


x_test.head()


# In[60]:


x_test.describe()


# In[61]:


submission.head()


# In[62]:


submission.describe()


# ### Visualize data composition

# First, we want to determine the distribution of the surfaces and group_ids in order to alter our cross-validation methodology accordingly

# In[63]:


sns.set(style='darkgrid')
sns.countplot(y = 'surface',
              data = y_train,
              order = y_train['surface'].value_counts().index)
plt.show()


# In[64]:


plt.figure(figsize=(30,10)) 
sns.set(style="darkgrid",font_scale=1.5)
sns.countplot(x="group_id", data=y_train, order = y_train['group_id'].value_counts().index)
plt.show()


# Next, we want to determine feature correlation in the X train and test sets in order to drop heavily correlated variables, as these will likely overfit the model.

# In[65]:


f,ax = plt.subplots(figsize=(8, 8))
plt.title("X Train")
sns.heatmap(x_train.iloc[:,3:].corr(), annot=True, linewidths=1.5, fmt= '.2f', annot_kws={"size": 10}, ax=ax)


# In[66]:


f,ax = plt.subplots(figsize=(8, 8))
plt.title("X Test")
sns.heatmap(x_test.iloc[:,3:].corr(), annot=True, linewidths=1.5, fmt= '.2f', annot_kws={"size": 10}, ax=ax)


# From these correlation matrices, we can see that the Y and Z orientations and X and W orientations are perfectly correlated. The angular velocity and linear accelerations in directions Y and Z are each also highly correlated, dropping the features associated with one of these directions may prove beneficial later on.

# ### Feature Selection
# 
# Conduct feature selection on x_train and x_test data for use in model and predictions

# First, we want to convert the quarternion into Euler angles and normalize the parameters (Refer to helper functions file)

# In[67]:


x_train = normalize(x_train)
x_train = add_euler_angles(x_train)
x_test = normalize(x_test)
x_test = add_euler_angles(x_test)


# Let's visualize the Euler angles for each set of X data:

# In[68]:


fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(15,5))

ax1.set_title('Roll')
sns.kdeplot(x_train['euler_X'], ax=ax1, label='train')
sns.kdeplot(x_test['euler_X'], ax=ax1, label='test')

ax2.set_title('Pitch')
sns.kdeplot(x_train['euler_Y'], ax=ax2, label='train')
sns.kdeplot(x_test['euler_Y'], ax=ax2, label='test')

ax3.set_title('Yaw')
sns.kdeplot(x_train['euler_Z'], ax=ax3, label='train')
sns.kdeplot(x_test['euler_Z'], ax=ax3, label='test')

plt.show()


# Now we want to create derivative features based on these existing variables and store them in the feature sets (in order to convert from the number of instances in the raw X datasets to the number of Y instances)

# In[69]:


train_features = pd.DataFrame()
test_features = pd.DataFrame()


# In[70]:


x_train = add_direction_vectors(x_train)
x_test = add_direction_vectors(x_test)
x_train.head()


# In[71]:


stats = ['norm_orientation_X', 'norm_orientation_Y', 'norm_orientation_Z', 
         'norm_orientation_W',
         'norm_velocity_X', 'norm_velocity_Y', 'norm_velocity_Z', 
         'norm_acceleration_X', 'norm_acceleration_Y', 'norm_acceleration_Z',
         'mod_linear_acceleration',
         'ratio_velocity-acceleration', 
         'orientation_vector_X', 'orientation_vector_Y', 'orientation_vector_Z']
train_features = descriptive_features(train_features, x_train, stats)
test_features = descriptive_features(test_features, x_test, stats)


# After generating these descriptive features, we want to begin engineering custom parameters. So far, I've added dot and cross products of the mean direction vector values. While using the mean vector values will run into all sorts of arithmetic issues (namely not generating a unit vector), it would the most representative of the simple statistical measures I've used.

# In[72]:


train_features = eng_features(train_features)
test_features = eng_features(test_features)
print("Train features: ", train_features.shape)
print("Test features: ", test_features.shape)


# In[73]:


train_features.head()


# In[74]:


test_features.head()


# The results suggest that the new features were constructed as intended, we now want to narrow down the most highly correlated features to simplify the model. We will examine the correlations on the training feature set and apply any changes to both sets of features

# In[75]:


#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
corr_matrix = train_features.corr().abs()
raw_corr = train_features.corr()

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))
top_corr = pd.DataFrame(sol).reset_index()
top_corr.columns = ["var1", "var2", "abs corr"]
# with .abs() we lost the sign, and it's very important.
for x in range(len(top_corr)):
    var1 = top_corr.iloc[x]["var1"]
    var2 = top_corr.iloc[x]["var2"]
    corr = raw_corr[var1][var2]
    top_corr.at[x, "raw corr"] = corr


# In[76]:


top_corr.head(15)


# In[77]:


top_corr.tail(15)


# Possible drops to reduce model complexity will be made here, although I experienced better performance :

# In[78]:


drops = []
# Takes the most correlated features out above a given threshold
threshold = .95
for i in range(len(top_corr)):
    if top_corr.iloc[i]["raw corr"] > threshold:
        if top_corr.iloc[i]["var1"] not in drops and top_corr.iloc[i]["var2"] not in drops:
            drops.append(top_corr.iloc[i]["var2"])
train_features = train_features.drop(columns=drops)
test_features = test_features.drop(columns=drops)


# ### Class Balancing

# We saw that the surface classes were very heavily imbalanced, and so to solve this I will use SMOTE (Synthetic Minority Over-sampling Technique) to balance them by creating resampled training sets, to be used in the final fitting. I still will retain the original data to run a separate cross-validation on and to easily switch to if desired.
# 
# The sampling strategy I intended is to set all classes to be of the same quantity as the majority class, alter "sampling_strategy" parameter if you wish to change this

# In[79]:


seed = 1920348
sm = SMOTE(sampling_strategy='not majority', random_state=seed, k_neighbors=10)
y_train_rs = pd.DataFrame()
train_features_rs, y_train_rs['surface'] = sm.fit_resample(train_features, y_train['surface'])


# Chart of the data to confirm that the resampling worked as intended:

# In[80]:


sns.set(style='darkgrid')
sns.countplot(y = 'surface',
              data = y_train_rs,
              order = y_train_rs['surface'].value_counts().index)
plt.show()


# In[81]:


train_features.head()


# In[82]:


y_train.head()


# In[83]:


test_features.head()


# ## Classification
# 
# Here I run the ensemble on both versions of training data, with a bagging classifier currently in use due to overfitting issues

# #### Cross-validation results for original data:

# In[84]:


# Choose ensemble method used
#model1 = ensemble.RandomForestClassifier(n_estimators=75)
#model2 = ensemble.BaggingClassifier(n_estimators=25)
#model3 = ensemble.ExtraTreesClassifier(n_estimators=25)
model = ensemble.RandomForestClassifier(n_estimators=75)

# create the ensemble model
seed = 89175915
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(model, train_features, 
                    y_train['surface'], cv=kfold)
for i in range(len(results)):
    print("Fold", i+1, "score: ", results[i])
print("Cross-validation score average on original data: ", results.mean())


# #### Feature importances for original data

# In[85]:


model.fit(train_features, y_train['surface'])
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

feature_importances = pd.DataFrame(importances, index = train_features.columns, columns = ['importance'])
feature_importances.sort_values('importance', ascending = False).plot(kind = 'bar',
                        figsize = (35,8), color = 'r', yerr=std[indices], align = 'center')
plt.xticks(rotation=90)
plt.show()


# #### Cross-validation results for resampled data:

# In[86]:


# create the ensemble model
seed = 666839274
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
model_rs = ensemble.RandomForestClassifier(n_estimators=75)
results = model_selection.cross_val_score(model_rs, train_features, 
                    y_train['surface'], cv=kfold)
for i in range(len(results)):
    print("Fold ", i+1, "score: ", results[i])
print("Cross-validation score average on resampled data: ", results.mean())


# #### Feature importances for resampled data

# In[87]:


model_rs.fit(train_features, y_train['surface'])
importances = model_rs.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

feature_importances = pd.DataFrame(importances, index = train_features.columns, columns = ['importance'])
feature_importances.sort_values('importance', ascending = False).plot(kind = 'bar',
                        figsize = (35,8), color = 'r', yerr=std[indices], align = 'center')
plt.xticks(rotation=90)
plt.show()


# The following code allows you to toggle between the original and resampled versions of the data, simply change "resample" to True or False as desired. In hindsight, resampling doesn't appear to do much good as currently set up.

# In[88]:


resample = True
if resample:
    x, y = train_features_rs, y_train_rs['surface']
    final_model = model_rs
else:
    x, y = train_features, y_train['surface']
    final_model = model


# In[89]:


sk.metrics.confusion_matrix(y,final_model.predict(x), y.unique())


# My confusion matrix currently suggests overfitting, likely due to too many features.

# Now we will create the final submission on the test data, first ensuring that the file is formatted as expected

# In[90]:


submission['surface'] = final_model.predict(test_features)
submission.head()


# In[91]:


print("Final submission file has dimensions: ", submission.shape)


# It is important to set "index = False" when creating the file, else you will wind up with an unwanted header column.

# In[92]:


submission.to_csv('final_submission_smote.csv', index = False)


# #### The end! Let me know if you have any comments or suggestions about the solution
