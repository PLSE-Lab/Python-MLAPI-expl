#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE 
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from seaborn import countplot,lineplot, barplot
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Load the datasets
df_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')
df_test = pd.read_csv('../input/X_test.csv')


# In[ ]:


df_train.describe()


# In[ ]:


df_test.describe()


# In[ ]:


y_train.describe()


# In[ ]:


# Check for missing values
print(df_train.isnull().sum())
print(df_test.isnull().sum())
print(y_train.isnull().sum())


# In[ ]:


# Count of target classes 
sns.set(style='whitegrid')
sns.countplot(y = 'surface',
              data = y_train,
              order = y_train['surface'].value_counts().index)
plt.show()


# In[ ]:


# Every 128 datapoints are regarded as one serie. Lets visulaize how the data points look in one serie
series1 = df_train.head(128)
plt.figure(figsize=(26, 16))
for i, col in enumerate(series1.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(series1[col])
    plt.title(col)


# In[ ]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df_train.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap="Greens")


# In[ ]:


f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(df_test.iloc[:,3:].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap="Greens")


# In[ ]:


# Encode the labels for traget predctions
from sklearn.preprocessing import LabelEncoder

# encode class values as integers so they work as targets for the prediction algorithm
encoder = LabelEncoder()
y = encoder.fit_transform(y_train["surface"])
y_count = len(list(encoder.classes_))


# In[ ]:


label_mapping = {i: l for i, l in enumerate(encoder.classes_)}


# In[ ]:


df_train["target"] = y.repeat(128)


# In[ ]:


y_train.shape


# In[ ]:


def plot_robot_series(series_id):
    robot_series_data = df_train[df_train["series_id"] == series_id]
    orientation_data = robot_series_data[["orientation_X", "orientation_Y", "orientation_Z"]]
    angular_data = robot_series_data[["angular_velocity_X", "angular_velocity_Y", "angular_velocity_Z"]]
    linear_data = robot_series_data[["linear_acceleration_X", "linear_acceleration_Y", "linear_acceleration_Z"]]
    surface = robot_series_data["target"].iloc[0]
    surface = label_mapping[surface]

    fig, axs = plt.subplots(figsize=(15,3), nrows=1, ncols=3)
    axs[0].plot(orientation_data)
    axs[0].set_title(surface+": orientation XYZ")
    axs[0].legend(("X", "Y", "Z"), loc="upper left")
    axs[1].plot(angular_data)
    axs[1].set_title(surface+": angular velocity")
    axs[1].legend(("X", "Y", "Z"), loc="upper left")
    axs[2].plot(linear_data)
    axs[2].set_title(surface+": linear acceleration")
    axs[2].legend(("X", "Y", "Z"), loc="upper left")
    plt.show()


# In[ ]:


for key in label_mapping:
    rows = df_train[df_train["target"] == key]
    #find the first row with this surface type
    row = df_train.index.get_loc(rows.iloc[0].name)
    sid = df_train.iloc[row]["series_id"]
    plot_robot_series(sid)
    #print(row)


# In[ ]:


# Conversion of quaternion to Euler angles
def quaternion_to_euler(x, y, z, w):
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


# In[ ]:


def fe_step0 (actual):
    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)
    actual['mod_quat'] = (actual['norm_quat'])**0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']
    
    return actual


# In[ ]:


df_train = fe_step0(df_train)
df_test = fe_step0(df_test)
df_test.head()


# In[ ]:


# Visualize the quaterions
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18, 5))

ax1.set_title('quaternion X')
sns.kdeplot(df_train['norm_X'], ax=ax1, label="train")
sns.kdeplot(df_test['norm_X'], ax=ax1, label="test")

ax2.set_title('quaternion Y')
sns.kdeplot(df_train['norm_Y'], ax=ax2, label="train")
sns.kdeplot(df_test['norm_Y'], ax=ax2, label="test")

ax3.set_title('quaternion Z')
sns.kdeplot(df_train['norm_Z'], ax=ax3, label="train")
sns.kdeplot(df_test['norm_Z'], ax=ax3, label="test")

ax4.set_title('quaternion W')
sns.kdeplot(df_train['norm_W'], ax=ax4, label="train")
sns.kdeplot(df_test['norm_W'], ax=ax4, label="test")

plt.show()


# In[ ]:


# Quaternions to Euler Angles
def fe_step1 (actual):    
    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual


# In[ ]:


df_train = fe_step1(df_train)
df_test = fe_step1(df_test)
df_test.head()


# In[ ]:


# Feature engineering
def feat_eng(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df


# In[ ]:


df_train = feat_eng(df_train)
df_test = feat_eng(df_test)


# In[ ]:


# Fill the missing values
df_train.fillna(0,inplace=True)
df_test.fillna(0,inplace=True)
df_train.replace(-np.inf,0,inplace=True)
df_train.replace(np.inf,0,inplace=True)
df_test.replace(-np.inf,0,inplace=True)
df_test.replace(np.inf,0,inplace=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_train, y_train["surface"], test_size = 0.3, random_state = 42, stratify = y_train["surface"], shuffle = True)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[ ]:


def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize=True,                  print_cm=True, cm_cmap=plt.cm.Greens):
    
    # to store results at various phases
    results = dict()
    
    # time at which model starts training 
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n \n')
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    
    
    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    print(X_train.shape, X_test.shape)
    y_pred = model.predict(X_test)
    # prediction = model.predict(test_data)
    #y_pred = np.argmax(y_pred, axis=1)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred
   
    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    
    
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    #plt.figure(figsize=(8,8))
    #plt.grid(b=False)
    #plot_confusion_matrix(cm, classes=class_labels, normalize=True, title='Normalized confusion matrix', cmap = cm_cmap)
    #plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classification Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results


# In[ ]:


labels = ['fine_concrete', 'concrete', 'soft_tiles', 'tiled', 'soft_pvc', 'hard_tiles_large_space', 'carpet', 'hard_tiles', 'wood']


# In[ ]:


rfc=RandomForestClassifier(n_estimators=200, max_depth=5,
                             random_state=0)
rfc_results =  perform_model(rfc, X_train, y_train, X_test, y_test, class_labels = labels)


# In[ ]:


dtc = DecisionTreeClassifier()
dtc_results =  perform_model(dtc, X_train, y_train, X_test, y_test, class_labels = labels)

