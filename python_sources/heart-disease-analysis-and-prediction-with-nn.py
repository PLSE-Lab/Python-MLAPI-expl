#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Setting the seed value to get consistent results 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

seed_value= 0

os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

np.random.seed(seed_value)

import tensorflow as tf

tf.random.set_seed(seed_value)


# In[ ]:


from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                        allow_soft_placement=True) 
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


# In[ ]:


#imports

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.compose import ColumnTransformer
from keras import layers
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils import plot_model
from keras.layers import Dropout
from keras.regularizers import l2
from keras.regularizers import l1
#np.random.seed(5)

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# In[ ]:


#Lets take a look at the data
df.head()


# In[ ]:


df.info()


# In[ ]:


#Good news! we have no NAN values in our dataset
df.isnull().sum()


# In[ ]:


#Exploring age distribution
plt.figure(figsize=(15,8))
sns.distplot(df.age,color='#86bf91')


# In[ ]:


labels=['Male','Female']
plt.figure(figsize=(6,6))
plt.pie(df.sex.value_counts(), labels=labels, autopct='%1.1f%%', shadow=True);


# As shown on the pie chart above,males are oversampled in our dataset.

# In[ ]:


#Binning the age data to see how aging affects heart disease
bins = np.linspace(df.age.min(), df.age.max(),8)
bins = bins.astype(int)
df_age = df.copy()
df_age['binned'] = pd.cut(df_age['age'],bins=bins)
df_age.head()


# In[ ]:


#Lets take a look at the target distribution with respect to ages
import seaborn as sns
plt.figure(figsize=(15,8))
sns.countplot(x='binned',hue = 'target',data = df_age,edgecolor=None)


# Except the bin with range 56-63, there are more patients with heart disease than patients with no heart disease. This makes sense considering our dataset consists of people that applied to hospitals because of related symptoms.

# In[ ]:


df_age.binned.value_counts()


# As you can see by the count values of the binned age groups, analyzing the data with counts don't make much sense because some bin counts are lower than the others. For example, we have only 6 people in the bin 70-77 , 29-35 and 89 people in the bin 56-63. Thus, we will be plotting the distrubitions with the percentages from now on.

# Lets split patients into 3 groups by age. We will be categorizing patients into young adults (ages 18-35 years), middle-aged adults (ages 36-55 years), and older adults (aged older than 55 years).

# In[ ]:


bins = [df.age.min(), 35, 55, np.inf]
labels = ['young','middle','older']
df_cat = df.copy()
df_cat['binned'] = pd.cut(df_cat['age'],bins=bins,labels=labels)
df_cat.head()


# Starting our analysis with the feature chest pain type - 'cp'

# In[ ]:



ax = df_cat.groupby('binned')['cp'].value_counts(normalize=True).unstack('cp').plot(kind='bar',figsize=(15,9),rot=0)
ax.set_xlabel("Binned Age Groups");
ax.set_ylabel("CP Percentages");


# As shown on the plot, "0" type chest pain is the most common one among patients in every age group. Another interesting take is that although we do not have any young patients with "2" type chest pain, "2" type chest pain is the second common type of chest pain among middle and older patients

# Now lets look at the effect of different chest pain types to heart disease

# In[ ]:


ax = df_cat.groupby('cp')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)
ax.legend(['Disease Free','Has Disease'])
ax.set_xlabel('Chest Pain Type');
ax.set_ylabel('Percentages');


# It seems that chest pain type is a good indicator for the heart disease since the percentages of patients with heart disease is relatively high for type1,type2 and type3 

# In[ ]:


df_cat.groupby('target')['cp'].value_counts().unstack('cp').plot(kind='bar',figsize=(15,9),rot=0)


# 
# 
# Chest pain type 0 is the most common one among the patients with no heart disease whereas chest pain type 2 is the most common for sick patients

# Enough for chest pain, lets take a look at the other variables. We are always told that high cholestorol is a indicator for the heart diseases. Lets see if our data supports this claim.

# First, we will take a look at the distrubition of serum choloestrol with regard to age.

# In[ ]:


bins = np.linspace(df_cat.chol.min(),df_cat.chol.max(),30)
plt.figure(figsize=(15,8))
sns.distplot(df_cat.chol,bins=bins)


# National Cholesterol Education Program (NCEP) guidelines provide specific numbers for cholesterol ranges:
# Normal: less than 200 mg/dL
# Borderline high: 200 to 239 mg/dL
# High: 240 mg/dL or above
# 
# Lets bin our data according to this.
# 

# In[ ]:



    
bins = [df_cat.chol.min(), 200, 239,np.inf]
labels = ['Normal','Borderline','High']
df_chol = df_cat.copy(deep=True)
df_chol['chol_bin'] = pd.cut(df_cat['chol'],bins = bins,labels = labels)
plt.figure(figsize=(15,8))
sns.countplot(x = 'chol_bin',data=df_chol,hue='target')


# Interesting, according to our data there is no relationship between cholestoral  levels and a heart disease. Keep in mind that this a small sample size and we cannot make concrete deductions about the whole population. At least we can conclude that cholestoral probably will not be a very important variable for our prediction model. We will come back to this later.

# Lets continue our analysis with the another variables. Well I cheated here a little bit and ask my sister who is a doctor about which variables do she think that most related to heart diseases. Her answer was ca(number of major vessels (0-3) colored by flourosopy),thalassemia(a type of blood disorder) and maximum heart rate . Domain knowledge always comes in handy. So, we will continue our analysis with the variable 'ca'.

# In[ ]:


ax = df.groupby('ca')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)
ax.legend(['Disease Free','Has Disease'])
ax.set_xlabel('# of major vessels (0-4) colored by flourosopy',fontdict={'fontsize':14});
ax.set_ylabel('Percentages',fontdict={'fontsize':14});


# Ca level 0 and 4 has the most patients with a heart disease. In addition, level of differences for each ca level is relatively big compared to other distributions. Maybe my sister was right and number of major vessels could be a good indicator for heart disease.

# Another issue I feel a need to mention is that as seen below, not all types of ca is sampled equally in our dataset. We have 5 patients with ca level 4 whereas we have 175 people with ca level 0.

# In[ ]:


df.ca.value_counts()


# Let's continue with the analysis of thalemesia named 'thal' in our dataset.

# In[ ]:


ax = df.groupby('thal')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)
ax.legend(['Disease Free','Has Disease'])
ax.set_xlabel('Thalemesia')
ax.set_ylabel = ('Percentages')


# Especially thal level 2 seems to be a good indicator since this value is not undersampled in our dataset with a count of 166!

# In[ ]:


df.thal.value_counts()


# Now on to the variable maximum heart rate called 'thalach' in our dataset.

# Normal maximum heart rate differs with the age of the patient. Thus we will be categorizing the heart rates of patients according to their ages. 
# 
# Young patients -> normal < 200 < High
# 
# Middle aged patients -> normal < 180 < High
# 
# Older patients -> normal < 170 < high
# 

# In[ ]:


#First define the limit for normal heart rate limit for each patient according to their age category

df_t = df_cat.copy(deep=True)
df_t.loc[df_t.binned=='young','hr_bin'] = 200
df_t.loc[df_t.binned=='middle','hr_bin'] = 185
df_t.loc[df_t.binned=='older','hr_bin'] = 160
df_t.head()


# In[ ]:


#Then categorizing the heart rate category as Normal or High in thalach_bin

df_t['thalach_bin'] = np.where(df_t.eval("thalach <= hr_bin "), "Normal", "High")
df_t


# In[ ]:


#grouping df_t to get the counts of the patients in each group

df_thalach = df_t.groupby(['thalach_bin','target','binned']).count()
df_thalach


# In[ ]:




#Dividing inital values for each age_binned group by summed up  entries per each age_binned group to get percentages

df_thalach.iloc[[0,3,6,9]]/= df_thalach.iloc[[0,3,6,9]].age.sum()
df_thalach.iloc[[1,4,7,10]]/= df_thalach.iloc[[1,4,7,10]].age.sum()
df_thalach.iloc[[2,5,8,11]]/= df_thalach.iloc[[2,5,8,11]].age.sum()

df_thalach = df_thalach.reset_index(level='target')
df_thalach = df_thalach.reset_index(level='binned')
df_thalach = df_thalach[['age','target','binned']]
df_thalach.rename(columns={'age':'density'},inplace=True)
df_thalach


# In[ ]:


#df_thalach.reset_index(level='thalach_bin',inplace=True)

df_thalach = df_thalach.reset_index()
plt.figure(figsize=(15,8));
g = sns.FacetGrid(df_thalach, col = 'binned', height=5, aspect=1.2,hue='target');
g.map(sns.barplot,  "thalach_bin", "density",alpha=0.6);
plt.legend();


# For young aged group, we have only patients with high max heart rate and percentage of patients with disease is higher for this category. Also, heart disease is less common for older patients with a Normal maximum heart rate compared to High maximum heart rate in the same age group.

# Next feature we will investigate is the Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria) called 'restecg' in our dataset

# In[ ]:


ax = df.groupby('restecg')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)
ax.legend(['Disease Free','Has Disease']);
ax.set_xlabel('Resting Electrocardiographic Results');
ax.set_ylabel('Percentages');


# Heart disease is much more common among patients with restecg measured 1 

# In[ ]:


df.restecg.value_counts()


# For resting blood pressure, we need to bin the blood pressure values. Although, normal rate for blood pressure differs slighltly with age, I will give constant values for blood pressure bin edges as follows:
# 
# Low < 80 < Normal < 120 < High

# In[ ]:


def trestbps_bin(row):
    
    if row['trestbps'] <= 80:
        value = 'Low'
    
    elif row['trestbps'] > 120:
        value = 'High'
    else:
        value = 'Normal'
        
    return value


# In[ ]:


df_trestbps = df.copy()
df_trestbps['trestbps_bin'] = df.apply(trestbps_bin,axis=1)


df_trestbps


# In[ ]:


ax = df_trestbps.groupby('trestbps_bin')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)
ax.legend(['Disease Free','Has Disease'])
ax.set_xlabel('Blood Pressure Bin');
ax.set_ylabel('Percentages');


# 
# Sick patients and healthy patients are nearly equally distributed for high blood pressure levels. Interestingly, for normal blood pressure levels percentage of patients without a heart disease is larger. I don't think this feature will be that important to our models predictions

# In[ ]:


df_trestbps.trestbps_bin.value_counts()


# Last but not least, we will be investigating the feature Exercise Induced Angina called 'exang' in out dataset.

# In[ ]:


ax = df.groupby('exang')['target'].value_counts(normalize=True).unstack('target').plot(kind='bar',figsize=(15,9),rot=0)
ax.legend(['Disease Free','Has Disease'])
ax.set_xlabel('Exercise Induced Angina');
ax.set_ylabel('Percentages');


# Exang looks like a good indicator for a heart disease since for both type 0 and type 1 difference between sick people and healthy people is high.

# TIME FOR PREDICTION!

# We will start by applying one hot encoding to categorical features and standard scaling to numerical features.Normally, these transformations produce numpy arrays instead of pandas dataframe. I will be changing to code a little bit to make the outputs of these transformations pandas dataframe in order to increase the readilibity. After evulation process is complete, generated data frames will be input to the Permuation Importance to figure out which features are most important for our model. Let's begin.

# In[ ]:


#first seperate categorical and numerical data to apply different transformations

y = df.target
df_transformed = df.copy(deep=True)
numeric_features = ['age','trestbps','chol','thalach','ca','oldpeak']
categorical_features = ['sex','cp','fbs','restecg','exang','slope','thal']

enc = OneHotEncoder(sparse=False,drop='first')
enc.fit(df_transformed[categorical_features])

col_names = enc.get_feature_names(categorical_features)
df_transformed = pd.concat([df_transformed.drop(categorical_features, 1),
          pd.DataFrame(enc.transform(df_transformed[categorical_features]),columns = col_names)], axis=1).reindex()
df_transformed.head()


# Categorical features of the original dataframe are one hot encoded.Let's move on to the continuous features

# In[ ]:


scaler = StandardScaler()


df_transformed[numeric_features]  = scaler.fit_transform(df_transformed[numeric_features])
df_transformed.head()


# Our dataframe is ready to feed into ML model

# In[ ]:


#Split the data 
X_train, X_test, y_train, y_test = train_test_split(df_transformed.drop('target',axis=1), df_transformed['target'], test_size = .2, random_state=10)
rf_model = RandomForestClassifier(max_depth=5, random_state=137)
rf_model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix
def conf_matrix(X_test,y_test,model):
    
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)
    
    


# In[ ]:


conf_matrix(X_test,y_test,rf_model)


# In[ ]:


from sklearn.metrics import accuracy_score
y_test_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_test_pred)


#  We have %80 test set accuracy with a Random Forest Classifier. Not bad. Let's try to improve this by tuning hyperparameters of the model.

# In[ ]:


from sklearn.model_selection import GridSearchCV
n_estimators = [10, 30, 50, 100]
max_depth = [5, 8, 15]
min_samples_split = [2, 5, 10, 15, 40]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(rf_model, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)


# In[ ]:


gridF.best_params_


# In[ ]:


hp_list = gridF.best_params_

improved_model = RandomForestClassifier(max_depth=hp_list['max_depth'], min_samples_leaf = hp_list['min_samples_leaf'],
                                        min_samples_split = hp_list['min_samples_split'], n_estimators = hp_list['n_estimators'],random_state = 137)


# In[ ]:


improved_model.fit(X_train, y_train)


# In[ ]:


conf_matrix(X_test,y_test,improved_model)


# In[ ]:


y_test_pred = improved_model.predict(X_test)
accuracy_score(y_test, y_test_pred)


# We managed to improve accuracy on the test set by %3 with fine tuning the hyperparameters

# In[ ]:



def model_nn(input_shape):
    

    # Define the input placeholder as a tensor with shape input_shape
    X_input = Input(input_shape)

  
    X = Dense(512,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform')(X_input)
    X = Activation('relu')(X)
    X = Dropout(0.5,seed=10)(X)
    X = Dense(256,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform')(X)
    X = Activation('relu')(X)
    X = Dropout(0.25,seed=10)(X)
    X = Dense(16,kernel_regularizer=l2(0.01),kernel_initializer = 'random_uniform')(X)
    X = Activation('relu')(X)
    X = Dropout(0.25,seed=10)(X)
    X = Dense(1, activation='sigmoid')(X)

    # Create model. 
    model = Model(inputs = X_input, outputs = X, name='nnModel')

    return model


# In[ ]:


nnModel = model_nn(X_train.shape)


# In[ ]:


nnModel.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
nnModel.fit(x = X_train, y = y_train, epochs = 100, batch_size = 16)


# In[ ]:


preds = nnModel.evaluate(x= X_test, y=y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# With the help of neural networks we managed the increase test set accuracy by %2

# Lets try the pin down which features are the most important for our model.

# In[ ]:


df_transformed.head()


# This part is where generating the transformed output as a dataframe comes handy. Weights of Permutation Importance are linked to the columns of the transformed dataframe. Permutation Importance simply shuffles each column in itself and calculates how much loss function suffered from reordering of the column.

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(rf_model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm,feature_names = X_test.columns.tolist())


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(improved_model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm,feature_names = X_test.columns.tolist())


# This actually matches the analysis done in the beginning. We predicted by looking at the distributions that thal, exang,ca,cp and thalach will be good indicators for heart disease and also predicted that chol, trestbps are not very good indicators. Output of permutation importance lies perfectly with our analysis!
