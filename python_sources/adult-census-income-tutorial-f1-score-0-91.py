#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income
# **Predict whether income exceeds $50K/yr based on census data**  
# 
# 
# ---
# 
# 
# My objective of creating this kernel is to see how **neural network** performs on this dataset as compared to relatively simpler methods (as tried in many kernels) and also to try out how **GridSearch** method with **Keras** using **Keras Wrapper for Scikit-learn**
# 
# 
# **Contents:**
# 1. [Data Exploration and Visualization](#Data-Exploration-(EDA))
# 2. [Data Preprocessing](#Data-Preprocessing)
# 3. [Modeling](#Modeling-Part)
# 4. [Predictions](#Predictions)
# 5. [Final Comments](#Conclusion/Comments)
# 

# <u>Let's get Started!</u>
# 
# Importing required modules

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.regularizers import l2
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Setting up seed for reproducible results

# In[ ]:


np.random.seed(42)
tf.random.set_seed(42)


# Reading data file

# In[ ]:


data_all = pd.read_csv("/kaggle/input/adult-census-income/adult.csv")


# Take a peek at the data, see what columns are present and the data types, also check if headers are picked correctly

# In[ ]:


data_all.head()


# Hmm... looks like we have '?' character in data for missing values, we will need to replace that

# Checking how many (Rows, Columns) are present in data

# In[ ]:


data_all.shape


# replacing '?' with NaN for now, it would be easier to fill NaN later with other resonable estimates

# In[ ]:


data_all.replace('?',np.nan, inplace=True)


# Checking number of classes and class distribution, we should check if our classes are balanced or skewed, it would help us choose correct performance metric for evaluating model

# In[ ]:


pd.value_counts(data_all['income'])/data_all['income'].count()*100


# We can see that there are **two classes** the **classes are skewed.**
# 
# Will choose <b>F1-Score</b> (Precision and Recall) as the performance metric for our model

# Let's look at data characteristics in bit more detail, and look for potential outliers in the data

# In[ ]:


data_all.describe(include="all").T


# Looks like Workclass, Occupation and native.country has NaN values  
# We will use mode (most occuring) as imputing method to fill these NaNs  
# 
# **Age** seem alright, minimum is 17 and maximum is 90 (who's still working at 90?)  
# **hours.per.week** - minimum is only 1 ? We will check this.. could be an outlier if salary is >50K

# In[ ]:


data_all[(data_all['hours.per.week'] <= 4) & (data_all['income'] == '>50K')].sort_values(by='education.num')


# Well at-least people earning > 50 K are highly educated (do probably not outliers), also looks like their occupations in 'execs' or 'profs'

# let's fill all missing values with `mode`

# In[ ]:


# fill values
print(data_all['workclass'].mode())
print(data_all['occupation'].mode())
print(data_all['native.country'].mode())
data_all['workclass'].fillna(data_all['workclass'].mode()[0], inplace=True)
data_all['occupation'].fillna(data_all['occupation'].mode()[0], inplace=True)
data_all['native.country'].fillna(data_all['native.country'].mode()[0], inplace=True)


# Final check if we still have any NaN's in our data

# In[ ]:


data_all.info()


# # Data Exploration (EDA)

# Now we will do some basic **EDA**, we will visualize independent variable against dependent and try to determine some relationships. We will look out for any outliers or anything out of ordinary. We will also do some **feature engineering** steps (clubbing similar classes, therby reducing features dimension)  
# 
# This section is a bit more detailed so please skip to [Data Preprocessing](#Data-Preprocessing) section directly if you want to

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="workclass",hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')


# **Observation 1:** Most of the employess are employed in Private sector and looks like people who are self employed (inc - i believe registered firms) are more likely to earn >50K a year  
# In general, count of people earning <=50K is more than count of people earning more

# Let's see some numbers - what percentage of people in each class earns more than 50K a year

# In[ ]:


data_all[data_all['income'] == '>50K']['workclass'].value_counts()/data_all['workclass'].value_counts()


# We will also club together some of the classes based on above information

# In[ ]:


data_all.replace({'workclass':{'Federal-gov':'fed_gov', 
                               'State-gov':'gov', 'Local-gov':'gov', 'Self-emp-not-inc':'gov', 
                               'Without-pay':'unemployed','Never-worked':'unemployed', 
                               'Self-emp-inc':'self-emp'}}
                 ,inplace=True)
# 'Self-emp-not-inc' is put in same class as 'State-gov' as the proportion of people earning >50K is same
# ideally we should check relationship with other dependent variables too 
# (e.g. is age, education, hours.per.week distribution also same for 'Self-emp-not-inc' and 'State-gov' and 'local-gov', 
# if yes then we can combine classes else we should keep it as a separate class) 
# before deciding to put in same class but we will keep things simple now


# Occupation vs Income

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="occupation",hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')


# People at higher-up positions (Exec-managerial) have more probablity of earning more than >50K per year (close to 50% but that still seems less than my estimate.. but then again it is 1984 census data and we are in currently in 2020)  
# As expected, blue-collared workforce (handlers etc..) have much less probabilty of earning >50K, some still seem to be earning > 50K but those are most probably older people

# Let's see some actual numbers

# In[ ]:


data_all[data_all['income'] == '>50K']['occupation'].value_counts()/data_all['occupation'].value_counts()


# We will again group some classes based on above numbers

# In[ ]:


data_all.replace({'occupation':{'Exec-managerial':'premium_pay', 
                                'Prof-specialty':'good_pay','Protective-serv':'good_pay', 'Tech-support':'good_pay',
                                'Craft-repair':'normal_pay', 'Sales':'normal_pay','Transport-moving':'normal_pay',
                                'Adm-clerical':'low_pay','Armed-Forces':'low_pay','Farming-fishing':'low_pay','Machine-op-inspct':'low_pay',
                                'Other-service':'poor_pay', 'Handlers-cleaners':'poor_pay', 'Priv-house-serv':'poor_pay'}},inplace=True)


# Education vs Income

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="education.num", hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=60, horizontalalignment='right')


# So proportion of people earning > 50K is more for highly educated people... only if I knew before ;)

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
g = sns.countplot(x="marital.status",hue="income", data=data_all, palette="Set2")
sns.despine()
g = g.set_xticklabels(g.get_xticklabels(), rotation=30, horizontalalignment='right')


# Not quite sure what to conclude.. seems like people happily married (or let's just say married) have higher chances of earning > 50K as compared to other (unhappy ?) people.. doesn't look quite intutive to me though  
# 'Never-married' people are probably younger folks and that's why they earn less

# Again, we will group some classes into one

# In[ ]:


data_all.replace({'marital.status':{'Never-married':'single','Divorced':'single','Separated':'single',
                                   'Widowed':'single','Married-spouse-absent':'single','Married-AF-spouse':'single',
                                   'Married-civ-spouse':'married'}},inplace=True)


# Gender vs Income

# In[ ]:


sns.set(rc={'figure.figsize':(4,4)})
sns.set_style("white")
sns.countplot(x="sex",hue="income", data=data_all, palette="Set3")
sns.despine()


# We can see the pay gap between genders, proportion of females earning > 50K is less as compared to males

# Do females earn more at higer age as compared to males?

# In[ ]:


sns.catplot(x="sex", y="age", kind="violin", inner="box", data=data_all[data_all['income'] == '>50K'], orient="v")


# Doesn't quite look like the case, both male and female have approximately same median age when they start to earn > 50K

# In[ ]:


sns.catplot(x="education", y="age", kind="violin", inner="box"
            , data=data_all[data_all['income'] == '>50K'], orient="v", aspect=2.5, height=5)


# We can see that less educated people earn > 50K/year at an older age as compared to others

# Is there any pay gap between people from different races ?

# In[ ]:


sns.set(rc={'figure.figsize':(6,4)})
sns.set_style("white")
sns.countplot(x="race", hue="income", data=data_all, palette="Set2")
sns.despine()


# let's see some numbers, graph isn't quite clear

# In[ ]:


data_all[data_all['income'] == '>50K']['race'].value_counts()/data_all['race'].value_counts()


# looks like some discrimination with black folks.. let draw a graph comparing education, age, [White, black] races and income (only > 50K)

# In[ ]:


g = sns.catplot(x="education", y="age", hue="race", kind="violin", inner="quartile", split=True
            , data=data_all[(data_all['income'] == '>50K') & (data_all['race'].isin(['White','Black']))], orient="v", aspect=2.5, height=5)
g = g.set_xticklabels(rotation=60, horizontalalignment='right')


# Things kind of look alright to me.. more or less ok (with few exceptions, but that could actually be an issue iwth data too)  
# 
# Let's see if low proportion os people earning >50K is because black people are more employed in low paying occupation ?

# In[ ]:


g = sns.catplot(x="occupation", y="age", hue="race", kind="violin", inner="quartile", split=True
            , data=data_all[(data_all['race'].isin(['White','Black']))], orient="v", aspect=2, height=4)
g = g.set_xticklabels(rotation=60, horizontalalignment='right')


# This seems like the case, more of them (as compared to white) are employed in low and poor pay occupations.. so this explains it
# 
# Also, we will group together some classes

# In[ ]:


data_all.replace({'race':{'Black':'not-white', 'Asian-Pac-Islander':'not-white', 'Amer-Indian-Eskimo':'not-white'
                          ,'Other':'not-white'}}
                 , inplace=True)


# relationship vs income

# In[ ]:


sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("white")
sns.countplot(x="relationship",hue="income", data=data_all, palette="Set2")
sns.despine()


# relation case looks similar (related) to 'marital.status' field.. people happily married are husband and wife (and this graph suggests the same, people who are married have more probability of earning > 50K).. we may remove one of the feature from our model

# In[ ]:


data_all.replace({'relationship':{'Husband':'family','Wife':'family','Not-in-family':'not_family','Own-child':'family',
                                  'Unmarried':'not_family','Other-relative':'not_family'}},inplace=True)


# Much awaited Age vs Income comparision

# In[ ]:


data_all['age_bins'] = pd.cut(data_all['age'], bins=4)
sns.set(rc={'figure.figsize':(6,4)})
sns.set_style("white")
sns.countplot(x="age_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# as expected, proportion of younger people earning more than 50K/year is less as compared to adult and senior people

# In[ ]:


data_all['hpw_bins'] = pd.cut(data_all['hours.per.week'], 4)
sns.set(rc={'figure.figsize':(6,4)})
sns.set_style("white")
sns.countplot(x="hpw_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# simple math.. the more you work more you would earn

# In[ ]:


data_all['cap_gain_bins'] = pd.cut(data_all['capital.gain'], [0,3000,7000,100000])
sns.set(rc={'figure.figsize':(5,3)})
sns.set_style("white")
sns.countplot(x="cap_gain_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# capital gains field is co-related with income, more gains means more income

# In[ ]:


data_all['cap_loss_bins'] = pd.cut(data_all['capital.loss'], [0,1000,5000])
sns.set(rc={'figure.figsize':(3,3)})
sns.set_style("white")
sns.countplot(x="cap_loss_bins",hue="income", data=data_all, palette="Set2")
sns.despine()


# Doesn't look like same case with capital loss though, income doesn't seem to be related with capital.loss directly

# Update native.country values.. using continent names rather than country names to club together classes

# In[ ]:


data_all.replace({'native.country':{'China': 'asia', 'Hong': 'asia', 'India': 'asia', 'Iran': 'asia', 'Cambodia': 'asia', 'Japan': 'asia', 'Laos': 'asia', 'Philippines': 'asia', 'Vietnam': 'asia', 'Taiwan': 'asia', 'Thailand': 'asia'}},inplace=True)
data_all.replace({'native.country':{'England': 'europe', 'France': 'europe', 'Germany': 'europe', 'Greece': 'europe', 'Holand-Netherlands': 'europe', 'Hungary': 'europe', 'Ireland': 'europe', 'Italy': 'europe', 'Poland': 'europe', 'Portugal': 'europe', 'Scotland': 'europe', 'Yugoslavia': 'europe'}},inplace=True)
data_all.replace({'native.country':{'Canada':'NAmerica','United-States':'NAmerica','Puerto-Rico':'NAmerica'}},inplace=True)
data_all.replace({'native.country':{'Columbia': 'SA', 'Cuba': 'SA', 'Dominican-Republic': 'SA', 'Ecuador': 'SA', 'El-Salvador': 'SA', 'Guatemala': 'SA', 'Haiti': 'SA', 'Honduras': 'SA', 'Mexico': 'SA', 'Nicaragua': 'SA', 'Outlying-US(Guam-USVI-etc)': 'SA', 'Peru': 'SA', 'Jamaica': 'SA', 'Trinadad&Tobago': 'SA'}},inplace=True)
data_all.replace({'native.country':{'South':'SA'}},inplace=True)


# In[ ]:


# Except North America
sns.set_style("white")
sns.countplot(x="native.country",hue="income", data=data_all[data_all['native.country'] != 'NAmerica'], palette="Set2")
sns.despine()


# No co-relation with any country.. although proportion of people from south america region earning less than 50K is more as compared to people from other region

# In[ ]:


data_all.drop(columns=['age_bins','hpw_bins','cap_gain_bins','cap_loss_bins'],inplace=True)


# **Summarized findings**
# 1. Majority of people work in **Private**-Sector
# 2. People who are <b>self-employed </b>(self-employed-inc), or with <b>higher-education degree</b> (Prof-school, doctorate, masters) generally earn more than 50K a year,
# 3. People who are married have higher chances of earning more than >50K (may be this is related to age)
# 4. Approx 1/3rd of male earns more than 50K. For females this number is much lower
# 5. As age increases, proportion of people earning more than 50K increases
# 6. As number of work hours per week increases, proportion of people earning > 50K also increases, although people at higher-up positions tend to earn > 50K even when working for very few number of hours
# 7. More capital gain translates to more than 50K income
# 8. Proportion of black people earning > 50K is less than white people but when taking education into account, they seem to be paid fairely (with few exceptions). When compared with occupation we found that more of them are employed in low, poor pay occupations and hence proportion is less

# # Data Preprocessing
# 
# In next few steps, we will convert categorical variables to numerical, scale numerical variables, convert target variable to binary and drop 'education' variable, we will keep all other variables

# let's first convert income variable to binary 0 and 1 (as needed by model)

# In[ ]:


data_all.at[data_all[data_all['income'] == '>50K'].index, 'income'] = 1
data_all.at[data_all[data_all['income'] == '<=50K'].index, 'income'] = 0


# just check if everything got converted to 0 and 1 and class distribution is still same.. verification step

# In[ ]:


pd.value_counts(data_all['income'])/data_all['income'].count()*100


# let's **drop eduction** field as the same information is already presented by another numerical variable **education.num**

# In[ ]:


# we will not use 'education' column as we have 'education.num'
data_all.drop(columns=['education'], inplace=True)


# We will next do a train test split - 0.3 part for testing and rest for training  
# We will `stratify` our data on 'income' so that class distribution is same in both training and test

# In[ ]:


X,y = data_all.loc[:,data_all.columns != 'income'], data_all.loc[:,data_all.columns == 'income']
X_train, X_test, y_train, y_test = train_test_split(data_all.loc[:,data_all.columns != 'income']
                                                    , data_all.loc[:,data_all.columns == 'income']
                                                    , test_size=0.1, random_state=42, stratify=data_all['income'], shuffle=True)


# In[ ]:


X_train.shape, X_test.shape


# Scaling numerical variables  
# **Tip:** Make sure to use same scale for both training and testing (meaning don't fit separately on train and test sets)

# In[ ]:


#### Keep numerical
scalar_train = StandardScaler()
num_col_names = X_train.select_dtypes(include=['int64']).columns
num_col_df = X_train[num_col_names].copy()
scaled_num_col_df = scalar_train.fit_transform(num_col_df)
for i,j in enumerate(num_col_names):
    X_train[j] = scaled_num_col_df[:,i]

#---------------------------------
num_col_df_test = X_test[num_col_names].copy()
scaled_num_col_df = scalar_train.transform(num_col_df_test)
for i,j in enumerate(num_col_names):
    X_test[j] = scaled_num_col_df[:,i]


# In[ ]:


X_train.head()


# Looks like all variables scaled perfectly  
# 
# Converting rest of the variables to `category` type foe easier preprocessing in next steps

# In[ ]:


category_cols = list(set(X_train.columns.tolist()) - set(num_col_names))
for col in category_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')


# we will now create separate binary features for each workclass type, occupation type, race type, country name and marital status

# In[ ]:


X_train = pd.get_dummies(columns=category_cols, data=X_train, prefix=category_cols, prefix_sep="_",drop_first=True)

X_test = pd.get_dummies(columns=category_cols, data=X_test, prefix=category_cols, prefix_sep="_",drop_first=True)


# Let's just see what all and how manu columns are present in our final dataset which we will use for training

# In[ ]:


print(X_train.columns, X_train.shape)


# # Modeling Part
# 
# While **Neural network** in not something which is really required for this dataset, we are using it to see how much i**mprovement in accuracy** they bring (as compared to other methods tried by people in kernels on Kaggle), we also aim to show how **GridSearchCV** which is a scikit-learn method can actually be used with **Keras** and we will be using **Keras Wrapper for Scikit-learn** to achieve this.  
# 
# **GriDSearchCV** is something you can use with **smaller neural network models** (small training dataset, less learnable parameters) and is **not recommended for large model** as it is** computationaly expensive **and is **not** the most **optimal** method for searching best set of parameter values. For **large models**, we can use **Bayesian Optimization** for finding best performing set of parameter values  
# 
# If you want to learn more, here is a link to a great artical explaining these methods in more detail - http://krasserm.github.io/2018/03/21/bayesian-optimization/

# Let's define are model structure  
# We will use only 3 layes (2 hidden and one output - 16,8,1), we will use `relu` as the activation function, `Adam` as loss optimization function and `sigmoid` as the activation function for output layer.. nothing too complicated here!

# In[ ]:


def model_func(activation='relu', weight_decay=1e-7, learning_rate=0.001, dropout_rate=0.3):
    
    model = Sequential()
    model.add(Dense(16,input_dim=X_train.shape[1], activation=activation
                    , kernel_regularizer=l2(weight_decay), kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation=activation,kernel_regularizer=l2(weight_decay), kernel_initializer="normal"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid',kernel_regularizer=l2(weight_decay),kernel_initializer="normal"))
    
    # Compile model
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    return model


# defining our parameter grid for the GridSearch part, for keeping things simple (more importantly saving on computation cost) we are only search best values for `weight_deacy` and `learning_rate` parameters

# In[ ]:


param_grid = {'weight_decay': [1e-3, 1e-4, 1e-5, 1e-6]
              , 'learning_rate': [0.03, 0.01, 0.007, 0.003, 0.001, 0.0007, 0.0003]
             }


# Keras Wrapper - Wrapping our model with KerasClassifier (to be able to use GriSearchCV)

# In[ ]:


my_model = KerasClassifier(build_fn=model_func, verbose=0)


# Using GridSearchCV

# In[ ]:


model = GridSearchCV(estimator=my_model, param_grid=param_grid, verbose=0)


# Fitting training data to our model  
# **Important** thing to note, I have used `epochs`=5 to save on time but you can use more `epochs`in your model

# In[ ]:


model.fit(X_train, y_train, epochs=5, validation_split=0.2, batch_size=256, verbose=0)


# Let's take a look at the best parameters model found

# In[ ]:


print('The parameters of the best model are: ')
print(model.best_params_)
best_param = model.best_params_


# In[ ]:


print("Best: %f using %s" % (model.best_score_, model.best_params_))
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
#print(means, stds)
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))


# Now that we have found best parameters, I am training model again using those parameters.. this time for a little more `epochs`, also we will use two `callbacks` - **Early Stopping** and **Model Checkpoint** to basically stop training when loss starts to increase (patience is set to 5), and save best model (with least validation loss, you can actucally use 'val_acc' as the monitor metric too, it depends on the problem statement you are solving)

# In[ ]:


early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
callback_best_model = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)


# In[ ]:


model_nn = model_func(weight_decay=best_param['weight_decay'], learning_rate=best_param['learning_rate'])


# In[ ]:


model_nn.fit(np.asarray(X_train.astype(np.float)), np.asarray(y_train.astype('int')), batch_size=256, validation_split=0.25, epochs=50, verbose=2, callbacks=[early_stopping, callback_best_model], shuffle=True)


# Let's take a look at the accuracy and loss curves  
# Accuracy should go up and loass should go down as number of epochs increases  
# Ideally both of these curves should not fluctuate (straight type of lines), if they are fluctuating then it means we need to refine our mode (change number units, number of hidden layes, use different activation, optimization parameters, smaller/dynamic learning rate and different batch_size etc.)

# In[ ]:


# figure size in inches
rcParams['figure.figsize'] = 6,4

plt.plot(model_nn.history.history['accuracy'])
plt.plot(model_nn.history.history['val_accuracy'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


plt.plot(model_nn.history.history['loss'])
plt.plot(model_nn.history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# Graphs don't look the best (Accuracy plot), We should ideally fine tune our parameters but I'll leave that up to you

# # Predictions

# Let's load our best model again and predict on the test set we had reserved earlier in the Data Processing step

# In[ ]:


model_nn = load_model('best_model.h5')


# In[ ]:


train_pred=model_nn.predict_classes(X_train)
test_pred = model_nn.predict_classes(X_test)


# In[ ]:


confusion_matrix_test = confusion_matrix(y_test.astype(int).values, test_pred)
confusion_matrix_train = confusion_matrix(y_train.astype(int).values, train_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# Printing **train** and **test** **F1 score, precision** and **recall** values  
# Values shouldn't be too far for train and test set, **big difference **in training and test set performace would indicate a **bias or variance** issue with our model

# In[ ]:


Precision= confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1])
Recall = confusion_matrix_train[0,0]/(confusion_matrix_train[0,0]+confusion_matrix_train[1,0])
Accuracy = (confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train.sum())

print("Training Precision: %.3f" % Precision)
print("Training Recall: %.3f" % Recall)
print("Training F1 Score: %.3f" % ((2 * Precision * Recall) / (Precision + Recall)))
print("Training Accuracy: %.3f" % Accuracy)
print("-----------------------")
Precision= confusion_matrix_test[0,0]/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1])
Recall = confusion_matrix_test[0,0]/(confusion_matrix_test[0,0]+confusion_matrix_test[1,0])
Accuracy = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test.sum())
print("Test Precision: %.3f" % Precision)
print("Test Recall: %.3f" % Recall)
print("Test F1 Score: %.3f" % ((2*Precision*Recall)/(Precision + Recall)))
print("Test Accuracy: %.3f" % Accuracy)


# Oh hey, look we didn't do too bad on the test set and our model seems alright (can still be optimized though)

# # Conclusion/Comments
# 
# 
# *   We learned how to deal with a classification problem
# *   We learned to use a neural network using Keras
# *   We learned to use GridSearchCV with Keras to search for best parameters
# 
# **Please upvote this kernel if you find it helpful. Also, let me know in comments if you have any questions
# Thank You!**
# 
