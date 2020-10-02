#!/usr/bin/env python
# coding: utf-8

# # Santander Customer Transaction Prediction
At Santander our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.

Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.
# In[1]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('../input/train.csv')
data_pred=pd.read_csv('../input/test.csv')

data.head()

# In[4]:


data.describe()


# No hay valores Null en ninguna de las variables
data.info(verbose=True, null_counts=True)data_pred.info(verbose=True, null_counts=True)
# Target is a binary variable that indicates if the tansaction was made or not

# In[5]:


c=data['target'].value_counts()
print('% of 0 ---> ', c[0]/(c[0]+c[1]))
print('% of 1 ---> ', c[1]/(c[0]+c[1]))


# In[6]:


data['target'].hist()


# # EDA

# Box plot helps to understand the data distribution among the diferente Features, there are big data range defereences that could affect some algorithm trainings.

# In[7]:


data[data.columns[2:102]].plot(kind='box', figsize=[15,4], title='Non standarized values')
data[data.columns[103:]].plot(kind='box', figsize=[15,4], title='Non standarized values')


# In[8]:


import seaborn as sns


# In[9]:


values=data.columns.drop(['ID_code', 'target'])
plt.figure(figsize=(20,10))
for val in values:
    sns.distplot(data[val], hist=False)

plt.title('Density non Stadarized Data')
plt.xlabel('features')
plt.ylabel('density')


# In[10]:


val_max=pd.DataFrame(data=data.max(), columns=['max'])
val_max=val_max[2:]
val_max['var']=val_max.index
val_max[['var', 'max']].head()


# In[11]:


val_max['max'].plot(kind='hist', title='Max values distribution')


# In[12]:


val_max['max'].plot(kind='box', title='Max values distribution')


# In[13]:


data.kurt().head(10)
Kur_max=pd.DataFrame(data=(data.kurtosis()) , columns=['Kurtosis'])
Kur_max['var']=Kur_max.index
Kur_max.sort_values('Kurtosis', ascending=False).head()


# ## Multicorrelation Analytic

# In[14]:


features=data.drop(columns=['ID_code', 'target'])

features.head()
# In[15]:


correlations = data.corr().unstack().sort_values(ascending=True)


# In[16]:


cor_abs=correlations.abs().reset_index()

I have to remove from corr_abs those values pairs wich are equals, this twins pairs values does not offers any information their correlation is obviously 1
# In[17]:


cor_abs=cor_abs[cor_abs['level_0']!=cor_abs['level_1']]


# In[18]:


cor_abs=cor_abs.set_axis(['level_0', 'level_1', 'cor'],axis=1, inplace=False)

The most correlated pairs of values does not approach to 1, they show a relatively low correlation, therefore 'features' does not have a big multicorrelation
# In[19]:


cor_abs.tail(10)


# In[20]:


corr=data.corr()
plt.figure(figsize=(17,12))

sns.heatmap(corr, cmap='coolwarm')


# # Train/Test Spliting

# First I will split my data set into train and test

# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


train, test=train_test_split(data, test_size=0.25)


# In[23]:


train.head()


# In[24]:


x=train[train.columns[2:202]]
y_train=train[train.columns[1:2]]


# In[25]:


xt=test[test.columns[2:202]]
y_test=test[test.columns[1:2]]


# # Data Standarization

# In[26]:


from sklearn import preprocessing
std=preprocessing.StandardScaler()


# In[27]:


x_names=x.columns


# In[28]:


x_tr=std.fit_transform(x)
x_train=pd.DataFrame(x_tr, columns=x_names)


# In[29]:


xts=std.fit_transform(xt)
x_test=pd.DataFrame(xts, columns=x_names)


# ## --- EDA Data Standarized vs Non Standarized

# We very can verify how the distribution of the feature approaches each other, with similar data ranges and close to a normal data distribution

# In[30]:


data[data.columns[2:102]].plot(kind='box', figsize=[15,4], title='Non standarized values')
x_train[x_train.columns[:100]].plot(kind='box', figsize=[15,4], title='Standarized values')
data[data.columns[103:]].plot(kind='box', figsize=[15,4], title='Non standarized values')
x_train[x_train.columns[101:]].plot(kind='box', figsize=[15,4], title='Standarized values')


# In[31]:


values=data.columns.drop(['ID_code', 'target'])
plt.figure(figsize=(20,10))
for val in values:
    sns.distplot(data[val], hist=False)
plt.title('Density non Stadarized Data')
plt.xlabel('features')
plt.ylabel('density')

plt.figure(figsize=(20,10))
for val in values:
    sns.distplot(x_train[val], hist=False)
plt.title('Density Stadarized Data')
plt.xlabel('features')
plt.ylabel('density')


# # Features Selection.

# Features selection is far from easy, as we demonstrated there isn't a correlation among variables.
# I tried several methods to evaluate different possibilities of featuring reduction and as we will see none of them gives concluded results.

# ## PCA

# In[34]:


from sklearn.decomposition import PCA
#import mglearn


# In[35]:


array=x_train.values


# In[36]:


pca=PCA(n_components=3)
pca.fit(array)
threeD=pca.transform(array)
threeD


# In[37]:


three_Df = pd.DataFrame(data = threeD, columns = ['PCA1', 'PCA2', 'PCA3']) 


# In[38]:


df_pca = pd.concat([three_Df, y_train], axis = 1)


# In[39]:


df_pca.head()


# In[40]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


# In[41]:



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d') 
ax.set_xlabel('PCA1', fontsize = 15)
ax.set_ylabel('PCA2', fontsize = 15)
ax.set_zlabel('PCA3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = [0,1]
colors = ['r','g']
for target, color in zip(targets,colors):
    indicesToKeep = df_pca['target'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'PCA1']
               , df_pca.loc[indicesToKeep, 'PCA2']
               , df_pca.loc[indicesToKeep, 'PCA3']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()

ax.view_init(azim=50)

    


# In[42]:


pca.explained_variance_ratio_


# That was something expected, the large number of features and the non-correlation. Dimensionality reduction to 3 var does not show any independent cluster of values.

# ## Forward Squential Feature Selector

# I tried the forward Sequential Feature Selector with an iteration from 50 to 200 features, Unfortunately, the training process took hours without concluyentes results.
x_arr=np.asarray(x_train)
y_arr=np.asarray(y_train['target'])
features=x_train.columns
y_arrfrom mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
feature_selector = sfs(estimator=knn,  
           k_features=(50,200),
           forward=True,
           verbose=2,
           scoring='accuracy',
           cv=4)feature_selector.fit(x_arr, y_arr, custom_feature_names=features)
# 
# 
# print('best combination (ACC: %.3f): %s\n' % (feature_selector.k_score_, feature_selector.k_feature_idx_))
# print('all subsets:\n', feature_selector.subsets_)
# plot_sfs(feature_selector.get_metric_dict(), kind='std_err');

# ## Lasso Regession an Features selection.

# In[43]:


from sklearn.linear_model import Lasso


# Alpha coeficiente evaluation for lasso regression. The Cross Validation algorithm GridSearchCV will iterate the Lasso algorith for diferent values of Alpha. In this case GridSearch will test a randon number of alpha values 

# In[44]:


from scipy.stats import uniform


# In[45]:


alphas=uniform.rvs(loc=0, scale=0.2, size=30)
alphas


# I just keep the value of alphas bolcked copy/paste, if not each time we run uniform.rvs the alphas array will change and therefore the result for best alpha.

# In[46]:


alphas=[0.12301225, 0.14288355, 0.18551073, 0.05006723, 0.0333933 ,
       0.03646111, 0.04268822, 0.10610886, 0.19878154, 0.01463984,
       0.09548202, 0.13826288, 0.12977404, 0.06173418, 0.09480236,
       0.15044969, 0.05521685, 0.00238981, 0.13915425, 0.15324187,
       0.18726584, 0.0666834 , 0.01948747, 0.02757435, 0.13793408,
       0.09817728, 0.02072232, 0.1429758 , 0.11844789, 0.04484972]


# In[47]:


from sklearn.model_selection import GridSearchCV
model=Lasso()
grid=GridSearchCV(estimator=model,param_grid=dict(alpha=alphas), cv=10)


# In[48]:


grid.fit(x_train, y_train)


# In[49]:


print('Best alpha--->', grid.best_params_)
print('Best score--->', grid.best_score_)


# In[50]:


model_lasso=Lasso(alpha=0.00238981)
model_lasso.fit(x_train, y_train)

model_lasso.coef_

# In[51]:


lasso_cf=list(model_lasso.coef_)
feature_names=x_train.columns.values.tolist()
coef_lasso=pd.DataFrame({'feature': feature_names, 'Coef':lasso_cf})
features_filter=coef_lasso[coef_lasso['Coef']!=0]
features_sel=features_filter['feature'].tolist()
print(features_sel)
len(features_sel)


# #### Same Lasso Algorithm but with Non normalized features.

# In[52]:


x=train[train.columns[2:102]]


# In[53]:


grid.fit(x, y_train)


# In[54]:


print('Best alpha--->', grid.best_params_)
print('Best score--->', grid.best_score_)


# In[55]:


model_lasso=Lasso(alpha=0.00238981)
model_lasso.fit(x, y_train)


# In[56]:


lasso_cf=list(model_lasso.coef_)
feature_names=x.columns.values.tolist()
coef_lasso=pd.DataFrame({'feature': feature_names, 'Coef':lasso_cf})
features_filter=coef_lasso[coef_lasso['Coef']!=0]
features_sel=features_filter['feature'].tolist()
print(features_sel)
len(features_sel)


# In[57]:


x_lasso_non=x_train[features_sel]
x_lasso_non.head()


# In[58]:


xtest_lasso_non=x_test[features_sel]


# ## We have fiferent Feature data-sets to be tested

# x_train and x_test ----> 200 features NORMALIZED

# x_lasso_non and xtest_lasso_non ----> 87 features Non NORMALIZED following lasso reduction model of features

# x_lasso and xtest_lasso ----> 87 features NORMALIZED

# In[59]:


x_lasso=x_train[features_sel]
x_lasso.columns


# In[60]:


xtest_lasso=x_test[features_sel]


# ## Aplico Regresion logistica con las nuevas variables.

# In[61]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[62]:


log_reg=LogisticRegression()
log_reg1=LogisticRegression()
log_reg2=LogisticRegression()
log_reg3=LogisticRegression()


# In[63]:


xtot_non=train[train.columns[2:202]]
xtot_non.head()


# In[64]:


log_reg.fit(xtot_non,y_train)


# In[65]:


log_reg1.fit(x_train,y_train)


# In[66]:


log_reg2.fit(x_lasso , y_train)


# In[67]:


log_reg3.fit(x_lasso_non , y_train)


# In[68]:


y_pred=log_reg.predict(xt)


# In[69]:


y_pred1=log_reg1.predict(x_test)


# In[70]:


y_pred2=log_reg2.predict(xtest_lasso)


# In[71]:


y_pred3=log_reg3.predict(xtest_lasso_non)


# In[72]:


print('score of 200 features normalized----->', log_reg1.score(x_test,y_test))
print('score of 200 features NO normalized----->', log_reg.score(xt,y_test))

print('score of 48 features normalizer------>', log_reg2.score(xtest_lasso,y_test))
print('score of 48 features NO normalizer--->', log_reg3.score(xtest_lasso_non,y_test))


# In[73]:


conf_matrix1=confusion_matrix(y_test, y_pred1)
conf_matrix2=confusion_matrix(y_test, y_pred2)
conf_matrix3=confusion_matrix(y_test, y_pred3)


# In[74]:


tp1=conf_matrix1[0,0]+conf_matrix1[1,1]
tp2=conf_matrix2[0,0]+conf_matrix2[1,1]
tp3=conf_matrix3[0,0]+conf_matrix3[1,1]
fp1=conf_matrix1[0,1]+conf_matrix1[1,0]
fp2=conf_matrix2[0,1]+conf_matrix2[1,0]
fp3=conf_matrix3[0,1]+conf_matrix3[1,0]


# In[75]:


print('True predictions 200 features normalized---->',tp1)
print('True predictions 48 features normalized----->',tp2)
print('True predictions 48 features NO normalized-->',tp3)

print('False predictions 200 features normalized--->',fp1)
print('False predictions 48 features normalized---->',fp2)
print('False predictions 48 features NO normalized->',fp3)




# # Deep Learning con Keras

# In[76]:


import tensorflow as tf
from tensorflow import keras


# In[77]:


from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as k
from tensorflow.keras.models import model_from_json




# In[78]:


features=xtot_non.shape[1]


# In[79]:


data=xtot_non.as_matrix()
lab=y_train.as_matrix()
label=to_categorical(lab)
data_test=xt.as_matrix()
lab_test=y_test.as_matrix()
label_test=to_categorical(lab_test)


# In[80]:


model=tf.keras.Sequential()
k.clear_session()

model.add(layers.Dense(400, activation='relu', input_shape=(features,)))
model.add(layers.Dense(400, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, label, epochs=25, batch_size=512)


# In[81]:


model.evaluate(data_test, label_test, batch_size=512)


# In[82]:


features1=x_train.shape[1]


# In[83]:


data1=x_train.as_matrix()
lab=y_train.as_matrix()
label=to_categorical(lab)


# In[84]:


data1_test=x_test.as_matrix()
lab_test=y_test.as_matrix()
label_test=to_categorical(lab_test)


# In[85]:


model1=tf.keras.Sequential()


# In[86]:


k.clear_session()


# In[87]:


model1.add(layers.Dense(800, activation='relu', input_shape=(features1,)))
model1.add(layers.Dense(800, activation='relu'))
model1.add(layers.Dense(400, activation='relu'))
model1.add(layers.Dense(200, activation='relu'))
model1.add(layers.Dense(2, activation='softmax'))


# In[88]:


model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

data.shape
lab.shape
# In[89]:


model1.fit(data1,label, epochs=300, batch_size=512)


# In[90]:


model1.evaluate(data1_test, label_test, batch_size=512)

prediction1=model1.predict(data1_test, batch_size=256)
# In[91]:


model_js=model1.to_json()
with open("model.json1", "w") as json_file:
    json_file.write(model_js)
# serialize weights to HDF5
model1.save_weights("model1.h5")
print("Saved model to disk")


# #### MODEL2 trained with 48 features Normalized

# In[92]:


features2=x_lasso.shape[1]

data2=x_lasso.as_matrix()
data2_test=xtest_lasso.as_matrix()

model2=tf.keras.Sequential()
k.clear_session()

model2.add(layers.Dense(400, activation='relu', input_shape=(features2,)))
model2.add(layers.Dense(400, activation='relu'))
model2.add(layers.Dense(200, activation='relu'))
model2.add(layers.Dense(100, activation='relu'))
model2.add(layers.Dense(2, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model2.fit(data2, label, epochs=25, batch_size=512)


# In[93]:


model2.evaluate(data2_test, label_test, batch_size=512)

prediction2=model2.predict(data2_test, batch_size=256)
# In[94]:


model_js=model2.to_json()
with open("model.json2", "w") as json_file:
    json_file.write(model_js)
# serialize weights to HDF5
model2.save_weights("model2.h5")
print("Saved model to disk")


# #### MODEL3 trained with 48 features NON Normalized

# In[95]:


features3=x_lasso_non.shape[1]

data3=x_lasso_non.as_matrix()
data3_test=xtest_lasso_non.as_matrix()

model3=tf.keras.Sequential()
k.clear_session()

model3.add(layers.Dense(400, activation='relu', input_shape=(features3,)))
model3.add(layers.Dense(400, activation='relu'))
model3.add(layers.Dense(200, activation='relu'))
model3.add(layers.Dense(100, activation='relu'))
model3.add(layers.Dense(2, activation='softmax'))

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model3.fit(data3, label, epochs=25, batch_size=512)


# In[96]:


model3.evaluate(data3_test, label_test, batch_size=512)


# prediction3=model3.predict(data3_test, batch_size=256)

# In[97]:


model_js=model3.to_json()
with open("model.json3", "w") as json_file:
    json_file.write(model_js)
# serialize weights to HDF5
model3.save_weights("model3.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json1', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")
 
# loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
loaded_model.evaluate(data_test, label_test, batch_size=10)
# # Random Forest Classifier

# In[98]:


from sklearn.ensemble import RandomForestClassifier as rfc
model=rfc(n_jobs=2, random_state=0)
model.fit(x_train, y_train)


# Fitting nomalized data -----> x_train, y_train, x_test

# In[99]:


x_train.shape


# In[100]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train, model.predict(x_train))


# Very poor accuracy I will not use random forest

# In[101]:


y_pred=model.predict(xt)
accuracy_score(y_test, y_pred)


# # Final Conclusion

# After several attempts over different model algorithms feeding with a combination of data sets, we didn't get a relevant variation of accuracy during the valuation process. The best accuracy refers to the simplest model, a logistic regression trained wit 200 normalized features.

# In[102]:



x_pred=data_pred[data_pred.columns[1:201]]
x_var=x_pred.columns


# In[103]:



x_norm=std.fit_transform(x_pred)
x_norm=pd.DataFrame(x_norm, columns=x_var)


# In[104]:


prediction=log_reg1.predict(x_norm)


# In[105]:


prediction=pd.DataFrame(data=prediction , columns=['target'])
prediction.head()


# In[106]:


ID_code=[]

for i in range(len(prediction)):
    s=str(i)
#    t=str(prediction[i])
    line='test_'+s
    ID_code.append(line)
    


# In[107]:


ID=pd.DataFrame(data=ID_code, columns=['ID_code'])


# In[108]:


ID.head()


# In[109]:


ID['target']=prediction.target
ID['target'].hist()


# In[110]:


p=ID['target'].value_counts()
print('% of 0 ---> ', p[0]/(p[0]+p[1]))
print('% of 1 ---> ', p[1]/(p[0]+p[1]))


# In[111]:


ID.to_csv('submission.csv', index=False)

