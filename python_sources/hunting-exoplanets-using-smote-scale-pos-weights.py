#!/usr/bin/env python
# coding: utf-8

# # Understanding flux
# 
# Flux (or radiant flux), F, is the total amount of energy that crosses a unit area per unit time. Flux is measured in joules per square metre per second (joules/m2/s), or watts per square metre (watts/m2).
# 
# The flux of an astronomical source depends on the luminosity of the object and its distance from the Earth, according to the inverse square law:
# 
# $\displaystyle  F=\frac{L}{4\pi r^2} $
# 
# where F = flux measured at distance r,
# L = luminosity of the source,
# r= distance to the source.
# Source: https://astronomy.swin.edu.au/cosmos/F/Flux
# 
# 

# ## Reading the file and plotting flux for exoplanets vs non exoplanets
# 1.We will transpose the file to make stars rows into column to help us plot few for some stars flux over period of time
# 2.We will divide the dataset based on labels intow two parts for different star type

# In[ ]:


import pandas as pd
exo_train=pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
exo_test=pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
train_exo_y=exo_train[exo_train['LABEL'] >1 ]
train_exo_n=exo_train[exo_train['LABEL'] < 2]
train_t_n=train_exo_n.iloc[:,1:].T
train_t_y=train_exo_y.iloc[:,1:].T
train_t_n.head(1)
exo_train['LABEL'].value_counts()


# **This is highly unbalanced data, other kernels have made prediction predicting everything as non exoplanet and claimed 99% accuracy which is not right**

# ## Making scatter plot for flux from two different categories

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2,subplot_titles=("Flux variation of star 37", "Flux variation of star 5086", 
                                                   "Flux variation of star 3000", "Flux variation of star 3001"))
fig.add_trace(
    go.Scatter(y=train_t_n[37], x=train_t_n.index),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_n[5086], x=train_t_n.index),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=train_t_n[3000], x=train_t_n.index),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_n[3001], x=train_t_n.index),
    row=2, col=2
)
fig.update_layout(height=600, width=800, title_text="Non Exoplanets Star examples",showlegend=False)
fig.show()


# In[ ]:


fig = make_subplots(rows=2, cols=2,subplot_titles=("Flux variation of star 0", "Flux variation of star 1", 
                                                   "Flux variation of star 35", "Flux variation of star 36"))
fig.add_trace(
    go.Scatter(y=train_t_y[0], x=train_t_y.index),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_y[1], x=train_t_y.index),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(y=train_t_y[35], x=train_t_y.index),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(y=train_t_y[36], x=train_t_y.index),
    row=2, col=2
)
fig.update_layout(height=600, width=800, title_text="Exoplanets Stars examples",showlegend=False)


# **Key Observations:**
# 1) Irregular pattern of flux due to outliers in some hours.
# 2) Ignoring the anomalies in flux due to recording in non exoplanets there are not a cylce of flux change
# 3) In expolnates examples we clearly see cycle of flux change. Like for star1, 160 hour cycle is evident that there is planet revolving around this star for this time period relative to earth. Like wise for star 0 it is 800 hours and for star 35, 200 hours.

# # Applying different models to predict type of stars(exo/vs non exo)

# In[ ]:


###Normalizing the flux#####
from sklearn.preprocessing import StandardScaler
trainx=exo_train.iloc[:,1:]
textx=exo_test.iloc[:,1:]
scaler=StandardScaler()
train_scaled=scaler.fit_transform(trainx)
test_scaled=scaler.fit_transform(textx)


# In[ ]:


### Applying SVC with linear Kernel
trainy=exo_train[['LABEL']]
testy=exo_test[['LABEL']]
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(train_scaled, trainy['LABEL'])
y_pred = svclassifier.predict(test_scaled)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy, y_pred))
print(classification_report(testy, y_pred))


# **The Accuracy is high because the category 2 the exoplanets stars count is very less in test data, even if we label all the stars to 1 we will have 99 % F1 score. We will try applying other kernel in SVC**

# In[ ]:


####Polynomial Kernel ###
svclassifier = SVC(kernel='poly')
svclassifier.fit(train_scaled, trainy['LABEL'])
y_pred = svclassifier.predict(test_scaled)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy, y_pred))
print(classification_report(testy, y_pred))


# Nothing has been predicted as exo planet which is why we see that Sklearn has throwed a warning that precision and F1 score is ill defined.

# In[ ]:


### RBF kernel###
svclassifier = SVC(kernel='rbf')
svclassifier.fit(train_scaled, trainy['LABEL'])
y_pred = svclassifier.predict(test_scaled)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy, y_pred))
print(classification_report(testy, y_pred))


# In[ ]:


###Sigmoid kernel###
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(train_scaled, trainy['LABEL'])
y_pred = svclassifier.predict(test_scaled)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy, y_pred))
print(classification_report(testy, y_pred))


# None of the Kernels worked in SVM. Lets try PCA now

# # Applying PCA to reduce flux columns to 6 which capture 82% variation in data
# 
# We will use common dimensionality reduction technique principal component analysis to reduce dimension to wherever we are getting 80% of variability in data.The six principal components will be used for prediction from now onwards

# In[ ]:


import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(train_scaled)
PCA(n_components=6)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
trns_x=pca.transform(train_scaled)
trns_y=pca.transform(test_scaled)
testy


# In[ ]:


##Applying SVC RBF to new transformed dataset #####
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(trns_x, trainy['LABEL'])
y_pred = svclassifier.predict(trns_y)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy['LABEL'], y_pred))
print(classification_report(testy['LABEL'], y_pred))


# Still we are having the model problem unable to predict star type exoplanet

# # Applying Logistic Regression
# Using Stats model and sklearn library we will see how logistic regression performs.

# In[ ]:


trainy.loc[trainy['LABEL'] == 1, 'new1'] = 0
trainy.loc[trainy['LABEL'] > 1, 'new1'] = 1
testy.loc[testy['LABEL'] > 1, 'new1'] = 1
testy.loc[testy['LABEL'] == 1, 'new1'] = 0


# In[ ]:


import statsmodels.api as sm
logit_model=sm.Logit(trainy['new1'],trns_x)
result=logit_model.fit()
print(result.summary2())


# Psuedo R square value suggest the model has poor prediction power. None of Flux values X1,x2,X3,X4,X5,X6 Which we derived from PCA are coming to be signhicant, hence this model is extremely poor.

# In[ ]:


###Scikitlearn Logistic regression ###
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(trns_x, trainy['new1'])
y_pred = logreg.predict(trns_y)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(trns_y, testy['new1'])))


# The accuracy is coming to be 99% on test data again this could be because model is predicting everything as 0 non exo planet. Now we will see ROC curve to get the story.

# # ROC curve tells the real story

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
logit_roc_auc = roc_auc_score(testy['new1'], logreg.predict(trns_y))
fpr, tpr, thresholds = roc_curve(testy['new1'], logreg.predict_proba(trns_y)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# The ROC AUC shows 50% AUC which is equivalent to random guessing , if we predict everything to 0 then this ROC curve comes.Stating F1 score and accuracy are farce metrics in case of imbalanced data.

# # Solving for class imbalance using SMOTE
# Synthetic Minority Oversampling Technique: SMOTE synthesises new minority instances between existing (real) minority instances. .The SMOTE will make 1 and 0 equal for our datasets.
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

# In[ ]:


from imblearn.over_sampling import SMOTE
over = SMOTE(random_state=0)
ov_train_x,ov_train_y=over.fit_sample(trns_x, trainy['new1'])
ov_train_y=ov_train_y.astype('int')
ov_train_y.value_counts()


# ### Using RBF SVC on oversampled data

# In[ ]:


ov_train_y=ov_train_y.values.tolist()
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(ov_train_x, ov_train_y)
y_pred = svclassifier.predict(trns_y)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testy['new1'], y_pred))
print(classification_report(testy['new1'], y_pred))


# Finally  we are able to predict the 1 out of 5 exoplanets correctly as shown in the confusion matrix.

# # Nueral Network - Feed Forward using Keras
# Although it is not sure that neural network are proven to work on imbalanced data, but i am giving it try by trying different activation functions and changing optimizers and learning rates.
# 
# * I have tried custom learning learning rate exp_decay which penalizes the increase in epochs
# * I have also tried clip value in stocashtic gradient descent to remove exploding gradient problem.
# * I have tried SELU which is removes vanishing gradient problem also newly introduced google activation function swish
# * I have also used L2 regularization to reduce overfitting in hidden layers
# * I have tried parameter reduce learning rate on plateau so the convergence alogirthm in case LR reduces.
# * I have also tried HE initialization for reducing vanishing gradient problem by intializing weights by formual v2=2/N
# 

# In[ ]:


import numpy as np
ov_train_y=np.array(ov_train_y)
ov_train_y.dtype
from sklearn.model_selection import train_test_split
tr_x,v_x,tr_y,V_y= train_test_split(ov_train_x, ov_train_y, test_size=0.2)


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.layers import LeakyReLU
model=keras.models.Sequential([
    keras.layers.Dense(300,activation="selu",input_shape=(8080,6)),    
    keras.layers.Dense(200,activation="selu",kernel_regularizer=keras.regularizers.l2(0.01)),   
    keras.layers.Dense(100,activation="selu",kernel_regularizer=keras.regularizers.l2(0.01)),   
    keras.layers.Dense(2,activation="softmax")
])
epochs=50
optimizers=keras.optimizers.SGD(clipvalue=1.0)
def exp_decay(lr0,s):
    def exp_decay_fn(epcohs):
        return lr0*0.1**(epochs/s)
    return exp_decay_fn

exp_decay_fn=exp_decay(lr0=0.1,s=20)
lr_sch=keras.callbacks.LearningRateScheduler(exp_decay_fn)
lr_sch2=keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizers,metrics=["accuracy"])
history=model.fit(tr_x,tr_y,epochs=50,callbacks=[lr_sch],validation_data=(v_x,V_y))
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
predict=model.predict_classes(trns_y)
print(confusion_matrix(testy['new1'], predict))
print(classification_report(testy['new1'], predict))


# **This combination of Selu/softmax ,l2 regularization, reducing LR on plateau and Sparse categorical cross entropy loss function is able to predict minority class but fails in predicting majority class.**

# ## Trying another combination of Activation Functions(ADAM)
# 
# Adam is known for its faster convergence over SGD

# In[ ]:


model=keras.models.Sequential([
    keras.layers.Dense(300,activation="swish",input_shape=(8080,6)),    
    keras.layers.Dense(200,activation="swish",kernel_initializer="he_normal"),   
    keras.layers.Dense(100,activation="swish",kernel_initializer="he_normal"),    
    keras.layers.Dense(2,activation="softmax")
])

optimizers=tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam"
    
)
lr_sch2=keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)
model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizers,metrics=["accuracy"])
history=model.fit(tr_x,tr_y,epochs=50,validation_data=(v_x,V_y),callbacks=[lr_sch2])
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
predict=model.predict_classes(trns_y)
print(confusion_matrix(testy['new1'], predict))
print(classification_report(testy['new1'], predict))


# Adam took us back to point zero where we are not able to predict minority class, this model  also have not been regularised- no droput or l1/l2 regularization hence with increasing epochs the accuracy keeps on increasing a bit , leading to overfitting.

# # Final Solution: Oversampled,Transformed,Weighted XGBOOST
# 
# I went through Kaggle discussion and found out a useful feature in xgboost: Scale POS weight
# 
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41359
# 
# **Now scale_pos_weight is the ratio of number of negative class to the positive class, but interestingly in an oversampled data which has equal classes instances we should be getting best accuracy at 0.5 , but i got that at 33% or 0.33**

# In[ ]:


# T - no. of total samples
# P - no. of positive samples
# scale_pos_weight = percent of negative / percent of positive
# which translates to:
# scale_pos_weight = (100*(T-P)/T) / (100*P/T)
# which further simplifies to beautiful:
#scale_pos_weight = |37/5387 - 1|=0.99
from xgboost import XGBClassifier
scale_pos_weight = [2,0.99,0.60,0.50,0.33,0.20,0.10]
for i in scale_pos_weight:
    print('scale_pos_weight = {}: '.format(i))
    clf = XGBClassifier(scale_pos_weight=i)
    clf.fit(ov_train_x, ov_train_y)
    predict = clf.predict(trns_y)    
    cm = confusion_matrix(testy['new1'], predict)  
    auc=metrics.roc_auc_score(testy['new1'], predict)
    print('Confusion Matrix: \n', cm)
    print('metrics: \n',classification_report(testy['new1'], predict))
    print('AUC of test set: {:.2f} \n'.format(metrics.roc_auc_score(testy['new1'], predict))) 


# ************The best prediction we got is at .33 weightage which gave us 2 out of 5 minority classes predicted , also it didnt compromise on majority prediction like NN, and gave a good f1 score and 64 % AUC which is drastic improvement from 50%
