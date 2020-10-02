#!/usr/bin/env python
# coding: utf-8

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Code Library, Styling, and Links</h1>
# `GITHUB` Version: &#x1F4D8; &nbsp; [kaggle_passnyc6.ipynb](https://github.com/OlgaBelitskaya/kaggle_notebooks/blob/master/kaggle_passnyc6.ipynb)
# 
# The previous notebooks:
# 
# &#x1F4D8; &nbsp; [PASSNYC. Data Exploration](https://www.kaggle.com/olgabelitskaya/passnyc-data-exploration)
# 
# &#x1F4D8; &nbsp; [PASSNYC. Numeric and Categorical Variables](https://www.kaggle.com/olgabelitskaya/passnyc-numeric-and-categorical-variables)
# 
# &#x1F4D8; &nbsp; [PASSNYC. Comparing All Districts with 5th District](passnyc-comparing-all-districts-with-5th-district)
# 
# &#x1F4D8; &nbsp; [PASSNYC. Regression Methods](https://www.kaggle.com/olgabelitskaya/passnyc-regression-methods)
# 
# &#x1F4D8; &nbsp; [PASSNYC. Neural Networks](https://www.kaggle.com/olgabelitskaya/passnyc-neural-networks)
# 
# Useful `LINKS`:
# 
# &#x1F4E1; &nbsp; [School Quality Reports. Educator Guide](http://schools.nyc.gov/NR/rdonlyres/967E0EE1-7E5D-4E47-BC21-573FEEE23AE2/0/201516EducatorGuideHS9252017.pdf)
# 
# &#x1F4E1; &nbsp; [New York City Department of Education](https://www.schools.nyc.gov)
# 
# &#x1F4E1; &nbsp; [NYC OpenData](https://opendata.cityofnewyork.us/)
# 
# &#x1F4E1; &nbsp; [Pandas Visualization](https://pandas.pydata.org/pandas-docs/stable/visualization.html) & &#x1F4E1; &nbsp; [Pandas Styling](https://pandas.pydata.org/pandas-docs/stable/style.html)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<style> \n@import url('https://fonts.googleapis.com/css?family=Orbitron|Roboto&effect=3d');\nbody {background-color: gainsboro;} \nh3 {color:#818286; font-family:Roboto;}\nspan {color:black; text-shadow:4px 4px 4px #aaa;}\ndiv.output_prompt,div.output_area pre {color:slategray;}\ndiv.input_prompt,div.output_subarea {color:#37c9e1;}      \ndiv.output_stderr pre {background-color:gainsboro;}  \ndiv.output_stderr {background-color:slategrey;}                \n</style>")


# In[ ]:


import warnings; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
import pylab as plt,seaborn as sns
import matplotlib.colors as mcolors
from descartes import PolygonPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import median_absolute_error,mean_absolute_error
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import explained_variance_score
from keras.models import Sequential,Model
from keras.optimizers import SGD,RMSprop
from keras.layers import Dense,Dropout,LSTM
from keras.layers import Activation,Flatten,Input,BatchNormalization
from keras.layers import Conv1D,MaxPooling1D,Conv2D,MaxPooling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
cmap=plt.cm.get_cmap('Spectral',4)
spectral_cmap=[]
for i in range(cmap.N):
    rgb=cmap(i)[:3]
    spectral_cmap.append(mcolors.rgb2hex(rgb))
plt.style.use('seaborn-whitegrid'); path='../input/'
fw='weights.passnyc.hdf5'


# In[ ]:


def scores(regressor,y_train,y_valid,y_test,
           y_train_reg,y_valid_reg,y_test_reg):
    print(20*"<=>"); print(regressor); print(20*"<=>")
    print("EV score. Train: ",
          explained_variance_score(y_train,y_train_reg))
    print("EV score. Valid: ",
          explained_variance_score(y_valid,y_valid_reg))
    print("EV score. Test: ",
          explained_variance_score(y_test,y_test_reg))
    print(20*"<=>")
    print("R2 score. Train: ",r2_score(y_train,y_train_reg))
    print("R2 score. Valid: ",r2_score(y_valid,y_valid_reg))
    print("R2 score. Test: ",r2_score(y_test,y_test_reg))
    print(20*"<=>")
    print("MSE score. Train: ",
          mean_squared_error(y_train,y_train_reg))
    print("MSE score. Valid: ",
          mean_squared_error(y_valid,y_valid_reg))
    print("MSE score. Test: ",
          mean_squared_error(y_test,y_test_reg))
    print(20*"<=>")
    print("MAE score. Train: ",
          mean_absolute_error(y_train,y_train_reg))
    print("MAE score. Valid: ",
          mean_absolute_error(y_valid,y_valid_reg))
    print("MAE score. Test: ",
          mean_absolute_error(y_test,y_test_reg))
    print(20*"<=>")
    print("MdAE score. Train: ",
          median_absolute_error(y_train,y_train_reg))
    print("MdAE score. Valid: ",
          median_absolute_error(y_valid,y_valid_reg))
    print("MdAE score. Test: ",
          median_absolute_error(y_test,y_test_reg))
def history_plot(fit_history,n):
    keys=list(fit_history.history.keys())[0:4]
    plt.figure(figsize=(11,10)); plt.subplot(211)
    plt.plot(fit_history.history[keys[0]][n:],
             color='slategray',label='train')
    plt.plot(fit_history.history[keys[2]][n:],
             color='#37c9e1',label='valid')
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.title('Loss Function')    
    plt.subplot(212)
    plt.plot(fit_history.history[keys[1]][n:],
             color='slategray',label='train')
    plt.plot(fit_history.history[keys[3]][n:],
             color='#37c9e1',label='valid')
    plt.xlabel("Epochs"); plt.ylabel("MAE"); plt.legend()
    plt.title('Mean Absolute Error'); plt.show()


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Data Loading and Preprocessing</h1>

# In[ ]:


school_explorer=pd.read_csv(path+'2016 School Explorer.csv')
d5_shsat=pd.read_csv(path+'D5 SHSAT Registrations and Testers.csv')
school_explorer.shape,d5_shsat.shape


# In[ ]:


drop_list=['Adjusted Grade','New?','Other Location Code in LCGMS']
school_explorer=school_explorer.drop(drop_list,axis=1)
school_explorer.loc[[427,1023,712,908],'School Name']=['P.S. 212 D12','P.S. 212 D30','P.S. 253 D21','P.S. 253 D27']
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].astype('object') 
for s in [",","$"," "]:
    school_explorer['School Income Estimate']=    school_explorer['School Income Estimate'].str.replace(s,"")
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].str.replace("nan","0")
school_explorer['School Income Estimate']=school_explorer['School Income Estimate'].astype(float)
school_explorer['School Income Estimate'].replace(0,np.NaN,inplace=True)
percent_list=['Percent ELL','Percent Asian','Percent Black',
              'Percent Hispanic','Percent Black / Hispanic',
              'Percent White','Student Attendance Rate',
              'Percent of Students Chronically Absent',
              'Rigorous Instruction %','Collaborative Teachers %',
              'Supportive Environment %','Effective School Leadership %',
              'Strong Family-Community Ties %','Trust %']
target_list=['Average ELA Proficiency','Average Math Proficiency']
economic_list=['Economic Need Index','School Income Estimate']
rating_list=['Rigorous Instruction Rating','Collaborative Teachers Rating',
             'Supportive Environment Rating','Effective School Leadership Rating',
             'Strong Family-Community Ties Rating','Trust Rating',
             'Student Achievement Rating']
for el in percent_list:
    school_explorer[el]=school_explorer[el].astype('object')
    school_explorer[el]=school_explorer[el].str.replace("%","")
    school_explorer[el]=school_explorer[el].str.replace("nan","0")
    school_explorer[el]=school_explorer[el].astype(float)
    school_explorer[el].replace(0,np.NaN,inplace=True)
    school_explorer[el]=school_explorer[el].interpolate()
for el in target_list+economic_list:
    school_explorer[el]=school_explorer[el].interpolate()
for el in rating_list:
    moda_value=school_explorer[el].value_counts().idxmax()
    school_explorer[el]=school_explorer[el].fillna(moda_value)    
category_list=['District','Community School?','City','Grades']               
for feature in category_list:
    feature_cat=pd.factorize(school_explorer[feature])
    school_explorer[feature]=feature_cat[0]    
for feature in rating_list:
    feature_pairs=dict(zip(['Not Meeting Target','Meeting Target', 
                            'Approaching Target','Exceeding Target'],
                            ['0','2','1','3']))
    school_explorer[feature].replace(feature_pairs,inplace=True)
    school_explorer[feature]=school_explorer[feature].astype(int)    
category_list=list(category_list+rating_list)
numeric_list=list(school_explorer.columns[[4,5]+list(range(13,24))+[25,27,29,31,33]+list(range(38,158))])    
print('Number of Missing Values: ',sum(school_explorer.isna().sum())) 


# In[ ]:


sat_list=['DBN','Number of students who registered for the SHSAT',
          'Number of students who took the SHSAT']
d5_shsat_2016=d5_shsat[sat_list][d5_shsat['Year of SHST']==2016].groupby(['DBN'],as_index=False).agg(np.sum)
d5_shsat_2016['Took SHSAT %']=d5_shsat_2016['Number of students who took the SHSAT']/d5_shsat_2016['Number of students who registered for the SHSAT']
d5_shsat_2016['Took SHSAT %']=d5_shsat_2016['Took SHSAT %'].fillna(0).apply(lambda x:round(x,3))
d5_shsat_2016.rename(columns={'DBN':'Location Code'},inplace=True)
d5_shsat_2016=pd.merge(school_explorer[['Location Code']+numeric_list+                         category_list+target_list],
         d5_shsat_2016,on='Location Code')
d5_shsat_2016.shape


# In[ ]:


features1=school_explorer[numeric_list].values
features2=school_explorer[numeric_list+category_list]
targets1=school_explorer['Average Math Proficiency'].values
targets2=school_explorer['Average ELA Proficiency'].values
features_enc=features2
encode=OneHotEncoder(sparse=False)
for column in category_list:
    encode.fit(features2[[column]])
    transform=encode.transform(features2[[column]])    
    transform=    pd.DataFrame(transform, 
                 columns=[(column+"_"+str(i)) 
                          for i in features2[column]\
                          .value_counts().index])
    transform=transform.set_index(features2.index.values)    
    features_enc=pd.concat([features_enc,transform],axis=1)
    features_enc=features_enc.drop(column,1)    
features2=features2.values
features_enc=features_enc.values


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Data Splitting for Neural Networks</h1>

# In[ ]:


# data = school_explorer
# features = numeric variables
# targets = Average Math Proficiency
X_train1,X_test1,y_train1,y_test1=train_test_split(features1,targets1,test_size=.2,random_state=1)
n=int(len(X_test1)/2)
X_valid1,y_valid1=X_test1[:n],y_test1[:n]
X_test1,y_test1=X_test1[n:],y_test1[n:]
[X_train1.shape,X_test1.shape,X_valid1.shape,
 y_train1.shape,y_test1.shape,y_valid1.shape]


# In[ ]:


# data = school_explorer
# eatures = numeric variables
# targets = Average ELA Proficiency
X_train2,X_test2,y_train2,y_test2=train_test_split(features1,targets2,test_size=.2,random_state=1)
n=int(len(X_test2)/2)
X_valid2,y_valid2=X_test2[:n],y_test2[:n]
X_test2,y_test2=X_test2[n:],y_test2[n:]
[X_train2.shape,X_test2.shape,X_valid2.shape,
 y_train2.shape,y_test2.shape,y_valid2.shape]


# In[ ]:


# data = school_explorer, 
# features = numeric & encoded categorical variables 
# targets = Average Math Proficiency
X_train3,X_test3,y_train3,y_test3=train_test_split(features_enc,targets1,test_size=.2,random_state=1)
n=int(len(X_test3)/2)
X_valid3,y_valid3=X_test3[:n],y_test3[:n]
X_test3,y_test3=X_test3[n:],y_test3[n:]
[X_train3.shape,X_test3.shape,X_valid3.shape,
 y_train3.shape,y_test3.shape,y_valid3.shape]


# In[ ]:


# data = school_explorer
# features = numeric & encoded categorical variables
# targets = Average ELA Proficiency
X_train4,X_test4,y_train4,y_test4=train_test_split(features_enc,targets2,test_size=.2,random_state=1)
n=int(len(X_test4)/2)
X_valid4,y_valid4=X_test4[:n],y_test4[:n]
X_test4,y_test4=X_test4[n:],y_test4[n:]
[X_train4.shape,X_test4.shape,X_valid4.shape,
 y_train4.shape,y_test4.shape,y_valid4.shape]


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Neural Network Models </h1>
# ### MLP => Numeric features

# In[ ]:


def mlp_model():
    model=Sequential()   
    model.add(Dense(138,input_dim=138))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dense(138*4))
    model.add(LeakyReLU(alpha=.02))    
    model.add(Dense(138*16))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dense(138*16))
    model.add(LeakyReLU(alpha=.02))    
    model.add(Dense(1))    
    model.compile(loss='mse',optimizer='rmsprop',
                  metrics=['mae'])
    return model
mlp_model1=mlp_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
history=mlp_model1.fit(X_train1,y_train1, 
                       epochs=200,batch_size=16,verbose=2,
                       validation_data=(X_valid1,y_valid1),
                       callbacks=[checkpointer,lr_reduction])


# In[ ]:


history_plot(history,50)
mlp_model1.load_weights(fw)
y_train_mlp1=mlp_model1.predict(X_train1)
y_valid_mlp1=mlp_model1.predict(X_valid1)
y_test_mlp1=mlp_model1.predict(X_test1)
ti='MLP; \nNumeric Features; \nAverage Math Proficiency'
scores(ti,y_train1,y_valid1,y_test1,
       y_train_mlp1,y_valid_mlp1,y_test_mlp1)


# In[ ]:


mlp_model2=mlp_model()
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
history=mlp_model2.fit(X_train2,y_train2, 
                       epochs=200,batch_size=16,verbose=2,
                       validation_data=(X_valid2,y_valid2),
                       callbacks=[checkpointer,lr_reduction])


# In[ ]:


history_plot(history,50)
mlp_model2.load_weights(fw)
y_train_mlp2=mlp_model2.predict(X_train2)
y_valid_mlp2=mlp_model2.predict(X_valid2)
y_test_mlp2=mlp_model2.predict(X_test2)
ti='MLP; \nNumeric Features; \nAverage ELA Proficiency'
scores(ti,y_train2,y_valid2,y_test2,
       y_train_mlp2,y_valid_mlp2,y_test_mlp2)


# ### MLP => Numeric & encoded categorical features

# In[ ]:


def cmlp_model():
    model=Sequential()    
    model.add(Dense(286,input_dim=286))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dense(286*4))
    model.add(LeakyReLU(alpha=.02))    
    model.add(Dense(286*16))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dense(1))    
    model.compile(loss='mse',optimizer='rmsprop',
                  metrics=['mae'])
    return model
mlp_model3=cmlp_model()


# In[ ]:


checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
history=mlp_model3.fit(X_train3,y_train3, 
                       epochs=200,batch_size=16,verbose=2,
                       validation_data=(X_valid3,y_valid3),
                       callbacks=[checkpointer,lr_reduction])


# In[ ]:


history_plot(history,50)
mlp_model3.load_weights(fw)
y_train_mlp3=mlp_model3.predict(X_train3)
y_valid_mlp3=mlp_model3.predict(X_valid3)
y_test_mlp3=mlp_model3.predict(X_test3)
ti='MLP; \nNumeric & Encoded Categorical Features; \nAverage Math Proficiency'
scores(ti,y_train3,y_valid3,y_test3,
       y_train_mlp3,y_valid_mlp3,y_test_mlp3)


# In[ ]:


mlp_model4=cmlp_model()
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,
                               verbose=2,factor=.75)
history=mlp_model4.fit(X_train4,y_train4, 
                       epochs=200,batch_size=16,verbose=2,
                       validation_data=(X_valid4,y_valid4),
                       callbacks=[checkpointer,lr_reduction])


# In[ ]:


history_plot(history,50)
mlp_model4.load_weights(fw)
y_train_mlp4=mlp_model4.predict(X_train4)
y_valid_mlp4=mlp_model4.predict(X_valid4)
y_test_mlp4=mlp_model4.predict(X_test4)
ti='MLP; \nNumeric & Encoded Categorical Features; \nAverage ELA Proficiency'
scores(ti,y_train4,y_valid4,y_test4,
       y_train_mlp4,y_valid_mlp4,y_test_mlp4)


# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'> &#x1F310; &nbsp; Predictions & Real Data</h1>
# ### Numeric features

# In[ ]:


plt.figure(figsize=(12,7))
plt.plot(y_test1[1:50],'-o',color='#3636ff',label='Real Data')
plt.plot(y_test_mlp1[1:50],'-o',color='#ff3636',label='MLP')
ti="Average Math Proficiency. NN Test Predictions vs Real Data"
plt.legend(); plt.title(ti);


# In[ ]:


plt.figure(figsize=(12,7))
plt.plot(y_test2[1:50],'-o',color='#3636ff',label='Real Data')
plt.plot(y_test_mlp2[1:50],'-o',color='#ff3636',label='MLP')
ti="Average ELA Proficiency. NN Test Predictions vs Real Data"
plt.legend(); plt.title(ti);


# ### Numeric & encoded categorical features

# In[ ]:


plt.figure(figsize=(12,7))
plt.plot(y_test3[1:50],'-o',color='#3636ff',label='Real Data')
plt.plot(y_test_mlp3[1:50],'-o',color='#ff3636',label='MLP')
ti="Average Math Proficiency. NN Test Predictions vs Real Data"
plt.legend(); plt.title(ti);


# In[ ]:


plt.figure(figsize=(12,7))
plt.plot(y_test4[1:50],'-o',color='#3636ff',label='Real Data')
plt.plot(y_test_mlp4[1:50],'-o',color='#ff3636',label='MLP')
ti="Average ELA Proficiency. NN Test Predictions vs Real Data"
plt.legend(); plt.title(ti);


# In this case, neural networks were used to predict the indicators of educational outcomes. 
# 
# It is difficult to achieve high accuracy here in general since the ultimate goal depends not only on the financing, social environment and professionalism of the staff but also on the personal qualities of the students. 
# 
# Adding category variables does not improve the accuracy of predictions because these indicators depend not exactly on geolocations but on the well-being of the surrounding area.
# 
# I believe that it is possible to reach at least the same accuracy with neural networks as with regression algorithms like `Gradient Boosting` or `Random Forest`.

# <h1 class='font-effect-3d' style='color:#37c9e1; font-family:Orbitron;'>&#x1F310; &nbsp; Let's Go Ahead</h1>
# It's time to move to the next step.
# 
# &#x1F4D8; &nbsp; [PASSNYC. 32 School Districts and D5. Part 2](https://www.kaggle.com/olgabelitskaya/passnyc-32-school-districts-and-d5-part-2)
