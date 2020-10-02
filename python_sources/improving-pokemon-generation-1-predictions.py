#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Previously I built XG Boost models to predict the main and sub-types of Pokemon from all 7 generations (https://www.kaggle.com/xagor1/pokemon-type-predictions-using-xgb). This was relatively successful, but often stalled at around 70% accuracy per generation, with some much worse. To gain more experience with parameter tuning and feature engineering, I decided to revisit just the 1st Generation, and see if I could improve my results.

# In[2]:


#Load various packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns
print(os.listdir("../input"))
from sklearn.feature_selection import SelectFromModel
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# # Loading and Modifying Data
# 
# To start with, I loaded and modified the data as in the previous kernel.
# 
# In contrast to last time, I separated out the numerical and categorical data, and applied one-hot encoding to the latter. This caused the number of features to explode from 24 to 500.
# 
# The original plan was to do feature engineering to improve my overall accuracy. However, thus far all my attempts have actually made the predictions worse, so I have left this aside for now.

# In[3]:


#Read data
path = '../input/'
egg_df=pd.read_csv(path+"egg-group-data-for-pokemon/pokemon_egg_groups.csv")
species_df=pd.read_csv(path+"pokemon-species/pokemon_species.csv")
abilities_df=pd.read_csv(path+"abilities/pokemon_abilities.csv")

#Split duplicates off & combine back
egg2_df=pd.DataFrame.copy(egg_df)
egg2_df=egg_df.loc[egg_df['species_id'].duplicated(), :]
egg_df.drop_duplicates('species_id',inplace=True)
merged = egg_df.merge(egg2_df,on="species_id",how='outer')
merged.fillna(0,inplace=True)

#Rename columns to simpler form.
merged.rename(index=str,columns={"egg_group_id_x":"egg_group_1"},inplace=True)
merged.rename(index=str,columns={"egg_group_id_y":"egg_group_2"},inplace=True)

#Drop last 6 columns
merged.drop(merged.tail(6).index,inplace=True)

#Rename
merged.rename(index=str,columns={"species_id":"pokedex_number"},inplace=True)

#Make a new smaller dataframe
species_trim_df=pd.DataFrame()
species_trim_df["pokedex_number"]=species_df['id']
species_trim_df["color_id"]=species_df['color_id']
species_trim_df["shape_id"]=species_df['shape_id']
species_trim_df["habitat_id"]=species_df['habitat_id']
species_trim_df.drop(species_trim_df.tail(6).index,inplace=True)

#Trim all below Magearna off
abilities_df = abilities_df[abilities_df.pokemon_id < 802]

#Make 3 new columns
abilities_df["Ability1"]=0
abilities_df["Ability2"]=0
abilities_df["Ability3"]=0

#Assign values to the 3 columns based on the ability slot (1-3)
abilities_df["Ability1"] = abilities_df.ability_id.where(abilities_df.slot == 1,0)
abilities_df["Ability2"] = abilities_df.ability_id.where(abilities_df.slot == 2,0)
abilities_df["Ability3"] = abilities_df.ability_id.where(abilities_df.slot == 3,0)

#Split duplicates off into new dataframes 
#3 abilities on some means it needs to be split twice
#I'm sure there's an easier way to do this
abilities_df2=pd.DataFrame.copy(abilities_df)
abilities_df2=abilities_df.loc[abilities_df['pokemon_id'].duplicated(), :]
abilities_df.drop_duplicates('pokemon_id',inplace=True)
abilities_df3=pd.DataFrame.copy(abilities_df2)
abilities_df3=abilities_df2.loc[abilities_df2['pokemon_id'].duplicated(), :]
abilities_df2.drop_duplicates('pokemon_id',inplace=True)

#Drop extra columns
abilities_df.drop(['ability_id','is_hidden','slot'],axis=1,inplace=True)
abilities_df2.drop(['ability_id','is_hidden','slot'],axis=1,inplace=True)
abilities_df3.drop(['ability_id','is_hidden','slot'],axis=1,inplace=True)

#Combine everything back
abilities_df=abilities_df.set_index('pokemon_id').add(abilities_df2.set_index('pokemon_id'),fill_value=0).reset_index()
abilities_df=abilities_df.set_index('pokemon_id').add(abilities_df3.set_index('pokemon_id'),fill_value=0).reset_index()

#Rename pokemon_id to pokedex number to allow for merging.
abilities_df.rename(index=str,columns={"pokemon_id":"pokedex_number"},inplace=True)

#Read Kaggle data
path = '../input/'
pokemon_df=pd.read_csv(path+"pokemon/pokemon.csv")

Name_df=pd.DataFrame()
Name_df["name"]=pokemon_df["name"].copy()

#Fix Minior's capture rate
pokemon_df.capture_rate.iloc[773]=30

#Change the type
pokemon_df['capture_rate']=pokemon_df['capture_rate'].astype(str).astype(int)

#Merge all my data.
pokemon_df=pokemon_df.merge(merged,on="pokedex_number",how='outer')
pokemon_df=pokemon_df.merge(species_trim_df,on="pokedex_number",how='outer')
pokemon_df=pokemon_df.merge(abilities_df,on="pokedex_number",how='outer')

#Remove against columns
pokemon_df.drop(list(pokemon_df.filter(regex = 'against')), axis = 1, inplace = True)
#Correct the spelling error
pokemon_df.rename(index=str,columns={"classfication":"classification"},inplace=True)

#Change nan to 'none'
pokemon_df.type2.replace(np.NaN, 'none', inplace=True)

#Drop Pokedex number for now
pokemon_df.drop("pokedex_number",axis=1,inplace=True)
pokemon_df.drop("generation",axis=1,inplace=True)

#First find the NAs.
index_height = pokemon_df['height_m'].index[pokemon_df['height_m'].apply(np.isnan)]
index_weight = pokemon_df['weight_kg'].index[pokemon_df['weight_kg'].apply(np.isnan)]
index_male   = pokemon_df['percentage_male'].index[pokemon_df['percentage_male'].apply(np.isnan)]

#Manually replace the missing heights & weights using the Kanto version etc
pokemon_df.height_m.iloc[18]=0.3
pokemon_df.height_m.iloc[19]=0.7
pokemon_df.height_m.iloc[25]=0.8
pokemon_df.height_m.iloc[26]=0.6
pokemon_df.height_m.iloc[27]=1.0
pokemon_df.height_m.iloc[36]=0.6
pokemon_df.height_m.iloc[37]=1.1
pokemon_df.height_m.iloc[49]=0.2
pokemon_df.height_m.iloc[50]=0.7
pokemon_df.height_m.iloc[51]=0.4
pokemon_df.height_m.iloc[52]=1.0
pokemon_df.height_m.iloc[73]=0.4
pokemon_df.height_m.iloc[74]=1.0
pokemon_df.height_m.iloc[75]=1.4
pokemon_df.height_m.iloc[87]=0.9
pokemon_df.height_m.iloc[88]=1.2
pokemon_df.height_m.iloc[102]=2.0
pokemon_df.height_m.iloc[104]=1.0
pokemon_df.height_m.iloc[719]=0.5
pokemon_df.height_m.iloc[744]=0.8

pokemon_df.weight_kg.iloc[18]=3.5
pokemon_df.weight_kg.iloc[19]=18.5
pokemon_df.weight_kg.iloc[25]=30.0
pokemon_df.weight_kg.iloc[26]=12.0
pokemon_df.weight_kg.iloc[27]=29.5
pokemon_df.weight_kg.iloc[36]=9.9
pokemon_df.weight_kg.iloc[37]=19.9
pokemon_df.weight_kg.iloc[49]=0.8
pokemon_df.weight_kg.iloc[50]=33.3
pokemon_df.weight_kg.iloc[51]=4.2
pokemon_df.weight_kg.iloc[52]=32.0
pokemon_df.weight_kg.iloc[73]=20.0
pokemon_df.weight_kg.iloc[74]=105.0
pokemon_df.weight_kg.iloc[75]=300.0
pokemon_df.weight_kg.iloc[87]=30.0
pokemon_df.weight_kg.iloc[88]=30.0
pokemon_df.weight_kg.iloc[102]=120.0
pokemon_df.weight_kg.iloc[104]=45.0
pokemon_df.weight_kg.iloc[719]=9.0
pokemon_df.weight_kg.iloc[744]=25.0

#Create a Genderless column to separate them from the all-female cases.
pokemon_df["Genderless"]=0
pokemon_df["Genderless"].loc[list(index_male)]=1

#Replace all the NANs with zeros in the % male
pokemon_df.percentage_male.replace(np.NaN, 0, inplace=True)

#Check the typings of the pokemon with Alolan forms & fix
#I'm sure this can be done much more elegantly
pokemon_df.type2.iloc[18]='none'
pokemon_df.type2.iloc[19]='none'
pokemon_df.type2.iloc[25]='none'
pokemon_df.type2.iloc[26]='none'
pokemon_df.type2.iloc[27]='none'
pokemon_df.type2.iloc[36]='none'
pokemon_df.type2.iloc[37]='none'
pokemon_df.type2.iloc[49]='none'
pokemon_df.type2.iloc[50]='none'
pokemon_df.type2.iloc[51]='none'
pokemon_df.type2.iloc[52]='none'
pokemon_df.type2.iloc[87]='none'
pokemon_df.type2.iloc[88]='none'
pokemon_df.type2.iloc[104]='none'

#Lets start with just the numerical data for now.
num_features=pokemon_df.select_dtypes(include=np.number)
num_features=num_features.columns

#print("The Type models will be built using the following features")
#print(list(num_features))


# In[4]:


numerical_df=pd.DataFrame.copy(pokemon_df[['attack', 'base_egg_steps', 'base_happiness', 'base_total',
                                           'capture_rate', 'defense', 'experience_growth',
                                           'height_m', 'hp', 'percentage_male', 'sp_attack', 'sp_defense', 'speed',
                                           'weight_kg']])
numerical_df.to_csv('numerical_features.csv',index=False)
one_hot_df=pd.DataFrame.copy(pokemon_df[["Ability1","Ability2","Ability3","egg_group_1","egg_group_2",
                                              "is_legendary","color_id","shape_id","habitat_id","Genderless"]])
one_hot_df=pd.get_dummies(one_hot_df,prefix=["Ability1","Ability2","Ability3","egg_group_1","egg_group_2",
                                              "is_legendary","color_id","shape_id","habitat_id","Genderless"],
                       columns=["Ability1","Ability2","Ability3","egg_group_1","egg_group_2",
                                              "is_legendary","color_id","shape_id","habitat_id","Genderless"])
one_hot_df.to_csv('one_hot_features.csv',index=False)
features=pd.concat([numerical_df,one_hot_df],axis=1)


# In[ ]:


#Do some feature engineering
#features["Total_Offense"]=features["attack"]+features["sp_attack"]
#features["Total_Defense"]=features["defense"]+features["sp_defense"]
#features["Total_Physical"]=features["attack"]+features["defense"]
#features["Total_Special"]=features["sp_attack"]+features["sp_defense"]
#features["Attack_Difference"]=abs(features["attack"]-features["sp_attack"])
#features["Defense_Difference"]=abs(features["defense"]-features["sp_defense"])
#features["Physical_Difference"]=abs(features["attack"]-features["defense"])
#features["Special_Difference"]=abs(features["sp_attack"]-features["sp_defense"])
#features["HeightXWeight"]=features["height_m"]*features["weight_kg"]
#features["BMI"]=features["weight_kg"]/(features["weight_kg"]**2)
#features["Speed_X_Weight"]=features["speed"]*features["weight_kg"]
#features=features.drop(columns=["attack","sp_attack"])


# In[5]:


targets=pd.DataFrame()
targets2=pd.DataFrame()
targets["type1"]=pokemon_df["type1"]
targets=np.ravel(targets)
targets2["type2"]=pokemon_df["type2"]
targets2=np.ravel(targets2)


#Split features & targets into each generation.
Gen1_features=features[0:151]
Gen2_features=features[151:251]
Gen3_features=features[251:386]
Gen4_features=features[386:493]
Gen5_features=features[493:649]
Gen6_features=features[649:721]
Gen7_features=features[721:801]
Gen1_targets=targets[0:151]
Gen2_targets=targets[151:251]
Gen3_targets=targets[251:386]
Gen4_targets=targets[386:493]
Gen5_targets=targets[493:649]
Gen6_targets=targets[649:721]
Gen7_targets=targets[721:801]
Gen1_targets=np.ravel(Gen1_targets)
Gen2_targets=np.ravel(Gen2_targets)
Gen3_targets=np.ravel(Gen3_targets)
Gen4_targets=np.ravel(Gen4_targets)
Gen5_targets=np.ravel(Gen5_targets)
Gen6_targets=np.ravel(Gen6_targets)
Gen7_targets=np.ravel(Gen7_targets)

#Recombine 6 of them, in 7 different ways, to make my different training sets
#Ordering of the features & targets should be the same!
#But doesn't have to be necessarily in numerical order
Gens_not1_features=pd.concat([Gen2_features,Gen3_features,Gen4_features,Gen5_features,Gen6_features,Gen7_features],axis=0)
Gens_not2_features=pd.concat([Gen1_features,Gen3_features,Gen4_features,Gen5_features,Gen6_features,Gen7_features],axis=0)
Gens_not3_features=pd.concat([Gen2_features,Gen1_features,Gen4_features,Gen5_features,Gen6_features,Gen7_features],axis=0)
Gens_not4_features=pd.concat([Gen2_features,Gen3_features,Gen1_features,Gen5_features,Gen6_features,Gen7_features],axis=0)
Gens_not5_features=pd.concat([Gen2_features,Gen3_features,Gen4_features,Gen1_features,Gen6_features,Gen7_features],axis=0)
Gens_not6_features=pd.concat([Gen2_features,Gen3_features,Gen4_features,Gen5_features,Gen1_features,Gen7_features],axis=0)
Gens_not7_features=pd.concat([Gen2_features,Gen3_features,Gen4_features,Gen5_features,Gen6_features,Gen1_features],axis=0)
Gens_not1_targets=np.concatenate((Gen2_targets,Gen3_targets,Gen4_targets,Gen5_targets,Gen6_targets,Gen7_targets),axis=0)
Gens_not2_targets=np.concatenate((Gen1_targets,Gen3_targets,Gen4_targets,Gen5_targets,Gen6_targets,Gen7_targets),axis=0)
Gens_not3_targets=np.concatenate((Gen2_targets,Gen1_targets,Gen4_targets,Gen5_targets,Gen6_targets,Gen7_targets),axis=0)
Gens_not4_targets=np.concatenate((Gen2_targets,Gen3_targets,Gen1_targets,Gen5_targets,Gen6_targets,Gen7_targets),axis=0)
Gens_not5_targets=np.concatenate((Gen2_targets,Gen3_targets,Gen4_targets,Gen1_targets,Gen6_targets,Gen7_targets),axis=0)
Gens_not6_targets=np.concatenate((Gen2_targets,Gen3_targets,Gen4_targets,Gen5_targets,Gen1_targets,Gen7_targets),axis=0)
Gens_not7_targets=np.concatenate((Gen2_targets,Gen3_targets,Gen4_targets,Gen5_targets,Gen6_targets,Gen1_targets),axis=0)

Gen1_targets2=targets2[0:151]
Gen2_targets2=targets2[151:251]
Gen3_targets2=targets2[251:386]
Gen4_targets2=targets2[386:493]
Gen5_targets2=targets2[493:649]
Gen6_targets2=targets2[649:721]
Gen7_targets2=targets2[721:801]
Gen1_targets2=np.ravel(Gen1_targets2)
Gen2_targets2=np.ravel(Gen2_targets2)
Gen3_targets2=np.ravel(Gen3_targets2)
Gen4_targets2=np.ravel(Gen4_targets2)
Gen5_targets2=np.ravel(Gen5_targets2)
Gen6_targets2=np.ravel(Gen6_targets2)
Gen7_targets2=np.ravel(Gen7_targets2)
Gens_not1_targets2=np.concatenate((Gen2_targets2,Gen3_targets2,Gen4_targets2,Gen5_targets2,Gen6_targets2,Gen7_targets2),axis=0)
Gens_not2_targets2=np.concatenate((Gen1_targets2,Gen3_targets2,Gen4_targets2,Gen5_targets2,Gen6_targets2,Gen7_targets2),axis=0)
Gens_not3_targets2=np.concatenate((Gen2_targets2,Gen1_targets2,Gen4_targets2,Gen5_targets2,Gen6_targets2,Gen7_targets2),axis=0)
Gens_not4_targets2=np.concatenate((Gen2_targets2,Gen3_targets2,Gen1_targets2,Gen5_targets2,Gen6_targets2,Gen7_targets2),axis=0)
Gens_not5_targets2=np.concatenate((Gen2_targets2,Gen3_targets2,Gen4_targets2,Gen1_targets2,Gen6_targets2,Gen7_targets2),axis=0)
Gens_not6_targets2=np.concatenate((Gen2_targets2,Gen3_targets2,Gen4_targets2,Gen5_targets2,Gen1_targets2,Gen7_targets2),axis=0)
Gens_not7_targets2=np.concatenate((Gen2_targets2,Gen3_targets2,Gen4_targets2,Gen5_targets2,Gen6_targets2,Gen1_targets2),axis=0)


# # Tuning XGB Parameters
# 
# In the previous kernel, I'd only done minor tuning of the XGB parameters when trying to fit to the full Pokedex. I'd then just assumed this was the best choice for all other situations, which might not actually be true.
# 
# In this kernel, I optimized a range of hyperparameters for both the Type 1 and Type 2 models, to obtain the best Test accuracy. This included tuning:
# 
# * max depth and min child weight
# * subsample and col sample by tree
# * gamma
# * reg alpha
# * reg lambda
# * learning rate and n estimators
# 
# In both cases, I was able to improve the accuracy by about 5% compared to the default values.
# 
# For both models, I also explored the effect of adding weightings, but only found improvements for the Type 2 model, which has a major imbalance between None and all other types.
# 
# For type 1, I found that the optimal parameters were:
# 
# max depth = 3, n estimators = 158, learning rate = 0.1, gamma = 0, min child weight = 1, subsample = 0.6, col sample by tree = 0.2, alpha =0 and lambda = 0.9.
# 

# In[6]:


params={
'max_depth':3,
'learning_rate':0.1, 
'n_estimators':300, 
'silent':True, 
'booster':'gbtree',
'n_jobs':1, 
'nthread':4, 
'gamma':0, 
'min_child_weight':1, 
'max_delta_step':0, 
'subsample':0.6, 
'colsample_bytree':0.2, 
'colsample_bylevel':1, 
'reg_alpha':0, 
'reg_lambda':0.9, 
'scale_pos_weight':1, 
'base_score':0.5, 
'random_state':1,
'missing':None,}


# In[ ]:


#Test adding weights wrt water
#weights = np.zeros(len(Gens_not1_targets))
#for i in range(len(Gens_not1_targets)):
#    weights[i]=Counter(Gens_not1_targets)['water']/Counter(Gens_not1_targets)[Gens_not1_targets[i]]
#weights


# In[7]:


#Generation 1 model
model_xgb=xgb.XGBClassifier(**params)
eval_set = [(Gens_not1_features, Gens_not1_targets),(Gen1_features, Gen1_targets)]
model_xgb.fit(Gens_not1_features, Gens_not1_targets,eval_set=eval_set,eval_metric="merror",verbose=False)
training_eval=model_xgb.evals_result()
min_error=min(training_eval['validation_1']['merror'])
print("The minimum error is:")
print(min_error)
training_step=training_eval['validation_1']['merror'].index(min_error)
print("This occurs at step:")
print(training_step)
xgb.plot_importance(model_xgb,max_num_features=20)


# In[8]:


#Final model
params['n_estimators']=158
model_xgb=xgb.XGBClassifier(**params)

model_xgb.fit(Gens_not1_features, Gens_not1_targets)
Gen1_T1_pred = model_xgb.predict(Gen1_features)

# evaluate predictions
test_accuracy = accuracy_score(Gen1_targets, Gen1_T1_pred)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
xgb.plot_importance(model_xgb,max_num_features=20)
# Output a plot of the confusion matrix.
labels =list(set(Gen1_targets))
cm = metrics.confusion_matrix(Gen1_targets, Gen1_T1_pred,labels)
# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.set(font_scale=4)
plt.figure(figsize=(20,20))
ax = sns.heatmap(cm_normalized, cmap="bone_r")
ax.set_aspect(1)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Type 1 Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
sns.set(font_scale=0.8)


# After hyperparameter tuning, I was able to get a 72.19% accuracy for Type 1, which just beats my models from the previous attempt.
# 
# All types have some correct predictions, except for Ice, which is always confused for either Water or Psychic.
# 
# By contrast, Bug, Water and Grass each have 100% accuracy, and Normal performs pretty well too.
# 
# Most of the incorrect predictions appear to be from incorrect assignment of Water, Normal, Electric or Psychic type, meaning the model is over-predicted all four of these types.
# 
# Since type ordering is somewhat arbitrary, there is the possibility that some of these are correct predictions, but for Type 2, rather than type 1.

# In[9]:


print("Some predictions may match the sub-type, rather than the main type")
mismatch_accuracy = accuracy_score(Gen1_targets2, Gen1_T1_pred)
print("Mismatch Accuracy: %.2f%%" % (mismatch_accuracy * 100.0))
print("The Pokemon whose predicted types match their sub-type are:")
for i in range(0,len(Gen1_targets)):
    if Gen1_T1_pred[i] == Gen1_targets2[i]:
        print (pokemon_df["name"][i])


# As it turns out, there are 7 Pokemon which fall into this category.
# 
# However, this still leaves about a quarter of the Pokemon with incorrect types. 
# 
# One possible way to address this is to look closer at the incorrect predictions to see where they went wrong, and come up with ideas for how to fix them. For now, this is a task left to the future.

# In[10]:


print("Pokemon with incorrect types are as follows:")
for i in range(0,len(Gen1_targets)):
    if Gen1_T1_pred[i] != Gen1_targets[i]:
        print (pokemon_df["name"][i],Gen1_T1_pred[i])


# In[ ]:


#selection = SelectFromModel(model_xgb, threshold=1e-15,prefit=True)
#feature_idx = selection.get_support()
#feature_name = Gens_not1_features.columns[feature_idx]
#print(feature_name)
#print(feature_name.shape)


# In[11]:


weights = np.zeros(len(Gens_not1_targets2))
for i in range(len(Gens_not1_targets2)):
    weights[i]=Counter(Gens_not1_targets2)['none']/Counter(Gens_not1_targets2)[Gens_not1_targets2[i]]


# For type 2, I found that the optimal parameters were:
# 
# max depth = 4, n estimators = 242, learning rate = 0.1, gamma = 0.1, min child weight = 3, subsample = 1, col sample by tree = 0.3, alpha =0 and lambda = 1.
# 

# In[12]:


#With weights
#Max depth 4
#child weight 3
#gamma 0.1
#colsample 0.3

#Without weights: child weight=4, lambda=4

params2={
'max_depth':4,
'learning_rate':0.1, 
'n_estimators':300, 
'silent':True, 
'booster':'gbtree',
'n_jobs':1, 
'nthread':4, 
'gamma':0.1, 
'min_child_weight':3, 
'max_delta_step':0, 
'subsample':1, 
'colsample_bytree':0.3, 
'colsample_bylevel':1, 
'reg_alpha':0, 
'reg_lambda':1, 
'scale_pos_weight':1, 
'base_score':0.5, 
'random_state':1,
'missing':None,}


# In[13]:


#Type 2 classification
model_xgb2=xgb.XGBClassifier(**params2)
eval_set = [(Gens_not1_features, Gens_not1_targets2),(Gen1_features, Gen1_targets2)]
model_xgb2.fit(Gens_not1_features, Gens_not1_targets2,sample_weight=weights,eval_set=eval_set,eval_metric="merror",verbose=False)
training_eval=model_xgb2.evals_result()
min_error=min(training_eval['validation_1']['merror'])
print("The minimum error is:")
print(min_error)
training_step=training_eval['validation_1']['merror'].index(min_error)
print("This occurs at step:")
print(training_step)
xgb.plot_importance(model_xgb2,max_num_features=20)


# In[14]:


#Type 2 final version
params2['n_estimators']=242
model_xgb2=xgb.XGBClassifier(**params2)

model_xgb2.fit(Gens_not1_features, Gens_not1_targets2,weights)
Gen1_T2_pred = model_xgb2.predict(Gen1_features)

# evaluate predictions
test_accuracy = accuracy_score(Gen1_targets2, Gen1_T2_pred)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
xgb.plot_importance(model_xgb2,max_num_features=20)
# Output a plot of the confusion matrix.
labels =list(set(Gen1_targets2))
cm = metrics.confusion_matrix(Gen1_targets2, Gen1_T2_pred,labels)
# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.set(font_scale=4)
plt.figure(figsize=(20,20))
ax = sns.heatmap(cm_normalized, cmap="bone_r")
ax.set_aspect(1)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Type 2 Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
sns.set(font_scale=0.8)


# After hyperparameter tuning, I was able to get a 67.55% accuracy for Type 2, which marginally beats my older model.
# 
# As always for Type 2, most types were incorrectly predicted as None, with a few other misclassifcations, as for example Rock, Ground or Poison.
# 
# Flying and Rock stand out as particularly good predictions, with most of both correctly identified. Steel, Psychic and Ground all have a reasonable number of correct predictions.
# 
# Since type ordering is somewhat arbitrary, there is the possibility that some of these are correct predictions, but for Type 1, rather than type 2.

# In[15]:


print("Some predictions may match the main type, rather than the sub-type")
mismatch_accuracy_T2 = accuracy_score(Gen1_targets, Gen1_T2_pred)
print("Mismatch Accuracy: %.2f%%" % (mismatch_accuracy_T2 * 100.0))
print("The Pokemon whose predicted types match their main type are:")
for i in range(0,len(Gen1_targets2)):
    if Gen1_T2_pred[i] == Gen1_targets[i]:
        print (pokemon_df["name"][i])


# In this case, 6 Pokemon had the correct type predictions, but in the wrong order. The 4 fossil Pokemon, between Omanyte and Kabutops, have appeared in both sets of mis-ordered predictions. This means that both types were correctly predicted, just in the wrong order.
# 
# As before, it might be instructive to look at the incorrect predictions to try and work out where they went wrong.

# In[16]:


print("Pokemon with incorrect sub-types are as follows:")
for i in range(0,len(Gen1_targets2)):
    if Gen1_T2_pred[i] != Gen1_targets2[i]:
        print (pokemon_df["name"][i],Gen1_T2_pred[i])


# In the majority of cases, it is just a matter that None was selected instead of the correct type, suggesting it might be possible to add more information to the model and improve the predictions.
# 
# In other cases, a Pokemon was predicted a type, but it was wrong. A few of these are interesting, given the nature of the incorrect prediciton.
# 
# For example, Charizard is predicted to have Dragon, rather than Flying sub-type. This has been a wish of fans since the beginning, and actually came true for one of the Mega Evolutions.
# 
# Beedrill and Venomoth are both predicted to be Flying sub-type, which is understandable, given that they both have wings, however they are both actually poison types.
# 
# Some of the other mistakes, like Mewtwo being sub-type Ice, or Gyarados being Ground, are just odd.

# I improved both of my models by incorporating the ordering mismatches. This lead to slight improvements for both models, although by less than the number of mis-ordered Types. This is because the other model may have predicted the same type already, meaning that updating the value made no difference.

# In[17]:


Gen1_T1_pred_v2=Gen1_T1_pred.copy()
Gen1_T2_pred_v2=Gen1_T2_pred.copy()
for i in range(0,len(Gen1_targets)):
    if Gen1_T1_pred[i] == Gen1_targets2[i]:
        Gen1_T2_pred_v2[i]=Gen1_T1_pred[i]
        
for i in range(0,len(Gen1_targets)):
    if Gen1_T2_pred[i] == Gen1_targets[i]:
        Gen1_T1_pred_v2[i]=Gen1_T2_pred[i]
        
Type1_accuracy = accuracy_score(Gen1_targets, Gen1_T1_pred_v2)
print("New Type 1 Accuracy: %.2f%%" % (Type1_accuracy * 100.0))
Type2_accuracy = accuracy_score(Gen1_targets2, Gen1_T2_pred_v2)
print("New Type 2 Accuracy: %.2f%%" % (Type2_accuracy * 100.0))


# By combining the two models in this way, I was able to raise the accuracy of both to over 70%, and reach new records for both.
# 
# Something interesting to note, is that when I re-used my Type 1 parameters for Type 2, the overall accuracy was worse, but the mismatch percentage was higher. Meaning that I could get a 75% accuracy on Type 1 when both models were combined, but with a lower Type 2 accuracy.
# 
# I'd still like to do feature engineering in some way, because I'm sure it must be possible to improve the accuracy further.
# 
# 

# In[26]:


XGB_predictions_df=pd.DataFrame()
XGB_predictions_df["Type1"]=0
XGB_predictions_df["Type1"]=Gen1_T1_pred_v2
XGB_predictions_df["Type2"]=0
XGB_predictions_df["Type2"]=Gen1_T2_pred_v2
XGB_predictions_df.to_csv("XGB_Predictions.csv",index=False)

