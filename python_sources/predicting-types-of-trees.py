#!/usr/bin/env python
# coding: utf-8

# ### Predicting types of trees in an area based on various geographic features
# 
# Forest Cover Types:
# 
# 1 - Spruce/Fir 2 - Lodgepole Pine 3 - Ponderosa Pine 4 - Cottonwood/Willow 5 - Aspen 6 - Douglas-fir 7 - Krummholz

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import pandas_profiling as pp
from sklearn.compose import ColumnTransformer
import  tensorflow as tf
import keras
from tpot import TPOTClassifier


# In[ ]:


df = pd.read_csv('../input/learn-together/train.csv')
df.head()


# In[ ]:


df_test = pd.read_csv('../input/learn-together/test.csv')
df_test.head()


# ### Dimensions of Data
# We have 15,120 rows and 56 columns in our training set while in our testing set we have 565,892 rows and 55 columns so the problem here is we have too many rows and our algorithms may take too long also in columns we have 56 this means that some algorithms can be distracted or suffer poor performance due to the curse of dimensionality.

# In[ ]:


print('Train size: ',df.shape)
print('Test size: ', df_test.shape)


# ### Data Description
# The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:
# 
# * 1 - Spruce/Fir
# * 2 - Lodgepole Pine
# * 3 - Ponderosa Pine
# * 4 - Cottonwood/Willow
# * 5 - Aspen
# * 6 - Douglas-fir
# * 7 - Krummholz
# 
# The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).

# ### Data Fields
# * 1. Elevation - Elevation in meters Aspect - Aspect in degrees azimuth Slope - Slope in degrees 
# * 2. Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features 
# * 3. Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features 
# * 4. Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway 
# * 5. Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice 
# * 6. Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice 
# * 7. Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice 
# * 8. Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points 
# * 9. Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation 
# * 10. Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation 
# * 11. Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation 
# 
# The wilderness areas are:
# * 1 - Rawah Wilderness Area 
# * 2 - Neota Wilderness Area 
# * 3 - Comanche Peak Wilderness Area 
# * 4 - Cache la Poudre Wilderness Area 
# 
# The soil types are:
# * 1 Cathedral family - Rock outcrop complex, extremely stony. 
# * 2 Vanet - Ratake families complex, very stony. 
# * 3 Haploborolis - Rock outcrop complex, rubbly.
# * 4 Ratake family - Rock outcrop complex, rubbly. 
# * 5 Vanet family - Rock outcrop complex complex, rubbly. 
# * 6 Vanet - Wetmore families - Rock outcrop complex, stony. 
# * 7 Gothic family. 
# * 8 Supervisor - Limber families complex. 
# * 9 Troutville family, very stony. 
# * 10 Bullwark - Catamount families - Rock outcrop complex, rubbly. 
# * 11 Bullwark - Catamount families - Rock land complex, rubbly. 
# * 12 Legault family - Rock land complex, stony. 
# * 13 Catamount family - Rock land - Bullwark family complex, rubbly. 
# * 14 Pachic Argiborolis - Aquolis complex. 
# * 15 unspecified in the USFS Soil and ELU Survey. 
# * 16 Cryaquolis - Cryoborolis complex. 
# * 17 Gateview family - Cryaquolis complex. 
# * 18 Rogert family, very stony. 
# * 19 Typic Cryaquolis - Borohemists complex. 
# * 20 Typic Cryaquepts - Typic Cryaquolls complex. 
# * 21 Typic Cryaquolls - Leighcan family, till substratum complex. 
# * 22 Leighcan family, till substratum, extremely bouldery. 
# * 23 Leighcan family, till substratum - Typic Cryaquolls complex. 
# * 24 Leighcan family, extremely stony. 
# * 25 Leighcan family, warm, extremely stony. 
# * 26 Granile - Catamount families complex, very stony. 
# * 27 Leighcan family, warm - Rock outcrop complex, extremely stony. 
# * 28 Leighcan family - Rock outcrop complex, extremely stony. 
# * 29 Como - Legault families complex, extremely stony. 
# * 30 Como family - Rock land - Legault family complex, extremely stony. 
# * 31 Leighcan - Catamount families complex, extremely stony. 
# * 32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony. 
# * 33 Leighcan - Catamount families - Rock outcrop complex, extremely stony. 
# * 34 Cryorthents - Rock land complex, extremely stony. 
# * 35 Cryumbrepts - Rock outcrop - Cryaquepts complex. 
# * 36 Bross family - Rock land - Cryumbrepts complex, extremely stony. 
# * 37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony. 
# * 38 Leighcan - Moran families - Cryaquolls complex, extremely stony. 
# * 39 Moran family - Cryorthents - Leighcan family complex, extremely stony. 
# * 40 Moran family - Cryorthents - Rock land complex, extremely stony.
# 
# Checking the data type of each attribute**

# In[ ]:


df.info()


# No missing values. No categorical

# In[ ]:


df.isnull().mean()


# In[ ]:


df.describe().T


# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(50,35))
plt.title('Pearson Correlation of Features', y=1.05, size=50)
sns.heatmap(df.corr(),linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


df.duplicated().sum() #No Duplicate Data


# In[ ]:


## Checking if we have a balance dataset
target = df.Cover_Type.value_counts()
sns.countplot(x='Cover_Type', data=df)
plt.title('Class Distribution');
print(target) # Balanced Train Dataset.


# In[ ]:


pp.ProfileReport(df)


# # Obeservations:
# 1. Soil_Type7 and Soil_Type15 have zero values, so removing them.
# 1. We have 10.5% and 12.5% zeros for Horizontal_Distance_To_Hydrology and Vertical_Distance_To_Hydrology. We need to check those zero values.
# 1. Hillshade_9am and Hillshade_3pm are 80% correlated. Need to check.
# 1. Hillshade_Noon and Slope are -50% correlated. Need to check.
# 1. Hillshade_9am and Aspect are -58% correlated. Need to check.
# 1. Hillshade_3PM and Aspect are 66% correlated. Need to check.
# 1. Vertical_Distance_To_Hydrology and Horizontal_Distance_To_Hydrology are 63% correlated. Need to check.
# 1. Soil_Types are highly cardinal(40 unique types). Need to check to reduce the cardinality.

# In[ ]:


#Soil_Type7 and  Soil_Type15 have zero values, so removing them. 
df.drop(['Soil_Type7', 'Id','Soil_Type15'], axis=1, inplace=True)


# In[ ]:


df.Cover_Type.value_counts()


# In[ ]:


# separate intro train and test set

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['Cover_Type'], axis=1),  # just the features
    df['Cover_Type'],  # the target
    test_size=0.2,  # the percentage of obs in the test set
    random_state=42)  # for reproducibility

X_train.shape, X_test.shape


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(y=df.Hillshade_9am, x=df.Hillshade_3pm)
plt.xlabel("Hillshade_3pm")
plt.ylabel("Hillshade_9am")
plt.title("Hillshade_3pm VS Hillshade_9am")


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(y=df.Hillshade_Noon, x=df.Slope)
plt.xlabel("Slope")
plt.ylabel("Hillshade_Noon")
plt.title("Slope VS Hillshade_Noon")


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(x=df.Hillshade_9am, y=df.Aspect)
plt.ylabel("Aspect")
plt.xlabel("Hillshade_9am")
plt.title("Aspect VS Hillshade_9am")


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(x=df.Hillshade_3pm, y=df.Aspect)
plt.ylabel("Aspect")
plt.xlabel("Hillshade_3pm")
plt.title("Aspect VS Hillshade_3pm")


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(x=df.Vertical_Distance_To_Hydrology, y=df.Horizontal_Distance_To_Hydrology)
plt.ylabel("Horizontal_Distance_To_Hydrology")
plt.xlabel("Vertical_Distance_To_Hydrology")
plt.title("Horizontal_Distance_To_Hydrology VS Vertical_Distance_To_Hydrology")


# In[ ]:


plt.figure(figsize=(12,12))
cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points', 'Cover_Type']
sns.pairplot(df[cols][df.Cover_Type==4])


# 

# In[ ]:


#tpot = TPOTClassifier(generations=5,population_size=10,verbosity=2, n_jobs=-1)
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#print(tpot.score(X_train, y_train))
#tpot.export('tpot_tree_classification_pipeline.py')
#!cat tpot_tree_classification_pipeline.py
#tpot.evaluated_individuals_
#tpot.fitted_pipeline_
#print(classification_report(y_test, tpot.predict(X_test)))
#print(confusion_matrix(y_test, tpot.predict(X_test)))


# ## After several hyperparameter tuning and different classifications, below one is ging best results so far.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator


# Average CV score on the training set was:0.8458153791211421
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.25, min_samples_leaf=17, min_samples_split=6, n_estimators=100, subsample=0.8)),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=18, min_samples_split=17, n_estimators=100)
)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)


# In[ ]:


print(exported_pipeline.score(X_test, y_test))
print(exported_pipeline.score(X_train, y_train))


# In[ ]:


print(classification_report(y_test, exported_pipeline.predict(X_test)))
print(confusion_matrix(y_test, exported_pipeline.predict(X_test)))


# In[ ]:


result_final = exported_pipeline.predict(df_test.drop(['Soil_Type7', 'Id', 'Soil_Type15'], axis=1))
result_final_proba = exported_pipeline.predict_proba(df_test.drop(['Soil_Type7', 'Id', 'Soil_Type15'], axis=1))
#df_test.drop(['Soil_Type7', 'Id', 'Soil_Type15'], axis=1, inplace=True)


# In[ ]:


result_final_proba[0]


# In[ ]:


# Save test predictions to file
#output = pd.DataFrame({'ID': df_test.Id,
#                       'Cover_Type': result_final})
#output.to_csv('submission.csv', index=False)


# In[ ]:


#pd.DataFrame(output).iloc[0]


# In[ ]:


#result_final_proba[0]


# Ckearly my model is overfitting. Will do some feature engineering and predict.

# ### Feature Engineering

# In[ ]:


import seaborn as sns
cols = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Cover_Type']
sns.pairplot(df[cols], hue="Cover_Type")


# As you can see there are some important relations that the model can infere from these new features according to the plots and also the correlation matrix and the heatmap. I will now add these features to the training data and the test data. I have read many resources as this study, this grat course and from that great kernel.
# 
# Also it seems that the vertical distance contain some negative number and it gave me better performance when taken the absolute for the column. It is really important to notice that Tree based models only fits vertical and horizontal lines so it is very important to engineer some oblique or tilted features like slope and etc... .

# In[ ]:


train = df.copy()
del df


# In[ ]:


test = df_test.copy()
del df_test


# In[ ]:


# train.head()
train['HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])
train['Neg_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

train['Neg_Elevation_Vertical'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']
train['Elevation_Vertical'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']

train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] )

train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])
train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])

train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

train['Slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)
train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways'])
train['Mean_Fire_Hyd']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'])

train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])

train['Neg_EHyd'] = train.Elevation-train.Horizontal_Distance_To_Hydrology


test['HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])
test['Neg_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

test['Neg_Elevation_Vertical'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']
test['Elevation_Vertical'] = test['Elevation'] + test['Vertical_Distance_To_Hydrology']

test['mean_hillshade'] = (test['Hillshade_9am']  + test['Hillshade_Noon']  + test['Hillshade_3pm'] )

test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])
test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

test['Slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)
test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways'])
test['Mean_Fire_Hyd']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'])


test['Vertical_Distance_To_Hydrology'] = abs(test["Vertical_Distance_To_Hydrology"])

test['Neg_EHyd'] = test.Elevation-test.Horizontal_Distance_To_Hydrology


# Now we should seperate the training set from the labels and name them x and y then we will split them into training and test sets to be able to see how well it would do on unseen data which will give anestimate on how well it will do when testing on Kaggle test data. I will use the convention of using 80% of the data as training set and 20% for the test set.

# In[ ]:


train.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x = train.drop(['Cover_Type'], axis = 1)

y = train['Cover_Type']
print( y.head() )

x_train, x_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.05, random_state=42 )


# It is important to know if the number of points in the classes are balanced. If the data is skewed then we will not be able to use accuracy as a performance metric since it will be misleading but if it is skewed we may use F-beta score or precision and recall. Precision or recall or F1 score. the choice depends on the problem itself. Where high recall means low number of false negatives , High precision means low number of false positives and F1 score is a trade off between them. You can refere to this article for more about precision and recall http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    
from sklearn import decomposition

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator


# Average CV score on the training set was:0.8458153791211421
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.25, min_samples_leaf=17, min_samples_split=6, n_estimators=100, subsample=0.8)),
    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.25, min_samples_leaf=18, min_samples_split=17, n_estimators=100)
)

exported_pipeline.fit(x_train, y_train)
results = exported_pipeline.predict(x_test)


# In[ ]:


print(exported_pipeline.score(x_test, y_test))
print(exported_pipeline.score(x_train, y_train))


# In[ ]:


print(classification_report(y_test, exported_pipeline.predict(x_test)))
print(confusion_matrix(y_test, exported_pipeline.predict(x_test)))


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

#uncomment the commented code and uncomment the commented to perform gridsearchCV
from xgboost import XGBClassifier as xgb

clf = ExtraTreesClassifier(n_estimators=950, random_state=0)
from sklearn.svm import LinearSVC
from mlxtend.classifier import StackingCVClassifier

c1 = ExtraTreesClassifier(n_estimators=500,bootstrap=True) 
c2= RandomForestClassifier(n_estimators=500,bootstrap=True)
c3=xgb();
meta = LinearSVC()
sclf = StackingCVClassifier(classifiers=[c1,c2,c3],use_probas=True,meta_classifier=meta)


# In[ ]:


sclf.fit(x_train, y_train)
print('Accuracy of classifier on training set: {:.2f}'.format(sclf.score(x_train, y_train) * 100))
print('Accuracy of classifier on test set: {:.2f}'.format(sclf.score(x_test, y_test) * 100))


# In[ ]:


test.head()


# In[ ]:


test.columns


# In[ ]:


test.head()

id = test['Id']
test.drop(['Id', 'Soil_Type7', 'Soil_Type15'] , inplace = True , axis = 1)
test = scaler.transform(test)


# In[ ]:


predictions = sclf.predict(test)


# In[ ]:


out = pd.DataFrame({'Id': id,'Cover_Type': predictions})
out.to_csv('submission.csv', index=False)
out.head(5)


# In[ ]:





# In[ ]:





# In[ ]:




