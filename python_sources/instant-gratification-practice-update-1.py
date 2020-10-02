#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor


# Earlier I had done the Titanic competition, enjoyed it. Now want to try this one too as it is also a Classification problem. 

# # Problem
# 
# I don't know! I have absolutely no idea. The sample submission has two columns, the second column has 0.5 in each row. What does that mean? 
# 
# Anyhoo, let's at least have a look at the data first.

# In[ ]:


X_full = pd.read_csv("../input/train.csv")
X_test_full = pd.read_csv("../input/test.csv")
print("Loaded.")


# # Data preparation

# In[ ]:


#X_full.shape  
#(262144, 258)
#X_test.shape
#(131073, 257)

X_full.head(10)


# So that is a LOT of data! I don't see an index, and 'target' is our target (don't say duh).
# 
# Well good thing is there's no categorical data. Now let's check for missing values.

# In[ ]:


#X_full.isnull().sum() #it's not showing some columns... how do  know for sure?
cols_missingvals = [col for col in X_full.columns if X_full[col].isnull().any()]
print(cols_missingvals)


# So no missing values too! This dataset makes no sense. 

# # Model
# 
# Let's split the data into train and valid and then use XGBClassifier to create the model. (Switched to XGBRegressor later!!)
# 
# And what about the features? How do I know which ones to add and which ones to remove? I guess that's the challenge. Well, doesn't seem like my cup of tea. I'll come back later. For now, I'll just add all the columns (removing target and id).

# In[ ]:



y = X_full.target
X = X_full.drop(['target','id'], axis=1)
X_test = X_test_full.drop(['id'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

my_model = XGBRegressor(n_estimators=100, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)


# # Predictions
# 
# Alright time for predictions, fingers crossed.

# In[ ]:


predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))


# 0.5??? MAE is usually in tens of thousands...
# 
# So how do I improve the model? Obviously I can't do any data cleaning. So I'm guessing my model's accuracy is dependent entirely on my features. So how do I select the features?
# 
# I could go Brute-force (I see no other option). Remove one column at a time, remodel and check the accuracy. That would take way too long! I'll try few...

# # Submission
# 
# **NOTE**: I commented the old code. To uncomment, select all (Ctrl+A) and hit Ctrl+/

# In[ ]:


# preds = my_model.predict(X_valid)
# preds_test = my_model.predict(X_test)
# preds_test_rounded = np.around(preds_test,decimals=1)

# output = pd.DataFrame({'id': X_test_full.id,
#                        'target': preds_test_rounded})
# output.to_csv('submission.csv', index=False)

# a = pd.read_csv('submission.csv')
# a.head(10)


# In[ ]:


# b = pd.read_csv('../input/sample_submission.csv')
# b.head(10)


# # Notes
# 
# * First I had used XGBClassifier. But that gave me targets in 0s and 1s. Since the output required was in decimals (0.5), I assumed this problem to be a regression one and switched to XGBRegressor. Takes way longer!
# * The Numpy.around() method came in handy. Had to Google a lot! 
# * Private Score: 0.50036, Public Score: 0.50093
# * I'm not gonna continue practicing on this, takes way too much time. I might return to this later. I could switch back to XGBClassifier and quickly try out different feature sets. Whichever gives the highest accuracy score can be used as the feature set for XGBRegressor. Maybe if I drop a few columns it'll get quicker.

# # Update - Feature Importance
# 
# So today I cam across this [this excellent kernel](https://www.kaggle.com/aleksandradeis/how-to-get-upvotes-for-a-kernel-on-kaggle) by [Aleksandra Deis](https://www.kaggle.com/aleksandradeis).
# 
# She used a bar graph to plot the feature importance and I realized it would be really useful for my project. Let's give it a shot.

# In[ ]:


import matplotlib.pyplot as plt

#plot bar chart with matplotlib
plt.figure(figsize=(17,10))

y_pos = np.arange(len(X.columns))

plt.bar(y_pos, my_model.feature_importances_, align='center', alpha=0.5)
plt.xticks(y_pos, X.columns)
plt.xticks(rotation=90)

plt.xlabel('Features')
plt.ylabel('Feature Importance')

plt.title('Feature importances')

plt.show()


# Woah! That is SO interesting! This is the first time I'm seeing how important data visualization is.
# 
# As we can see there are many columns that don't have any importance at all. I'll drop them and check my model's accuracy. But first I need to divide the graph. I need to make separate graphs using fewer columns at a time so that I can see the names of the unnecessary columns.
# 
# Actually, on another look, it seems: no. useful columns < no. of useless columns. So I'll find out the names of the useful columns and add them as features.

# On second thought, what if I just used plotly instead? I could hover over each column, it'd show me the column name and I could note it down. That would be quicker (Now I gotta Google bar charts using plotly...). You could also use the zoom feature.
# 

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

data = [go.Bar(
            x=y_pos,
            y=my_model.feature_importances_
    )]

#iplot(data)

layout = go.Layout(
    xaxis = go.layout.XAxis(
        tickmode = 'array',
        tickvals = y_pos,
        ticktext = X.columns,
        tickangle = -90
    )
)

fig = go.Figure(
    data = data,
    layout = layout
)

iplot(fig)


# Awesome! :D So let's note down the features (excited!).
# 

# In[ ]:


features = ['bluesy-rose-wallaby-discard','cranky-cardinal-dogfish-ordinal','homey-sepia-bombay-sorted','hasty-blue-sheep-contributor',
            'blurry-wisteria-oyster-master','baggy-mustard-collie-hint','beady-champagne-bullfrog-grandmaster','blurry-flax-sloth-fepid',
           'grumpy-zucchini-kudu-kernel','bluesy-amber-walrus-fepid','hazy-tan-schnauzer-hint','gloppy-turquoise-quoll-goose',
            'snoopy-red-zonkey-unsorted','snappy-brass-malamute-entropy','squeaky-khaki-lionfish-distraction',
            'crappy-pumpkin-saola-grandmaster','wheezy-harlequin-earwig-gaussian','tasty-buff-monkey-learn','dorky-turquoise-maltese-important',
           'hasty-puce-fowl-fepid','stuffy-periwinkle-zebu-discard','breezy-myrtle-loon-discard','woolly-gold-millipede-fimbus',
           'bluesy-amethyst-octopus-gaussian','dorky-cream-flamingo-novice','gimpy-asparagus-eagle-novice','stealthy-yellow-lobster-goose',
           'freaky-olive-insect-ordinal','greasy-scarlet-paradise-goose','pretty-copper-insect-discard','gloppy-buff-frigatebird-dataset',
           'wheezy-lavender-catfish-master','cheeky-pear-horse-fimbus','stinky-olive-kiwi-golden','stealthy-azure-gopher-hint',
            'sleazy-russet-iguana-unsorted','surly-corn-tzu-kernel','woozy-apricot-moose-hint','greasy-magnolia-spider-grandmaster',
           'chewy-bistre-buzzard-expert','wheezy-myrtle-mandrill-entropy','muggy-turquoise-donkey-important','blurry-buff-hyena-entropy']


# That was mucher tougher than I thought. I'm sure there must be an easier way... 
# 
# Let's give it a shot then!

# In[ ]:


y = X_full.target
X = X_full[features]
X_test = X_test_full[features]
X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)


# In[ ]:


# my_model = XGBClassifier(n_estimators=100, learning_rate=0.05)
# my_model.fit(X_train, y_train, 
#              early_stopping_rounds=5, 
#              eval_set=[(X_valid, y_valid)], 
#              verbose=False)
# predictions = my_model.predict(X_valid)
# print("Accuracy Score: " + str(accuracy_score(predictions, y_valid)))

#Accuracy Score: 0.5095271700776288


# In[ ]:


my_model = XGBRegressor(n_estimators=100, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)
predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))


# **WTF???**
# 
# WHYYYYYY???

# In[ ]:


preds = my_model.predict(X_valid)
preds_test = my_model.predict(X_test)
preds_test_rounded = np.around(preds_test,decimals=1)

output = pd.DataFrame({'id': X_test_full.id,
                       'target': preds_test_rounded})
output.to_csv('submission.csv', index=False)

