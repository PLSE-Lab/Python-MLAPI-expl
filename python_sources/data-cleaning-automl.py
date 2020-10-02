#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import category_encoders as ce

df = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")


# In[ ]:


## clean titles/target variable

# only use most popular titles
popular_titles = df['Q6'].value_counts() > 500
popular_titles = popular_titles[popular_titles == True].index.tolist()

df = df[df.Q6.isin(popular_titles)]

# remove rows with unlikely job titiles
indexNames = df[df.Q6.isin({"Student", "Other", "Not Employed"})].index
df = df.drop(indexNames)


# In[ ]:


## clean & transform input variables

# remove rows w/out salary info
indexNames = df[-df.Q9.str.contains('0', regex=True, na=False)].index
df = df.drop(indexNames)

## split out predictors & target variable

# all the questions we'd like to use: Q10+ are 
# multipart & encoded sepeartely
other_variables = {"Q6", "Q4", "Q7", "Q8", "Q9", 
                 "Q10"}
df_subsample = df.filter(other_variables)

# split into predictor & target variables
X = df_subsample.drop("Q6", axis=1)
y = df_subsample["Q6"]


# In[ ]:


#df.head()

# Q6 = job title
# Q4 = level of education (lab encoder)
# Q7 = industry employment (one hot)
# Q8 = years of experience (make numeric)
# Q9 = comp, want to drop "I do not wish..." (make numeric)
# Q10 = degree to which business uses ML (lab encoder)
# Q11 ... = all parts to 1-hot, drop other_text
# Q12 ... = tools used in job (one hot)
# Q15 ... = cloud services (one hot)
# Q16 ... = programming languages (one hot)
# Q19 ...  = deep learning frameworks (one hot)


# In[ ]:


# mulitple choice questions requiring one-hot encoding
one_hot_qs = df[df.columns[df.nunique() <= 1]].filter(regex="Q11")

for i in {"Q12", "Q15", "Q16", "Q19"}:
    new_data = df[df.columns[df.nunique() <= 1]].filter(regex=i)
    
    one_hot_qs = pd.concat([one_hot_qs.reset_index(drop=True), 
                            new_data.reset_index(drop=True)], 
                           axis=1)
    
# Q7 = industry employment (one hot)
encoder = ce.OneHotEncoder()

one_hot_qs = encoder.fit_transform(one_hot_qs, y)


# In[ ]:


# label encoding
encoder = ce.OrdinalEncoder(cols=["Q4","Q8","Q9","Q10"])

encoder.fit(X, y)
X_cleaned = encoder.transform(X)

# one hot encoding
encoder = ce.OneHotEncoder(cols=["Q7"])

X_cleaned = encoder.fit_transform(X_cleaned, y)
X_cleaned = pd.concat([one_hot_qs.reset_index(drop=True), 
                        X_cleaned.reset_index(drop=True)], 
                       axis=1)

# encode target variable
encoder = ce.OrdinalEncoder()
y_encoded = encoder.fit_transform(y)


# In[ ]:


# Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y,
                                                    train_size=0.80, test_size=0.20)

# vesion w/ y numerically encoded
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_cleaned, y_encoded,
                                                    train_size=0.80, test_size=0.20)


# In[ ]:


# look at classes & imbalances
y.value_counts()


# # Vanilla XGBoost

# In[ ]:


from xgboost import XGBClassifier

my_model = XGBClassifier()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X_train, y_train, verbose=False)


# In[ ]:


from sklearn.metrics import auc, accuracy_score, confusion_matrix

# make predictions
xgb_predictions = my_model.predict(X_test)

# accuracy... but this is an imbalanced multi-class prob
print(accuracy_score(y_test, xgb_predictions))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
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
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
    fig.set_figheight(15)
    fig.set_figwidth(15)
    return ax


# In[ ]:


# Plot non-normalized confusion matrix
plot_confusion_matrix(xgb_predictions, y_test, 
                      classes=unique_labels(y_test),
                      normalize=True,
                      title='Confusion matrix, without normalization')


# In[ ]:


from xgboost import plot_importance

# plot feature importance
plt.rcParams["figure.figsize"] = (14, 14)

plot_importance(my_model)

# Q4 = level of education (lab encoder)
# Q7 = industry employment (one hot)
# Q8 = years of experience (make numeric)
# Q9 = comp, want to drop "I do not wish..." (make numeric)
# Q10 = degree to which business uses ML (lab encoder)
# Q11 ... = role at work (1 = Analyze and understand data to influence product or business decisions,
# 5 = Do research that advances the state of the art of machine learning)
# Q12 ... = tools used in job (one hot)
# Q15 ... = cloud services (one hot)
# Q16 ... = programming languages (one hot)
# Q19 ... = deep learning frameworks (one hot)


# # TPOT
# 
# Documentation: https://epistasislab.github.io/tpot/

# In[ ]:


from tpot import TPOTClassifier

tpot = TPOTClassifier(generations=8, population_size=20, 
                      verbosity=2, early_stop=2)
tpot.fit(X_train_encoded, y_train_encoded)

print("Accuracy is {}%".format(tpot.score(X_test_encoded, y_test_encoded)*100))


# In[ ]:


tpot.export('tpot_pipeline.py')


# In[ ]:


get_ipython().system('cat tpot_pipeline.py')


# In[ ]:


tpot_predictions = tpot.predict(X_test_encoded)

# Plot non-normalized confusion matrix
plot_confusion_matrix(tpot_predictions, y_test_encoded["Q6"], 
                      classes=unique_labels(y_test),
                      normalize=True,
                      title='Confusion matrix, without normalization')


# # H20.ai AutoML
# 
# Documentation: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

# In[ ]:


import h2o
from h2o.automl import H2OAutoML

h2o.init()


# In[ ]:


# convert data to h20Frame
train_data = h2o.H2OFrame(X_train)
test_data = h2o.H2OFrame(list(y_train))

train_data = train_data.cbind(test_data)

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=10, seed=1)
# y is the name of the H20 frame with the 
aml.train(y="C1", training_frame=train_data)


# In[ ]:


# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

# The leader model is stored here
aml.leader


# In[ ]:


# convert data to h20Frame
train_data = h2o.H2OFrame(X_test)
test_data = h2o.H2OFrame(list(y_test))

testing_data = train_data.cbind(test_data)

h2o_predictions = aml.predict(testing_data)


# In[ ]:


h2o_predictions


# In[ ]:


h2o_predictions = h2o_predictions["predict"].as_data_frame()

# Plot non-normalized confusion matrix
plot_confusion_matrix(h2o_predictions, y_test, 
                      classes=unique_labels(y_test),
                      normalize=True,
                      title='Confusion matrix, without normalization')


# # Overall accuracy

# In[ ]:


print("Vanilla XGBoost:")
print( (y_test, xgb_predictions))
print("TPOT:")
print(accuracy_score(y_test_encoded, tpot_predictions))
print("H20 AutoML:")
print(accuracy_score(y_test, h2o_predictions))

