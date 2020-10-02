#!/usr/bin/env python
# coding: utf-8

# # Exercises
# 
# ## Set Up
# 
# We've had our fun and learned a fair amount with the Taxi data. But now you have enough tools to put together compelling solutions to real-world problems. The following scenario will require you to pick the right techniques for each part of your data science project. Along the way, you'll use SHAP values along with your other insights tools.
# 
# **The questions below give you feedback on your work by using some checking code. Run the following cell to set up our feedback system.**

# In[ ]:


import sys
sys.path.append('../input/ml-insights-tools')
from ex4 import *
print("Setup Complete")


# ## The Scenario
# A hospital has struggled with "readmissions," where they release a patient before the patient has recovered enough, and the patient returns with health complications. 
# 
# The hospital wants your help identifying patients at highest risk of being readmitted. Doctors (rather than your model) will make the final decision about when to release each patient; but they hope your model will highlight issues the doctors should consider when releasing a patient.
# 
# The hospital has given you relevant patient medical information.  Here is a list of columns in the data:
# 

# In[ ]:


import pandas as pd
data = pd.read_csv('../input/hospital-readmissions/train.csv')


# ## Step 1:
# You have built a simple model, but the doctors say they don't know how to evaluate a model, and they'd like you to show them some evidence the model is doing something in line with their medical intuition. Create any graphics or tables that will show them a quick overview of what the model is doing?
# 
# They are very busy. So they want you to condense your model overview into just 1 or 2 graphics, rather than a long string of graphics.
# 
# We'll start after the point where you've built a basic model. Just run the following cell to build the model called `my_model`.

# In[ ]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)


# In[ ]:


from sklearn import metrics
import numpy as np

my_preds = my_model.predict(val_X)
# A random baseline to compare our predictions to. Flip a coin for each patient.
baseline_preds = np.random.rand(len(val_y))

def report_summary_stats(predictions):
    # Round to binary 0/1 predictions
    pred_labels = predictions.round()
    acc = metrics.accuracy_score(val_y, pred_labels)
    pre = metrics.precision_score(val_y, pred_labels)
    rec = metrics.recall_score(val_y, pred_labels)
    print("Accuracy = {:.1%}, Precision = {:.1%}, Recall = {:.1%}".format(
        acc, pre, rec
    ))
    
print("** Our model **")
report_summary_stats(my_preds)
print("** Random baseline **")
report_summary_stats(baseline_preds)


# In[ ]:


import matplotlib.pyplot as plt

def plot_precision_recall_curve(preds):
    plt.figure(figsize=(12, 8))
    precision, recall, _ = metrics.precision_recall_curve(val_y, preds)

    average_precision = metrics.average_precision_score(val_y, preds)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: Average Precision={0:0.2f}'.format(
              average_precision))
    
plot_precision_recall_curve(my_preds)


# In[ ]:


# Baseline precision-recall curve
plot_precision_recall_curve(baseline_preds)

