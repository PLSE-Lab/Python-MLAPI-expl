#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The goal of this notebook is to train a multi-class classifier to identify fish based on their body measurements. This was just a fun way to practice multi-class classification.
# 
# ![](https://i.pinimg.com/originals/db/6a/94/db6a94f49205d32b2404cbb6cf562721.jpg)

# In[ ]:


# Import relevant modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style ='white', palette = 'colorblind')


# These data have six different body measurements for seven fish species.

# In[ ]:


# Load the Data
df = pd.read_csv("../input/fish-market/Fish.csv")
df.head(10)


# In[ ]:


print("The species were {}.".format(list(set(df.Species))))
print('There are {} observations in our dataframe.'.format(len(df)))


# Since there aren't many features in this dataset, I will first take a look at their distributions.

# In[ ]:


# Plotting Distributions of Columns
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
fig, ax = plt.subplots(2, 3, sharey = True, figsize = [16, 10])
fig.suptitle('Distribution of Fish Measurements', fontsize = 26)
fig.subplots_adjust(hspace = 0.2)
i = 0
j = 0
for col in df.iloc[:, 1:]:
    ax[i, j].hist(df[col], bins = 20, histtype = 'bar')
    ax[i, j].set_title(col, fontsize = 20)
    ax[i, j].text(0.65, 0.9, r"$\mu$ = {: .1f}".format(df[col].mean()), transform = ax[i, j].transAxes, fontsize = 16)
    ax[i, j].text(0.65, 0.8, r"$\sigma$ = {: .1f}".format(df[col].std()), transform = ax[i, j].transAxes, fontsize = 16)
    if j < 2:
        j += 1
    else:
        i += 1
        j -= 2


# The distribution for weight is a bit skewed, which isn't inherently a problem, but I will take a closer look at the summary statistics.

# In[ ]:


df.describe().round(2)


# Here we see that the minimum weight in the sample is 0.00, which is obviously impossible. One of the fish had an incorrect weight entered. I am going to replace the zero value with the median weight for all of the fish.

# In[ ]:


# Replace Zeroes with Median Weight of all fish
median_weight = df['Weight'][df['Weight']!=0].median()
df['Weight'] = df['Weight'].mask(df['Weight'] == 0, median_weight)


# Now we can check again to make sure it is all in order:

# In[ ]:


df['Weight'].describe().round(2)


# # Some Feature Engineering
# 
# Let's take a look at how correlated our features are with each other. Because we have several columns in the dataframe with different measurements of length, these may all be highly collinear, and it therefore might not be necessary to keep them all in our model. Since our dataframe only has 159 observations in it, it wouldn't hurt to reduce the number of features we include in our classifier, if possible.

# In[ ]:


# Correlation Matrix of Features
corrs = df.corr().round(2)
plt.figure(figsize = (10, 8))
sns.heatmap(corrs, cmap = 'Greys')


# As I suspected, the three length metrics are highly correlated with one another. I therefore chose to only keep `Length3`, which is the diagonal length of the fish.

# In[ ]:


df = df.drop(['Length1', 'Length2'], axis = 1)


# I then chose to construct a couple of other features to that may make the model a bit more discriminant for classifying between the seven types of fish. The new features are width-for-height (the width divided by the height) and the weight-for-height (the weight divided by the height). I tried other combinations but these were the most successful.

# In[ ]:


df['wid_for_height'] = df['Width'] / df['Height'] 
df['weight_for_height'] = df['Weight'] / df['Height']


# # Training the Model
# 
# Now I will split the data into training and testing sets, using an 80-20 split.

# In[ ]:


x = df.iloc[:, 1:]
y = df.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 123)


# I tried out a few different classification models, including K-Nearest Neighbors, KMeans, and multinomial logistic regression. The multinomial logit was easily the most successful, so I will only present those results here.

# In[ ]:


#Multinomial Logit
log = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 20000)
log.fit(x_train, y_train)
preds = log.predict(x_test)
species_list = log.classes_.tolist()
conf_mx = confusion_matrix(y_test, preds, species_list)
print("\t\t Model Metrics")
print("Precision: \t", precision_score(y_test, preds, average = 'weighted').round(2))
print("Recall: \t", recall_score(y_test, preds, average = 'weighted').round(2))
print("F1 Score: \t", f1_score(y_test, preds, average = 'weighted').round(2))


# In the end, 89% of my predictions were of the correct class, and 88% of the true classes were correctly identified. Not too bad for a model with very basic information!    
# 
# As a frame of reference, let's do two little experiments to gauge how well this model is doing. First, how do these compare to just randomly guessing one of the seven fish? And second, what if I only guessed that every fish was a Perch, which was the most common fish in the sample?

# In[ ]:


randompreds = np.random.choice(list(set(df['Species'])), size = len(y_test))
all_perch = np.full(len(y_test), fill_value = 'Perch')

print("If I randomly guessed, the precision score would be {}.".format(precision_score(y_test, randompreds, average = 'weighted')))
print("If I guessed all fish were perch, it would be {}.".format(precision_score(y_test, all_perch, average = 'weighted')))


# So at least we know the classifier is doing a lot better than naive guessing! But it is also important to know if there are specific classes that the model is struggling to predict.

# In[ ]:


# Matrix of errors
row_sums = conf_mx.sum(axis = 1, keepdims = True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

fig = plt.figure(figsize = (10, 8))
fig.tight_layout()
ax = fig.add_subplot(111)
cax = ax.matshow(norm_conf_mx, cmap = 'gist_heat_r')
plt.title('Proportion of Incorrect Predictions\nfor Fish Classifier', fontsize = 16)
fig.colorbar(cax)
plt.gca().xaxis.tick_bottom()
ax.set_xticklabels([''] + species_list, rotation = 30)
ax.set_yticklabels([''] + species_list)
plt.xlabel('Predicted Class', fontsize = 16)
plt.ylabel('Actual Class', fontsize = 16)


# It is clear that there are two fish that the model is struggling to identify. 100% of whitefish were misclassified as perch, but this is not too surprising because there were only 6 whitefish in the entire sample and only 1 in the test data.    
# 
# But the model also struggled alot to correctly classify Roach, which was the third largest group of fish in the dataset.  Here, the model incorrectly classified roaches as perch for about 60% of the cases in which the fish was actually a roach.

# # Conclusion
# 
# Based on pretty basic data, the model was reasonably strong at classifying the fish into their proper categories. It was pretty bad at classifying roaches or whitefish though. So, if this model was needed for that purpose, it would require more fine-tuning before it would be ready to be used for prediction. Thanks for joining me in this little multi-classification problem and I hope you enjoyed it!
