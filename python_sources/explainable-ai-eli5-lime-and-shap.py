#!/usr/bin/env python
# coding: utf-8

# **Open the Black Box: an Introduction to Model Interpretability with LIME and SHAP - Kevin Lemagnen# Introduction to Model Interpretability**
# 
# (Published here for learning purposes only)
# Source : https://github.com/klemag/pydata_nyc2018-intro-to-model-interpretability
# Thanks Kevin!!

# In[ ]:


import pandas as pd
# Some sklearn tools for preprocessing and building a pipeline. 
# ColumnTransformer was introduced in 0.20 so make sure you have this version
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Our algorithms, by from the easiest to the hardest to intepret.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


# Note: This notebook uses features introduced in Python 3.6 and sklearn 0.20.
# 
# 
# First we'll need a few imports:

# ### The Dataset

# The dataset can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/bank+marketing). It consists of data from marketing campaigns of a Portuguese bank. We will try to build classifiers that can predict whether or not the client targeted by the campaign ended up subscribing to a term deposit (column `y`).

# In[ ]:


df = pd.read_csv('../input/bank-additional-full.csv', sep = ';')


# In[ ]:


df.y.value_counts()


# The dataset is imbalanced, we will need to keep that in mind when building our models!

# In[ ]:


# Get X, y
y = df["y"].map({"no":0, "yes":1})
X = df.drop("y", axis=1)


# Let's look at the features in the X matrix:

# 1. age (numeric)
# 2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5. default: has credit in default? (categorical: 'no','yes','unknown')
# 6. housing: has housing loan? (categorical: 'no','yes','unknown')
# 7. loan: has personal loan? (categorical: 'no','yes','unknown')
# 8. contact: contact communication type (categorical: 'cellular','telephone') 
# 9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14. previous: number of contacts performed before this campaign and for this client (numeric)
# 15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# 16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17. cons.price.idx: consumer price index - monthly indicator (numeric) 
# 18. cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
# 19. euribor3m: euribor 3 month rate - daily indicator (numeric)
# 20. nr.employed: number of employees - quarterly indicator (numeric)

# Note the comment about `duration` feature. We will exclude it from our analysis.

# In[ ]:


X.drop("duration", inplace=True, axis=1)


# In[ ]:


X.dtypes


# In[ ]:


# Some such as default would be binary features, but since
# they have a third class "unknown" we'll process them as non binary categorical
num_features = ["age", "campaign", "pdays", "previous", "emp.var.rate", 
                "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]

cat_features = ["job", "marital", "education","default", "housing", "loan",
                "contact", "month", "day_of_week", "poutcome"]


# 1. We'll define a new `ColumnTransformer` object (new in sklearn 0.20) that keeps our numerical features and apply one hot encoding on our categorical features. That will allow us to create a clean pipeline that includes both features engineering (one hot encoding here) and training the model (a nice way to avoid data leakage)

# In[ ]:


preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 
                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),
                                   cat_features)])


# Now we can define our 4 models as sklearn `Pipeline` object, containing our preprocessing step and training of one given algorithm.

# In[ ]:


# Logistic Regression
lr_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42))])

# Decision Tree
dt_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", DecisionTreeClassifier(class_weight="balanced"))])

# Random Forest
rf_model = Pipeline([("preprocessor", preprocessor), 
                     ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])

# XGBoost
xgb_model = Pipeline([("preprocessor", preprocessor), 
                      # Add a scale_pos_weight to make it balanced
                      ("model", XGBClassifier(scale_pos_weight=(1 - y.mean()), n_jobs=-1))])


# Let's split the data into training and test sets.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)


# We're good to go!

# ## Eli5 to intepret "white box" models

# ### With Logistic Regression

# First let's fine tune our logistic regression and evaluate its performance.

# In[ ]:


gs = GridSearchCV(lr_model, {"model__C": [1, 1.3, 1.5]}, n_jobs=-1, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)


# Let's see our best parameters and score

# In[ ]:


print(gs.best_params_)
print(gs.best_score_)


# In[ ]:


lr_model.set_params(**gs.best_params_)


# In[ ]:


lr_model.get_params("model")


# Now we can fit the model on the whole training set and calculate accuracy on the test set.

# In[ ]:


lr_model.fit(X_train, y_train)


# Generate predictions

# In[ ]:


y_pred = lr_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# Let's use `eli5` to visualise the weights associated to each feature:

# In[ ]:


import eli5
eli5.show_weights(lr_model.named_steps["model"])


# That gives us the weights associated to each feature, that can be seen as the contribution of each feature into predicting that the class will be y=1 (the client will subscribe after the campaign).
# 
# The names for each features aren't really helping though, we can pass a list of column names to `eli5` but we'll need to do a little gymnastics first to extract names from our preprocessor in the pipeline (since we've generated new features on the fly with the one hot encoder)

# In[ ]:


preprocessor = lr_model.named_steps["preprocessor"]


# In[ ]:


ohe_categories = preprocessor.named_transformers_["categorical"].categories_


# In[ ]:


new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]


# In[ ]:


all_features = num_features + new_ohe_features


# Great, so now we have a nice list of columns after processing. Let's visualise the data in a dataframe just for sanity check:

# In[ ]:


pd.DataFrame(lr_model.named_steps["preprocessor"].transform(X_train), columns=all_features).head()


# Looks good!

# In[ ]:


eli5.show_weights(lr_model.named_steps["model"], feature_names=all_features)


# Looks like it's picking principally on whether the month is march or not, the marketting campaign seem to have been more efficient in march?

# We can also use `eli5` to explain a specific prediction, let's pick a row in the test data:

# In[ ]:


i = 4
X_test.iloc[[i]]


# In[ ]:


y_test.iloc[i]


# Our client subsribed to the term deposit after the campaign! Let's see what our model would have predicted and how it would explain it.
# 
# We'll need to first transform our row into the format expected by our model as `eli5` cannot work directly with our pipeline.
# 
# Note: `eli5` actually does support pipeline, but with a limited number of transformations only. In our pipeline it does not support the `passthrough` transformation (which, funny enough, doesn't do anything...)

# In[ ]:


eli5.show_prediction(lr_model.named_steps["model"], 
                     lr_model.named_steps["preprocessor"].transform(X_test)[i],
                     feature_names=all_features, show_feature_values=True)


# ### with a Decision Tree

# `eli5` can also be used to intepret decision trees:

# In[ ]:


gs = GridSearchCV(dt_model, {"model__max_depth": [3, 5, 7], 
                             "model__min_samples_split": [2, 5]}, 
                  n_jobs=-1, cv=5, scoring="accuracy")

gs.fit(X_train, y_train)


# Let's see our best parameters and score

# In[ ]:


print(gs.best_params_)
print(gs.best_score_)


# In[ ]:


dt_model.set_params(**gs.best_params_)


# In[ ]:


dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# For Decision Trees, `eli5` only gives feature importance, which does not say in what direction a feature impact the predicted outcome.

# In[ ]:


eli5.show_weights(dt_model.named_steps["model"], feature_names=all_features)


# Here the most important feature seems to be `nr.employed`. We can also get an explanation for a given prediction, this will calculate the contribution of each feature in the prediction:

# In[ ]:


eli5.show_prediction(dt_model.named_steps["model"], 
                     dt_model.named_steps["preprocessor"].transform(X_test)[i],
                     feature_names=all_features, show_feature_values=True)


# Here the explanation for a single prediction is calculated by following the decision path in the tree, and adding up contribution of each feature from each node crossed into the overall probability predicted.

# `eli5` can also be used to explain black box models, but we will use `Lime` and `SHAP` for our two last models instead.

# ## LIME to generate local intepretations of black box models

# LIME stands for `Local Interpretable Model-Agnostic Explanations`. We can use it with any model we've built in order to explain why it took a specific decision for a given observation. To do so, LIME creates a dataset in the locality of our observation by perturbating the different features. Then it fits a local linear model on this data and uses the weights on each feature to provide an explanation.

# ### with a Random Forest

# In[ ]:


gs = GridSearchCV(rf_model, {"model__max_depth": [10, 15], 
                             "model__min_samples_split": [5, 10]}, 
                  n_jobs=-1, cv=5, scoring="accuracy")

gs.fit(X_train, y_train)


# Let's see our best parameters and score

# In[ ]:


print(gs.best_params_)
print(gs.best_score_)


# In[ ]:


rf_model.set_params(**gs.best_params_)


# In[ ]:


rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# We can look at the features importance with Eli5 first:

# In[ ]:


eli5.show_weights(rf_model.named_steps["model"], 
                  feature_names=all_features)


# We can explain roughly what our model seems to focus on mostly. We also get the standard deviation of feature importance accross the multiple trees in our ensemble.

# ### Let's train our XGB model as well

# In[ ]:


gs = GridSearchCV(xgb_model, {"model__max_depth": [5, 10],
                              "model__min_child_weight": [5, 10],
                              "model__n_estimators": [25]},
                  n_jobs=-1, cv=5, scoring="accuracy")

gs.fit(X_train, y_train)


# Let's see our best parameters and score.

# In[ ]:


print(gs.best_params_)
print(gs.best_score_)
xgb_model.set_params(**gs.best_params_)
xgb_model.fit(X_train, y_train)


# Generate predictions

# In[ ]:


y_pred = xgb_model.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# ### Create an explainer

# In order to explain why the model classifies invidividual observations as class 0 or 1, we are going to use the `LimeTabularExplainer` from the library `lime`, this is the main explainer to use for tabular data. Lime also provides an explainer for text data, for images and for time-series.
# 
# When using the tabular explainer, we need to provide our training set as parameter so that `lime` can compute statistics on each feature, either `mean` and `std` for numerical features, or frequency of values for categorical features. Those statistics are used to scale the data and generate new perturbated data to train our local linear models on.

# In[ ]:


from lime.lime_tabular import LimeTabularExplainer


# The parameters passed to the explainer are:
# - our training set, we need to make sure we use the training set *without* one hot encoding
# - `mode`: the explainer can be used for classification or regression
# - `feature_names`: list of labels for our features
# - `categorical_features`: list of indexes of categorical features
# - `categorical_names`: dict mapping each index of categorical feature to a list of corresponding labels
# - `dicretize_continuous`: will discretize numerical values into buckets that can be used for explanation. For instance it can tell us that the decision was made because distance is in bucket [5km, 10km] instead of telling us distance is an importante feature.

# First, in order to get the `categorical_names` parameter we need to build a dictionary with indexes of categorical values in original dataset as keys and lists of possible categories as values:

# In[ ]:


categorical_names = {}
for col in cat_features:
    categorical_names[X_train.columns.get_loc(col)] = [new_col.split("__")[1] 
                                                       for new_col in new_ohe_features 
                                                       if new_col.split("__")[0] == col]


# In[ ]:


categorical_names


# `Lime` needs the dataset that is passed to have categorical values converted to integer labels that maps to the values in `categorical_names`. For instance, label `0` for the column `2` will map to `divorced`. We will use a custom helper function to do so, that converts data from original to LIME and from LIME to original format.
# 
# That function is going over all categorical features and replacing strings by the correct integer labels, feel free to check `helpers.py`.

# In[ ]:


def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):
    """Converts data with categorical values as string into the right format 
    for LIME, with categorical values as integers labels.

    It takes categorical_names, the same dictionary that has to be passed
    to LIME to ensure consistency. 

    col_names and invert allow to rebuild the original dataFrame from
    a numpy array in LIME format to be passed to a Pipeline or sklearn
    OneHotEncoder
    """

    # If the data isn't a dataframe, we need to be able to build it
    if not isinstance(X, pd.DataFrame):
        X_lime = pd.DataFrame(X, columns=col_names)
    else:
        X_lime = X.copy()

    for k, v in categorical_names.items():
        if not invert:
            label_map = {
                str_label: int_label for int_label, str_label in enumerate(v)
            }
        else:
            label_map = {
                int_label: str_label for int_label, str_label in enumerate(v)
            }

        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)

    return X_lime


# Let's check that it worked:

# In[ ]:


convert_to_lime_format(X_train, categorical_names).head()


# In[ ]:


explainer = LimeTabularExplainer(convert_to_lime_format(X_train, categorical_names).values,
                                 mode="classification",
                                 feature_names=X_train.columns.tolist(),
                                 categorical_names=categorical_names,
                                 categorical_features=categorical_names.keys(),
                                 discretize_continuous=True,
                                 random_state=42)


# Great, our explainer is ready. Now let's pick an observation we want to explain.

# #### Explain new observations

# We'll create a variable called `observation` that contains our ith observation in the test dataset.

# In[ ]:


i = 2
X_observation = X_test.iloc[[i], :]
X_observation


# In[ ]:


print(f"""* True label: {y_test.iloc[i]}
* LR: {lr_model.predict_proba(X_observation)[0]}
* DT: {dt_model.predict_proba(X_observation)[0]}
* RF: {rf_model.predict_proba(X_observation)[0]}
* XGB: {xgb_model.predict_proba(X_observation)[0]}""")


# Let's convert our observation to lime format and convert it to a numpy array.

# In[ ]:


observation = convert_to_lime_format(X_test.iloc[[i], :],categorical_names).values[0]
observation


# In order to explain a prediction, we use the `explain_instance` method on our explainer. This will generate new data with perturbated features around the observation and learn a local linear model. It needs to take:
# - our observation as a numpy array
# - a function that uses our model to predict probabilities given the data (in same format we've passed in our explainer). That means we cannot pass directly our `rf_model.predict_proba` because our pipeline expects string labels for categorical values. We will need to create a custom function `rf_predict_proba` that first converts back integer labels to strings and then calls `rf_model.predict_proba`.
# - `num_features`: number of features to consider in explanation

# In[ ]:


# Let write a custom predict_proba functions for our models:
from functools import partial

def custom_predict_proba(X, model):
    X_str = convert_to_lime_format(X, categorical_names, col_names=X_train.columns, invert=True)
    return model.predict_proba(X_str)


# In[ ]:


lr_predict_proba = partial(custom_predict_proba, model=lr_model)
dt_predict_proba = partial(custom_predict_proba, model=dt_model)
rf_predict_proba = partial(custom_predict_proba, model=rf_model)
xgb_predict_proba = partial(custom_predict_proba, model=xgb_model)


# Let's test our custom function to make sure it generates propabilities properly

# In[ ]:


explanation = explainer.explain_instance(observation, lr_predict_proba, num_features=5)


# Now that we have generated our explanation, we have access to several representations. The most useful one when working in a notebook is `show_in_notebook`.
# 
# 
# On the left it shows the list of probabilities for each class, here the model classified our observation as 0 (non subsribed) with a high probability.
# * If you set `show_table=True`, you will see the table with the most important features for this observation on the right.

# In[ ]:


explanation.show_in_notebook(show_table=True, show_all=False)


# You can also save the explanation to an html file with `save_to_file` to share it.

# In[ ]:


explanation.save_to_file("explanation.html")


# LIME is fitting a linear model on a local perturbated dataset. You can access the coefficients, the intercept and the R squared of the linear model by calling respectively `.local_exp`, `.intercept` and `.score` on your explanation.

# In[ ]:


print(explanation.local_exp)
print(explanation.intercept)
print(explanation.score)


# In[ ]:


# dt_predict_proba


# If your R-squared is low, the linear model that LIME fitted isn't a great approximation to your model, which means you should not rely too much on the explanation it provides.

# In[ ]:


explanation = explainer.explain_instance(observation, dt_predict_proba, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)
print(explanation.score)


# In[ ]:


explanation = explainer.explain_instance(observation, rf_predict_proba, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)
print(explanation.score)


# In[ ]:


explanation = explainer.explain_instance(observation, xgb_predict_proba, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)
print(explanation.score)


# ## More local interpretation with SHAP

# In[ ]:


import shap
# Need to load JS vis in the notebook
shap.initjs() 


# SHAP has a generic explainer that works for any model and a TreeExplainer optimised for tree based models. Here we will focus on the `TreeExplainer` with our XGB model (the hardest to intepret)

# In[ ]:


explainer = shap.TreeExplainer(xgb_model.named_steps["model"])


# In order to compute the shapley values with the tree explainer, we need to call the `shap_values` methods passing a dataset. That can be quite computationally expensive, so we will only pass 1000 samples picked at random.

# In[ ]:


observations = xgb_model.named_steps["preprocessor"].transform(X_train.sample(1000, random_state=42))
shap_values = explainer.shap_values(observations)


# Now we can start visualising our explanations using the `force_plot` function from the shap package passing our first shap_value (we also need to pass `explainer.expected_value` which is the base value).

# In[ ]:


i = 0
shap.force_plot(explainer.expected_value, shap_values[i], 
                features=observations[i], feature_names=all_features)


# This explanation shows how each feature contributes to shifting the prediction from the base value to the output value of the model either by decreasing or increasing the probability of our class.

# We can also visualise all points in our dataset at once with a given class by passing all explanations for that class to `force_plot`

# In[ ]:


shap.force_plot(explainer.expected_value, shap_values,
                features=observations, feature_names=all_features)


# We can see our 1000 samples on the x axis. The y-axis corresponds to the same scale we were looking at before, where blue values corresponds to the probability decreasing, red increasing. Hover with your mouse on a point to see the main features impacting a given observation. You can also use the drop down on the left to visualise the impact of specific features, for example duration only.

# Another interesting plot that we can generate with SHAP is the `summary_plot`, it can be seen as a feature importance plot with more meaningful insights. Below we're plotting the summary plot for class 1 on the whole subset.
# The colour corresponds to the value of the feature and the x axis corresponds to the SHAP value, meaning the impact on the probability. 

# In[ ]:


shap.summary_plot(shap_values, features=observations, feature_names=all_features)


# That's better than the built-in feature importance on RandomForest because not only we can see what features are important but also how they affect our predictions.

# In[ ]:


shap.dependence_plot("nr.employed", shap_values, 
                     pd.DataFrame(observations, columns=all_features))


# # Intepreting models with non tabular data

# The tools we have seen above also work with text data and images. There are plenty of examples available online for text-data. Here we will just demonstrate how to use `Lime` to explain an image classifier.

# ## Interpreting image classifiers

# Lime can also be used to explain decisions made for image classification. 
# 
# In this example we will use the pretrained `InceptionV3` model available with Keras. Lime is quite slow with images, so it's wiser to stick to a "shallow" deep learning model.

# In[ ]:


from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array


# Let's create a new instance of InceptionV3

# In[ ]:


model = InceptionV3()


# Now we'll load a picture of a toucan, we need to make sure we load it at the good size for inception, here 229*229

# In[ ]:


image_raw = load_img("data/toucan.jpg", target_size=(229, 229))
image_raw


# We need to process the image to get a numpy array compatible with our model. Here we simply loads it to an array, reshape it and use the preprocess_input method provided by Keras that ensures all the preprocessing steps are made for us.

# In[ ]:


# Convert to numpy array, reshape and preprocess
image = img_to_array(image_raw)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)


# Now that our image is ready, generate predictions by using `.predict` as usual.

# In[ ]:


predictions = model.predict(image)


# You can check what labels your predictions correspond to by calling the function `decode_predictions` on your predictions. By default it returns the 5 more likely predictions

# In[ ]:


decode_predictions(predictions)


# Great, we predicted a toucan with a probability of 99%, that's promising!

# Remember that LIME needs the indices of the class we are interested in. Execute the cell bellow to get the indices corresponding to the 5 most probably classes we predicted above. Those indices correspond to the classes used in the ImageNet dataset that was used to train our model.

# In[ ]:


model.predict(image).argsort()[0, -5:][::-1]


# Here the toucan corresponds to index 96, the school bus to index 779, etc..

# Let's get started. First import the `LimeImageExplainer` and instantiate a new explainer

# In[ ]:


from lime.lime_image import LimeImageExplainer


# In[ ]:


explainer = LimeImageExplainer()


# The explainer is the same as before, we call `explain_instance` to generate a new explanation. We need to provide:
# - our observation: here the first row of our numpy matrix (that has only one row since we only have one image)
# - our predict function, we can simply use the one from our model here
# - `top_labels` the number of classes to explain. Here our model generate probabilities for more than a 1000 classes (and we looked at the five first). We do not want LIME to generate local models to explain each of those classes. As lime is pretty slow with images, let's only ask for the explanation to our two main classes, toucan and school bus
# - `num_samples`: the number of new datapoints to create to fit a linear model, let's set it to 1000
# 
# *WARNING*: that will be slow. 

# In[ ]:


explanation = explainer.explain_instance(image[0], model.predict, 
                                         top_labels=2, num_samples=100,
                                         random_seed=42)


# In[ ]:


from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt


# First let's check the explanation for the predicted class `toucan`. That corresponds to label 96 in the ImageNet classes. We need to use the method `get_image_and_mask` on our explanation object with the following parameters:
# - index of the class to explain. We'll start with the index of the main class predicted, that was 96
# - positive_only: in order to show the part of the image that contribute positively to this class being selected
# - num_features: number of superpixels to use. LIME breaks down our image into a set of superpixels, each containing several pixels. Those superpixels are equivalent to `features` in tabular data.
# - hide_rest: to hide the rest of the image
# 
# That returns a new image and a mask as numpy arrays. You can then use `mark_boundaries` to show the image together with the mask.

# In[ ]:


temp, mask = explanation.get_image_and_mask(96, positive_only=True, num_features=5, hide_rest=True)
# plot image and mask together
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# What feature do you expect to be the most important in that decision? Plot the image with only the main feature (`num_features=1`)

# In[ ]:


temp, mask = explanation.get_image_and_mask(96, positive_only=True, num_features=1, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# The second class predicted by our model was a bus (label 779), set `positive_only=False` in order to see what features contributed positively and negatively to that decision. What do you see?

# In[ ]:


temp, mask = explanation.get_image_and_mask(779, positive_only=False, num_features=8, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))


# Great, now you can try to change the number of features you're looking at and deactivate `positive_only` in order to see features that contribute negatively to the class. You can also look at other classes or try other pictures.
